"""Test anti-thrashing: compaction should not fire repeatedly on stale usage data."""
import sys, os, asyncio, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent import Agent
from pi_agent.agent_types import (
    AssistantMessage, TextBlock, TextContent, make_user_message,
    Usage, CompactionSettings, DEFAULT_COMPACTION_SETTINGS,
    ToolResultMessage, ToolCallBlock,
)
from pi_agent.llm import Model, LlmProvider, StreamEvent
from pi_agent.session import Session, build_session_context
from pi_agent.utils import estimate_context_tokens


class StaleUsageProvider(LlmProvider):
    """
    Returns a single assistant message with very high usage,
    simulating a pre-compaction assistant whose usage reflects
    a much larger context than what currently exists.
    """
    def __init__(self):
        self._called = False

    async def stream(self, model, context, *, api_key=None, signal=None,
                     headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        msg = AssistantMessage(
            content=[TextBlock(type="text", text="OK")],
            provider=model.provider, model=model.id,
            stopReason="end_turn",
            usage=Usage(input=500000, output=1000, totalTokens=501000),  # huge!
            timestamp=ts,
        )
        yield StreamEvent(type="start", partial=msg.model_copy(deep=True))
        yield StreamEvent(type="done", partial=msg)

    async def complete(self, model, context, *, api_key=None, signal=None,
                       headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        return AssistantMessage(
            content=[TextBlock(type="text", text="done")],
            provider=model.provider, model=model.id,
            stopReason="end_turn",
            usage=Usage(input=10, output=5, totalTokens=15),
            timestamp=ts,
        )


async def test_anti_thrashing_skips_stale_anchor():
    """After compaction, kept pre-compaction assistant has stale usage -> skip."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.jsonl")
    session = await Session.create(path, cwd=tmpdir)

    model = Model(id="mock", provider="mock", contextWindow=128000, maxTokens=4096)
    mock = StaleUsageProvider()
    settings = CompactionSettings(enabled=True, reserveTokens=16000, keepRecentTokens=20000)

    # Setup: manually create a session that has a compaction entry
    # plus a kept pre-compaction assistant with stale usage

    # 1. A pre-compaction assistant message (old, with huge usage)
    pre_comp_assistant = AssistantMessage(
        content=[TextBlock(type="text", text="pre-compaction work")],
        provider="mock", model="mock",
        stopReason="end_turn",
        usage=Usage(input=90000, output=1000, totalTokens=91000),  # stale, large
        timestamp=1000000,  # old
    )

    # 2. A user message (the "kept" message after compaction boundary)
    kept_user = make_user_message("continue from here", timestamp=2000000)

    # 3. A compaction entry
    from pi_agent.agent_types import SessionEntry
    from datetime import datetime, timezone

    # Append in order: pre_comp user, pre_comp assistant, kept user, compaction
    await session.append_message(make_user_message("original request", timestamp=500000))
    await session.append_message(pre_comp_assistant)
    await session.append_message(kept_user)

    # Now do a real compaction on these entries to create a valid compaction entry
    agent = Agent(model=model, provider=mock, session=session,
                  compaction_settings=settings)

    # Restore messages so agent has them
    ctx = build_session_context(session.entries)
    agent._messages = list(ctx.messages)

    # Manually trigger one compaction
    # First check - should trigger
    result1 = agent.check_compaction()
    print(f"  Before compaction: check_compaction() = {result1}")

    # Do the actual compaction
    compact_result = await agent.compact()
    if compact_result:
        print(f"  Compaction done: tokens_before={compact_result.tokens_before}")

    # After compaction, messages should be rebuilt
    ctx2 = build_session_context(session.entries)
    agent._messages = list(ctx2.messages)

    # Now check again - anti-thrashing should kick in because the last
    # successful assistant in context is pre_comp_assistant (kept after compaction),
    # and its timestamp is before the compaction entry
    result2 = agent.check_compaction()
    print(f"  After compaction: check_compaction() = {result2}")

    # Should skip because anchor is pre-compaction stale usage
    # But note: if the session doesn't have a compaction entry with first_kept_entry_id,
    # the anti-thrashing won't apply. Let's verify the compaction entry is there.
    comp_entries = [e for e in session.entries if e.type == "compaction"]
    print(f"  Compaction entries in session: {len(comp_entries)}")
    assert len(comp_entries) >= 1, "Should have compaction entry"

    # The key assertion: after real compaction, a second check should NOT trigger
    # because the anchor assistant (pre_comp_assistant) is from before compaction
    assert not result2, \
        f"Anti-thrashing should prevent second compaction: " \
        f"anchor is pre-compaction with stale usage"

    print("Test PASSED: anti-thrashing prevents repeated compaction")

    os.remove(path)
    os.rmdir(tmpdir)


async def test_anti_thrashing_allows_fresh_anchor():
    """After compaction, a NEW assistant with fresh usage should still trigger."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.jsonl")
    session = await Session.create(path, cwd=tmpdir)

    model = Model(id="mock", provider="mock", contextWindow=128000, maxTokens=4096)
    settings = CompactionSettings(enabled=True, reserveTokens=16000, keepRecentTokens=20000)

    # Create a pre-compaction message
    old_assistant = AssistantMessage(
        content=[TextBlock(type="text", text="old")],
        provider="mock", model="mock",
        stopReason="end_turn",
        usage=Usage(input=500, output=100, totalTokens=600),
        timestamp=1000000,
    )
    await session.append_message(make_user_message("original", timestamp=500000))
    await session.append_message(old_assistant)
    await session.append_message(make_user_message("continued", timestamp=3000000))

    mock1 = StaleUsageProvider()
    agent = Agent(model=model, provider=mock1, session=session,
                  compaction_settings=settings)

    ctx = build_session_context(session.entries)
    agent._messages = list(ctx.messages)

    # First compaction
    result1 = agent.check_compaction()
    print(f"  Before: check_compaction() = {result1}")
    await agent.compact()

    # Now simulate a NEW assistant message AFTER compaction
    # Usage must exceed contextWindow - reserveTokens (128000 - 16000 = 112000)
    new_assistant = AssistantMessage(
        content=[TextBlock(type="text", text="fresh work")],
        provider="mock", model="mock",
        stopReason="end_turn",
        usage=Usage(input=120000, output=5000, totalTokens=125000),  # above threshold
        timestamp=int(time.time() * 1000),  # NOW, post-compaction
    )
    await session.append_message(make_user_message("more work"))
    await session.append_message(new_assistant)

    # Rebuild
    ctx = build_session_context(session.entries)
    agent._messages = list(ctx.messages)

    # Debug: inspect context
    raw = [m.model_dump(by_alias=True, exclude_none=True, mode="json") for m in ctx.messages]
    print(f"  Context messages ({len(ctx.messages)}):")
    for m in ctx.messages:
        r = getattr(m, 'role', '?')
        u = getattr(m, 'usage', None)
        ts = getattr(m, 'timestamp', 0)
        tok = u.total_tokens if hasattr(u, 'total_tokens') else (getattr(u, 'totalTokens', '?') if u else '?')
        print(f"    {r}: ts={ts}, tokens={tok}")
    tokens = estimate_context_tokens(raw)
    print(f"  Estimated tokens: {tokens}, threshold: {model.context_window - settings.reserve_tokens}")

    anchor_ts = Agent._find_estimate_anchor_timestamp(raw)
    comp_entries_debug = [e for e in session.entries if e.type == "compaction"]
    if comp_entries_debug:
        comp_ts = Agent._entry_timestamp_ms(comp_entries_debug[-1])
        print(f"  anchor_ts={anchor_ts}, comp_ts={comp_ts}, anchor<=comp={anchor_ts is not None and anchor_ts <= comp_ts}")

    # Check again - should return True because the anchor is post-compaction
    result2 = agent.check_compaction()
    print(f"  After fresh response: check_compaction() = {result2}")
    if not result2:
        # Print more debug
        for e in session.entries:
            print(f"    Entry: type={e.type}, ts={e.timestamp}")
        # Let's check what last_compaction and firstKeptEntryId look like
        last_comp = None
        for e in reversed(session.entries):
            if e.type == "compaction":
                last_comp = e
                break
        if last_comp:
            print(f"    Last compaction: firstKeptEntryId={last_comp.first_kept_entry_id}, ts={last_comp.timestamp}")
            # Check if firstKeptEntryId matches any entry
            for e in session.entries:
                if e.id == last_comp.first_kept_entry_id:
                    print(f"    Found matching entry: type={e.type}, ts={e.timestamp}")

    assert result2, "Should trigger: anchor is fresh, post-compaction"

    print("Test PASSED: fresh anchor still triggers compaction")

    os.remove(path)
    os.rmdir(tmpdir)


async def main():
    print("=" * 50)
    print("Anti-Thrashing Tests")
    print("=" * 50 + "\n")

    await test_anti_thrashing_skips_stale_anchor()
    await test_anti_thrashing_allows_fresh_anchor()
    print()
    print("ALL ANTI-THRASHING TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
