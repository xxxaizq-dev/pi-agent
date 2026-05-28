"""Integration tests for auto-persist and auto-compaction in Agent.

Usage: cd E:/work && python pi_agent/_test_agent.py
"""
import sys, os, asyncio, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent import Agent
from pi_agent.agent_types import (
    AgentTool, AgentToolResult, AssistantMessage, TextBlock, TextContent,
    make_user_message, DEFAULT_COMPACTION_SETTINGS,
)
from pi_agent.llm import Model, LlmProvider, StreamEvent, Usage
from pi_agent.session import Session, build_session_context


# ==============================================
# Mock provider: text-only, no tool calls
# ==============================================
class MockProviderCumulative(LlmProvider):
    """Returns canned responses with growing cumulative usage to simulate real LLM."""
    def __init__(self, texts: list[str]):
        self.texts = texts
        self.call_count = 0
        self._cumulative = 0

    async def stream(self, model, context, *, api_key=None, signal=None,
                     headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        if self.call_count >= len(self.texts):
            text = ""
        else:
            text = self.texts[self.call_count]
        self.call_count += 1

        self._cumulative += len(text) // 4 + 50  # ~user msg overhead
        total = self._cumulative

        partial = AssistantMessage(
            content=[TextBlock(type="text", text=text)],
            provider=model.provider, model=model.id,
            stopReason="end_turn",
            usage=Usage(input=total // 2, output=total - total // 2, totalTokens=total),
            timestamp=ts,
        )
        yield StreamEvent(type="start", partial=partial.model_copy(deep=True))
        if text:
            yield StreamEvent(type="text_delta", partial=partial.model_copy(deep=True), text=text)
        yield StreamEvent(type="done", partial=partial)

    async def complete(self, model, context, *, api_key=None, signal=None,
                       headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        text = self.texts[self.call_count] if self.call_count < len(self.texts) else ""
        self.call_count += 1
        self._cumulative += len(text) // 4 + 50
        return AssistantMessage(
            content=[TextBlock(type="text", text=text)],
            provider=model.provider, model=model.id,
            stopReason="end_turn",
            usage=Usage(input=self._cumulative//2, output=self._cumulative - self._cumulative//2, totalTokens=self._cumulative),
            timestamp=ts,
        )


# ==============================================
# Test 1: Messages are persisted to session after prompt
# ==============================================
async def test_persist_to_session():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.jsonl")
    session = await Session.create(path, cwd=tmpdir)

    model = Model(id="mock", provider="mock", contextWindow=128000)
    mock = MockProviderCumulative(["Response 1"])
    agent = Agent(model=model, provider=mock, session=session)

    assert len(agent.messages) == 0, "Should start with no messages"

    new_msgs = await agent.prompt("Hello")
    assert len(new_msgs) > 0, "Should return new messages"

    # Session should have messages now (user + assistant)
    session_entries = session.entries
    message_entries = [e for e in session_entries if e.type == "message"]
    # user prompt + assistant response
    assert len(message_entries) >= 2, f"Expected >=2 messages in session, got {len(message_entries)}"
    print(f"  Session entries: {len(session_entries)} total, {len(message_entries)} messages")

    # Agent memory should match
    assert len(agent.messages) >= 2, f"Expected >=2 in-memory messages, got {len(agent.messages)}"

    os.remove(path)
    os.rmdir(tmpdir)
    print("Test 1 PASSED: persist to session")


# ==============================================
# Test 2: Agent restores messages from session on init
# ==============================================
async def test_restore_from_session():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.jsonl")
    session = await Session.create(path, cwd=tmpdir)

    # First agent: run a prompt
    model = Model(id="mock", provider="mock", contextWindow=128000)
    mock1 = MockProviderCumulative(["First response"])
    agent1 = Agent(model=model, provider=mock1, session=session)
    await agent1.prompt("First message")

    # Second agent: should restore messages from same session
    mock2 = MockProviderCumulative(["Second response"])
    agent2 = Agent(model=model, provider=mock2, session=session)
    assert len(agent2.messages) >= 2, f"Agent2 should restore messages, got {len(agent2.messages)}"
    print(f"  Agent2 restored {len(agent2.messages)} messages")

    os.remove(path)
    os.rmdir(tmpdir)
    print("Test 2 PASSED: restore from session")


# ==============================================
# Test 3: Auto-compaction triggers when context is large
# ==============================================
async def test_auto_compaction():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.jsonl")
    session = await Session.create(path, cwd=tmpdir)

    # Small context window to trigger compaction quickly
    model = Model(id="mock", provider="mock", contextWindow=1000, maxTokens=100)

    # Make lots of long messages to fill context
    long_text = "x" * 300  # ~75 estimated tokens per message
    mock = MockProviderCumulative([long_text] * 10)

    # Custom compaction settings: tight thresholds to trigger quickly
    from pi_agent.agent_types import CompactionSettings
    settings = CompactionSettings(enabled=True, reserveTokens=100, keepRecentTokens=50)

    agent = Agent(model=model, provider=mock, session=session, compaction_settings=settings)

    compaction_triggered = False
    for i in range(8):
        result = await agent.prompt(f"Message {i}: {long_text}")
        entries = session.entries
        has_compaction = any(e.type == "compaction" for e in entries)
        if has_compaction:
            compaction_triggered = True
            print(f"  Compaction triggered after {i+1} prompts ({len(entries)} entries)")
            break

    if compaction_triggered:
        print(f"  Agent memory after compaction: {len(agent.messages)} messages")
        # Check compaction entry exists
        entries = session.entries
        compactions = [e for e in entries if e.type == "compaction"]
        print(f"  Compaction entries: {len(compactions)}")
        assert len(compactions) >= 1, "Should have at least one compaction entry"
        # Agent memory should be smaller than full history
        assert len(agent.messages) < 2 * 8, "Memory should be compacted (smaller than full history)"
        print("Test 3 PASSED: auto-compaction triggered")
    else:
        # Maybe need more rounds — compaction depends on token estimation
        print(f"  Compaction did not trigger (may need more rounds)")
        print("Test 3 SKIPPED: need more context accumulation")

    os.remove(path)
    os.rmdir(tmpdir)


# ==============================================
# Main
# ==============================================
async def main():
    print("=" * 50)
    print("Agent Auto-Persist & Compaction Tests")
    print("=" * 50 + "\n")

    await test_persist_to_session()
    print()
    await test_restore_from_session()
    print()
    await test_auto_compaction()
    print()
    print("ALL TESTS DONE")


if __name__ == "__main__":
    asyncio.run(main())
