"""Test overflow detection and recovery in Agent."""
import sys, os, asyncio, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent import Agent
from pi_agent.agent_types import (
    AssistantMessage, TextBlock, DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
)
from pi_agent.llm import Model, LlmProvider, StreamEvent, Usage
from pi_agent.session import Session


# ==============================================
# Mock provider that returns overflow error first, then success
# ==============================================
class OverflowThenOkProvider(LlmProvider):
    def __init__(self):
        self._first_call = True

    async def stream(self, model, context, *, api_key=None, signal=None,
                     headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        if self._first_call:
            self._first_call = False
            # Simulate context overflow
            msg = AssistantMessage(
                content=[TextBlock(type="text", text="")],
                provider=model.provider, model=model.id,
                stopReason="error",
                errorMessage="This model's maximum context length is 128000 tokens. "
                             "However, your messages resulted in 150000 tokens. "
                             "Please reduce the length of the messages.",
                usage=Usage(input=0, output=0, totalTokens=0),
                timestamp=ts,
            )
            yield StreamEvent(type="start", partial=msg.model_copy(deep=True))
            yield StreamEvent(type="error", text=msg.error_message, partial=msg)
        else:
            # Recovery: normal response after compaction
            msg = AssistantMessage(
                content=[TextBlock(type="text", text="OK, recovered after compaction!")],
                provider=model.provider, model=model.id,
                stopReason="end_turn",
                usage=Usage(input=100, output=50, totalTokens=150),
                timestamp=ts,
            )
            yield StreamEvent(type="start", partial=msg.model_copy(deep=True))
            yield StreamEvent(type="text_delta", partial=msg.model_copy(deep=True), text="OK, recovered after compaction!")
            yield StreamEvent(type="done", partial=msg)

    async def complete(self, model, context, *, api_key=None, signal=None,
                       headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        if self._first_call:
            self._first_call = False
            return AssistantMessage(
                content=[TextBlock(type="text", text="")],
                provider=model.provider, model=model.id,
                stopReason="error",
                errorMessage="prompt is too long",
                usage=Usage(input=0, output=0, totalTokens=0),
                timestamp=ts,
            )
        return AssistantMessage(
            content=[TextBlock(type="text", text="Recovered!")],
            provider=model.provider, model=model.id,
            stopReason="end_turn",
            usage=Usage(input=100, output=50, totalTokens=150),
            timestamp=ts,
        )


# ==============================================
# Test: Overflow triggers compaction + retry
# ==============================================
async def test_overflow_recovery():
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.jsonl")
    session = await Session.create(path, cwd=tmpdir)

    model = Model(id="mock", provider="mock", contextWindow=128000, maxTokens=4096)
    mock = OverflowThenOkProvider()
    settings = CompactionSettings(enabled=True)

    agent = Agent(model=model, provider=mock, session=session, compaction_settings=settings)

    # Pre-load some messages so that compact() has something to summarize
    from pi_agent.agent_types import make_user_message
    for i in range(3):
        await session.append_message(make_user_message(f"Previous message {i}: " + "hello " * 50))

    # Restore messages from session
    from pi_agent.session import build_session_context
    ctx = build_session_context(session.entries)
    agent._messages = list(ctx.messages)

    new_msgs = await agent.prompt("Do something useful")

    # Should have recovered: the error message is NOT in the result
    assistant_msgs = [m for m in new_msgs if getattr(m, 'role', None) == "assistant"]
    error_msgs = [m for m in assistant_msgs if getattr(m, 'stop_reason', '') == "error"]
    assert len(error_msgs) == 0, f"Should not have error messages, got {len(error_msgs)}"
    print(f"  Assistant messages in result: {len(assistant_msgs)}")

    # Compaction should have been triggered
    entries = session.entries
    compactions = [e for e in entries if e.type == "compaction"]
    assert len(compactions) >= 1, f"Should have compaction entry, got {len(compactions)}"
    print(f"  Compaction entries: {len(compactions)}")
    print(f"  Total session entries: {len(entries)}")

    # Agent memory should NOT contain the error message
    error_in_memory = any(
        getattr(m, 'stop_reason', '') == "error"
        for m in agent.messages
        if getattr(m, 'role', None) == "assistant"
    )
    assert not error_in_memory, "Error message should not be in agent memory"

    os.remove(path)
    os.rmdir(tmpdir)
    print("Test PASSED: overflow recovery")


# ==============================================
# Test: No session → overflow is NOT recovered (no compaction possible)
# ==============================================
async def test_overflow_no_session():
    model = Model(id="mock", provider="mock", contextWindow=128000, maxTokens=4096)
    mock = OverflowThenOkProvider()

    agent = Agent(model=model, provider=mock)  # no session

    new_msgs = await agent.prompt("Hello")

    # Without session, overflow is just returned as error
    assistant_msgs = [m for m in new_msgs if getattr(m, 'role', None) == "assistant"]
    assert len(assistant_msgs) >= 1
    assert assistant_msgs[-1].stop_reason == "error", "Should get error without session"

    print("Test PASSED: overflow without session returns error")

    # The overflow error IS in memory (can't recover without session)
    last = agent.messages[-1]
    assert getattr(last, 'stop_reason', '') == "error"


# ==============================================
# Test: _is_context_overflow detection
# ==============================================
def test_overflow_detection():
    # Positive cases
    cases = [
        "This model's maximum context length is 128000 tokens.",
        "prompt is too long",
        "reduce the length of the prompt",
        "too many tokens in the request",
        "context window exceeded: max_tokens=4096",
        "token limit exceeded",
    ]
    for case in cases:
        msg = AssistantMessage(
            content=[], provider="mock", model="mock",
            stopReason="error", errorMessage=case, timestamp=123,
        )
        assert Agent._is_context_overflow(msg), f"Should detect: {case}"

    # Negative cases
    not_overflow = [
        AssistantMessage(content=[TextBlock(type="text", text="ok")], provider="mock",
                         model="mock", stopReason="end_turn", timestamp=123),
        AssistantMessage(content=[], provider="mock", model="mock",
                         stopReason="error", errorMessage="Rate limit exceeded", timestamp=123),
        AssistantMessage(content=[], provider="mock", model="mock",
                         stopReason="error", errorMessage="Authentication failed", timestamp=123),
    ]
    for msg in not_overflow:
        assert not Agent._is_context_overflow(msg), f"Should NOT detect: {getattr(msg, 'error_message', '')}"

    print("Test PASSED: overflow detection")


# ==============================================
# Main
# ==============================================
async def main():
    print("=" * 50)
    print("Overflow Detection & Recovery Tests")
    print("=" * 50 + "\n")

    test_overflow_detection()
    print()
    await test_overflow_no_session()
    print()
    await test_overflow_recovery()
    print()
    print("ALL OVERFLOW TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
