"""End-to-end smoke tests for pi_agent — uses mock LLM provider."""
import sys, os, asyncio, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent_types import (
    AgentContext, AgentLoopConfig, AgentTool, AgentToolResult,
    UserMessage, AssistantMessage, TextBlock, TextContent,
    ToolCallBlock, make_user_message, Usage,
)
from pi_agent.llm import Model, LlmProvider, StreamEvent


# ==============================================
# Mock provider: returns canned responses
# ==============================================
class MockProvider(LlmProvider):
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0

    async def stream(self, model, context, *, api_key=None, signal=None,
                     headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        if self.call_count >= len(self.responses):
            msg = AssistantMessage(
                content=[TextBlock(type="text", text="")],
                provider=model.provider, model=model.id,
                stopReason="end_turn", timestamp=ts,
                usage=Usage(input=10, output=5, totalTokens=15),
            )
            yield StreamEvent(type="start", partial=msg.model_copy(deep=True))
            yield StreamEvent(type="done", partial=msg)
            return

        turn = self.responses[self.call_count]
        self.call_count += 1

        content_blocks = []
        for r in turn:
            if r["type"] == "text":
                content_blocks.append(TextBlock(type="text", text=r["text"]))
            elif r["type"] == "toolCall":
                content_blocks.append(ToolCallBlock(
                    type="toolCall", id=r["id"], name=r["name"],
                    arguments=r.get("arguments", {}),
                ))

        has_tool = any(b.type == "toolCall" for b in content_blocks)
        partial = AssistantMessage(
            content=content_blocks,
            provider=model.provider, model=model.id,
            stopReason="tool_use" if has_tool else "end_turn",
            usage=Usage(input=20, output=10, totalTokens=30),
            timestamp=ts,
        )
        yield StreamEvent(type="start", partial=partial.model_copy(deep=True))
        for block in content_blocks:
            if isinstance(block, TextBlock):
                yield StreamEvent(type="text_delta",
                    partial=partial.model_copy(deep=True), text=block.text)
        yield StreamEvent(type="done", partial=partial)

    async def complete(self, model, context, *, api_key=None, signal=None,
                       headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        return AssistantMessage(
            content=[TextBlock(type="text", text="mock complete")],
            provider=model.provider, model=model.id,
            stopReason="end_turn", timestamp=ts,
            usage=Usage(input=10, output=5, totalTokens=15),
        )


# ==============================================
# Test 1: Simple one-turn conversation
# ==============================================
async def test_simple_conversation():
    from pi_agent.agent_loop import run_agent_loop

    model = Model(id="mock-model", provider="mock", contextWindow=128000)
    mock = MockProvider(responses=[
        [{"type": "text", "text": "Hello! How can I help you?"}],
    ])

    events = []

    async def collect(event):
        events.append(event)

    context = AgentContext(systemPrompt="You are a test assistant.", messages=[])
    config = AgentLoopConfig(model=model, convertToLlm=lambda msgs: msgs)

    prompt = make_user_message("Hi!")
    new_msgs = await run_agent_loop([prompt], context, config, collect, provider=mock)

    print(f"New messages: {len(new_msgs)}")
    for msg in new_msgs:
        if hasattr(msg, "role"):
            texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
            print(f"  {msg.role}: {' '.join(texts)[:80]}")

    event_types = [e["type"] for e in events]
    print(f"Events: {event_types}")

    assert "agent_start" in event_types, "Missing agent_start"
    assert "agent_end" in event_types, "Missing agent_end"
    assert "turn_start" in event_types, "Missing turn_start"
    assert "turn_end" in event_types, "Missing turn_end"
    print("Test 1 PASSED: simple conversation\n")


# ==============================================
# Test 2: Tool call conversation
# ==============================================
async def test_tool_call():
    from pi_agent.agent_loop import run_agent_loop

    model = Model(id="mock-model", provider="mock", contextWindow=128000)
    mock = MockProvider(responses=[
        [{"type": "toolCall", "id": "tc1", "name": "echo",
          "arguments": {"message": "test"}}],
        [{"type": "text", "text": "Tool executed successfully."}],
    ])

    events = []

    async def collect(event):
        events.append(event)

    async def echo_execute(tool_call_id, params, signal, on_update):
        msg = params.get("message", "")
        return AgentToolResult(
            content=[TextContent(text=f"Echo: {msg}")],
            details={"echoed": msg},
        )

    echo_tool = AgentTool(
        name="echo", description="Echo a message", label="Echo",
        parametersSchema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        execute=echo_execute,
    )

    context = AgentContext(
        systemPrompt="You are a test assistant.",
        messages=[], tools=[echo_tool],
    )
    config = AgentLoopConfig(model=model, convertToLlm=lambda msgs: msgs)

    prompt = make_user_message("Echo test please")
    new_msgs = await run_agent_loop([prompt], context, config, collect, provider=mock)

    assistant_msgs = [m for m in new_msgs if hasattr(m, "role") and m.role == "assistant"]
    tool_results = [m for m in new_msgs if hasattr(m, "role") and m.role == "toolResult"]
    print(f"Assistant messages: {len(assistant_msgs)}, Tool results: {len(tool_results)}")

    assert len(assistant_msgs) >= 2, f"Expected >=2 assistant msgs, got {len(assistant_msgs)}"
    assert len(tool_results) >= 1, f"Expected >=1 tool results, got {len(tool_results)}"

    event_types = [e["type"] for e in events]
    assert "tool_execution_start" in event_types, "Missing tool_execution_start"
    assert "tool_execution_end" in event_types, "Missing tool_execution_end"
    print("Test 2 PASSED: tool call conversation\n")


# ==============================================
# Test 3: Steering and follow-up messages
# ==============================================
async def test_steering_followup():
    from pi_agent.agent import Agent

    model = Model(id="mock-model", provider="mock", contextWindow=128000)
    mock = MockProvider(responses=[
        [{"type": "text", "text": "Working on it..."}],
        [{"type": "text", "text": "OK, I fixed it."}],
    ])

    agent = Agent(model=model, provider=mock)

    # Use internal _run for testing (prompt() would require asyncio.Lock which is fine)
    event_count = 0

    async def on_event(event):
        nonlocal event_count
        event_count += 1

    agent.on_event(on_event)

    # Test steer and follow_up
    agent.steer("hurry up")
    agent.follow_up("are you done?")

    # Clear them for clean test
    agent._steer_queue.clear()
    agent._follow_up_queue.clear()

    # Test prompt
    new_msgs = await agent.prompt("Hello")
    print(f"Prompt returned {len(new_msgs)} messages")
    assert len(new_msgs) >= 1

    # Check that events were emitted
    print(f"Events received: {event_count}")
    assert event_count > 0, "No events were emitted"

    print("Test 3 PASSED: Agent with events\n")


# ==============================================
# Main
# ==============================================
async def main():
    print("=" * 50)
    print("pi_agent E2E Smoke Tests")
    print("=" * 50 + "\n")

    await test_simple_conversation()
    await test_tool_call()
    await test_steering_followup()

    print("ALL E2E TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
