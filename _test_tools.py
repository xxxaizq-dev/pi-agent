"""Test Agent with real tools (read_file, write_file, bash)."""
import sys, os, asyncio, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent import Agent
from pi_agent.agent_types import (
    AssistantMessage, TextBlock, ToolCallBlock, make_user_message, Usage,
)
from pi_agent.llm import Model, LlmProvider, StreamEvent
from pi_agent.tools import create_default_tools


# ==============================================
# Mock provider: returns one tool call then one text response
# ==============================================
class MockCodingProvider(LlmProvider):
    def __init__(self):
        self._call = 0

    async def stream(self, model, context, *, api_key=None, signal=None,
                     headers=None, max_tokens=None, reasoning=None):
        ts = int(time.time() * 1000)
        self._call += 1

        if self._call == 1:
            # First response: call read tool
            partial = AssistantMessage(
                content=[
                    ToolCallBlock(
                        type="toolCall", id="tc1", name="read",
                        arguments={"file_path": "test.txt"},
                    ),
                ],
                provider=model.provider, model=model.id,
                stopReason="tool_use",
                usage=Usage(input=100, output=50, totalTokens=150),
                timestamp=ts,
            )
            yield StreamEvent(type="start", partial=partial.model_copy(deep=True))
            yield StreamEvent(type="done", partial=partial)
        else:
            # Second response: text based on tool result
            partial = AssistantMessage(
                content=[TextBlock(type="text", text="File read successfully. The content is: hello world")],
                provider=model.provider, model=model.id,
                stopReason="end_turn",
                usage=Usage(input=200, output=80, totalTokens=280),
                timestamp=ts,
            )
            yield StreamEvent(type="start", partial=partial.model_copy(deep=True))
            yield StreamEvent(type="text_delta", partial=partial.model_copy(deep=True),
                            text="File read successfully. The content is: hello world")
            yield StreamEvent(type="done", partial=partial)

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


async def test_read_file_tool():
    tmpdir = tempfile.mkdtemp()
    # Create a test file
    test_file = os.path.join(tmpdir, "test.txt")
    with open(test_file, "w") as f:
        f.write("hello world")

    model = Model(id="mock", provider="mock", contextWindow=128000)
    mock = MockCodingProvider()

    tools = create_default_tools(cwd=tmpdir)
    agent = Agent(model=model, provider=mock, tools=tools, system_prompt="You are a coding assistant.")

    new_msgs = await agent.prompt("Read test.txt and tell me what's inside")

    assistant_msgs = [m for m in new_msgs if getattr(m, "role", None) == "assistant"]
    tool_results = [m for m in new_msgs if getattr(m, "role", None) == "toolResult"]
    print(f"  Assistant msgs: {len(assistant_msgs)}")
    print(f"  Tool results: {len(tool_results)}")

    assert len(tool_results) >= 1, "Should have at least 1 tool result"
    assert len(assistant_msgs) >= 2, "Should have 2 assistant messages (tool call + final)"
    print("Test 1 PASSED: read_file tool")

    os.remove(test_file)
    os.rmdir(tmpdir)


async def test_write_file_tool():
    tmpdir = tempfile.mkdtemp()
    out_file = os.path.join(tmpdir, "output.txt")

    # Manual tool invocation to test write directly
    from pi_agent.tools import create_default_tools
    tools = create_default_tools(cwd=tmpdir)
    write_tool = [t for t in tools if t.name == "write"][0]

    result = await write_tool.execute("tc1", {"file_path": out_file, "content": "generated content here"})
    assert "written" in result.content[0].text.lower() or "File written" in result.content[0].text, \
        f"Unexpected result: {result.content[0].text[:100]}"
    assert os.path.exists(out_file), "File should exist"
    with open(out_file) as f:
        assert f.read() == "generated content here"
    print("Test 2 PASSED: write_file tool")

    os.remove(out_file)
    os.rmdir(tmpdir)


async def test_bash_tool():
    # Manual tool invocation
    tools = create_default_tools(cwd=".")
    bash_tool = [t for t in tools if t.name == "bash"][0]

    result = await bash_tool.execute("tc1", {"command": "echo hello from bash"})
    text = result.content[0].text
    assert "hello from bash" in text, f"Unexpected output: {text[:100]}"
    assert result.details["exitCode"] == 0
    print("Test 2 PASSED: bash tool")


async def test_tool_permissions():
    """Verify tools reject paths outside working directory."""
    tmpdir = tempfile.mkdtemp()
    tools = create_default_tools(cwd=tmpdir)
    read_tool = [t for t in tools if t.name == "read"][0]
    write_tool = [t for t in tools if t.name == "write"][0]

    # Try to read outside cwd
    result = await read_tool.execute("tc1", {"file_path": "/etc/passwd"})
    assert "access denied" in result.content[0].text.lower() or "Error" in result.content[0].text, \
        f"Should deny access: {result.content[0].text[:100]}"
    print("  read: access denied OK")

    # Try to write outside cwd
    result = await write_tool.execute("tc1", {"file_path": "/etc/hacked", "content": "bad"})
    assert "access denied" in result.content[0].text.lower() or "Error" in result.content[0].text, \
        f"Should deny access: {result.content[0].text[:100]}"
    print("  write: access denied OK")

    print("Test 4 PASSED: tool permissions")

    os.rmdir(tmpdir)


async def main():
    print("=" * 50)
    print("Built-in Tools Tests")
    print("=" * 50 + "\n")

    await test_read_file_tool()
    await test_write_file_tool()
    await test_bash_tool()
    await test_tool_permissions()
    print()
    print("ALL TOOL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
