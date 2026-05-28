"""Test message conversion pipeline (messages.py)."""
import sys, os, asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_agent.agent_types import (
    CompactionSummaryMessage, ToolResultMessage, UserMessage, TextContent,
    make_compaction_summary_message,
)
from pi_agent.messages import (
    convert_to_llm,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX,
)


async def test_compaction_summary_xml_wrapped():
    msg = CompactionSummaryMessage(
        summary="## Goal\nBuild a web app\n\n## Progress\n- [x] Done",
        tokensBefore=50000, timestamp=1234567890000,
    )
    result = await convert_to_llm([msg])
    content = result[0].content
    assert COMPACTION_SUMMARY_PREFIX in content
    assert COMPACTION_SUMMARY_SUFFIX in content
    assert "## Goal" in content
    print("Test 1 PASSED: compaction summary XML wrapped")


async def test_bash_tool_result_formatted():
    """Bash tool result should get markdown + exit code from details."""
    msg = ToolResultMessage(
        toolCallId="tc1", toolName="bash",
        content=[TextContent(text="5 passing\n1 failing")],
        details={"command": "npm test", "exitCode": 1, "timedOut": False, "truncated": False},
        isError=False, timestamp=0,
    )
    result = await convert_to_llm([msg])
    content = result[0].content
    text = content[0].text if isinstance(content, list) else content
    assert "Ran `npm test`" in text
    assert "```" in text
    assert "5 passing" in text
    assert "Command exited with code 1" in text
    print("Test 2 PASSED: bash tool result formatted")


async def test_bash_tool_result_timed_out():
    """Bash timeout should be marked."""
    msg = ToolResultMessage(
        toolCallId="tc1", toolName="bash",
        content=[TextContent(text="")],
        details={"command": "sleep 999", "exitCode": -1, "timedOut": True, "timeout": 120},
        isError=False, timestamp=0,
    )
    result = await convert_to_llm([msg])
    text = result[0].content[0].text if isinstance(result[0].content, list) else result[0].content
    assert "timed out" in text.lower()
    print("Test 3 PASSED: bash timeout marked")


async def test_tool_error_marked():
    """Tool result with isError prepends error marker."""
    msg = ToolResultMessage(
        toolCallId="tc1", toolName="read",
        content=[TextContent(text="file not found: /bad/path.txt")],
        isError=True, timestamp=0,
    )
    result = await convert_to_llm([msg])
    text = result[0].content[0].text if isinstance(result[0].content, list) else result[0].content
    assert "Tool error" in text
    print("Test 4 PASSED: tool error marked")


async def test_normal_tool_result_pass_through():
    """Non-bash, non-error tool result passes through with truncation only."""
    msg = ToolResultMessage(
        toolCallId="tc1", toolName="read",
        content=[TextContent(text="file content here")],
        isError=False, timestamp=0,
    )
    result = await convert_to_llm([msg])
    text = result[0].content[0].text if isinstance(result[0].content, list) else result[0].content
    assert "file content here" in text
    assert "Tool error" not in text  # no error marker on normal results
    print("Test 5 PASSED: normal tool result pass-through")


async def test_bash_output_truncation():
    """Very long bash output should be truncated."""
    msg = ToolResultMessage(
        toolCallId="tc1", toolName="bash",
        content=[TextContent(text="A" * 12000)],  # > MAX_BASH_OUTPUT_CHARS (8000)
        details={"command": "cat huge.txt", "exitCode": 0, "timedOut": False, "truncated": True},
        isError=False, timestamp=0,
    )
    result = await convert_to_llm([msg])
    text = result[0].content[0].text if isinstance(result[0].content, list) else result[0].content
    assert "truncated" in text.lower()
    assert len(text) < 11000
    print("Test 6 PASSED: bash output truncated")


async def test_user_pass_through():
    msg = UserMessage(content="hello", timestamp=0)
    result = await convert_to_llm([msg])
    assert result[0] is msg
    print("Test 7 PASSED: user pass-through")


print("=" * 50)
print("Message Conversion Pipeline Tests")
print("=" * 50 + "\n")

asyncio.run(test_compaction_summary_xml_wrapped())
asyncio.run(test_bash_tool_result_formatted())
asyncio.run(test_bash_tool_result_timed_out())
asyncio.run(test_tool_error_marked())
asyncio.run(test_normal_tool_result_pass_through())
asyncio.run(test_bash_output_truncation())
asyncio.run(test_user_pass_through())

print()
print("ALL MESSAGE CONVERSION TESTS PASSED")
