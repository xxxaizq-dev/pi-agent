"""
Message conversion pipeline: AgentMessage → LLM-compatible messages.

Each message role gets dedicated formatting before being passed to the LLM.
Design boundary: what's stored ≠ what the LLM sees.

Currently active branches:
  - compaction summary → <summary> XML wrapped with explanatory prefix
  - bash tool result    → markdown code block + exit code + truncation
  - other tool results  → error marking + output truncation
  - user/assistant      → pass-through
"""

from __future__ import annotations

import time
from typing import Any

from .agent_types import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

# ============================================================================
# Formatting constants
# ============================================================================

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted "
    "into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

MAX_TOOL_OUTPUT_CHARS = 10000
MAX_BASH_OUTPUT_CHARS = 8000


# ============================================================================
# Per-role formatting helpers
# ============================================================================

def _format_bash_tool_result(details: dict[str, Any] | None, output: str) -> str:
    """Format a bash tool result as structured markdown for the LLM."""
    if not details:
        return output

    command = details.get("command", "")
    exit_code = details.get("exitCode", 0)
    timed_out = details.get("timedOut", False)
    was_truncated = details.get("truncated", False)

    text_parts: list[str] = [f"Ran `{command}`"]

    if output:
        if len(output) > MAX_BASH_OUTPUT_CHARS:
            removed = len(output) - MAX_BASH_OUTPUT_CHARS
            output = (
                output[:MAX_BASH_OUTPUT_CHARS]
                + f"\n\n[... {removed} more characters truncated]"
            )
        text_parts.append(f"\n```\n{output}\n```")
    else:
        text_parts.append("(no output)")

    if timed_out:
        text_parts.append(f"\nCommand timed out after {details.get('timeout', '?')}s")
    elif exit_code is not None and exit_code != 0:
        text_parts.append(f"\nCommand exited with code {exit_code}")

    if was_truncated:
        text_parts.append("\n[Output was truncated]")

    return "".join(text_parts)


def _format_tool_content(content: list[TextContent | Any]) -> str:
    """Extract text from tool result content blocks, with truncation."""
    text = "".join(
        c.text if hasattr(c, "text") else str(c)
        for c in content
    )
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        removed = len(text) - MAX_TOOL_OUTPUT_CHARS
        text = text[:MAX_TOOL_OUTPUT_CHARS] + (
            f"\n\n[... {removed} more characters truncated]"
        )
    return text


# ============================================================================
# Main conversion function
# ============================================================================

async def convert_to_llm(
    messages: list[AgentMessage],
) -> list[UserMessage | AssistantMessage | ToolResultMessage]:
    """Convert AgentMessage[] to LLM-compatible Message[].

    Compaction summaries are wrapped in <summary> XML.
    Bash tool results get structured markdown with exit code.
    Other tool results get error markers and truncation.
    User and assistant messages pass through unchanged.
    """
    result: list[UserMessage | AssistantMessage | ToolResultMessage] = []

    for msg in messages:
        role = getattr(msg, "role", None)

        if role == "user":
            result.append(
                msg
                if isinstance(msg, UserMessage)
                else UserMessage(
                    content=getattr(msg, "content", ""),
                    timestamp=getattr(msg, "timestamp", int(time.time() * 1000)),
                )
            )

        elif role == "assistant":
            result.append(
                msg
                if isinstance(msg, AssistantMessage)
                else AssistantMessage(
                    content=getattr(msg, "content", []),
                    provider=getattr(msg, "provider", ""),
                    model=getattr(msg, "model", ""),
                    timestamp=getattr(msg, "timestamp", int(time.time() * 1000)),
                    stopReason=getattr(msg, "stop_reason", "end_turn"),
                )
            )

        elif role == "toolResult":
            trm = (
                msg
                if isinstance(msg, ToolResultMessage)
                else ToolResultMessage(
                    toolCallId=getattr(msg, "tool_call_id", ""),
                    toolName=getattr(msg, "tool_name", ""),
                    content=getattr(msg, "content", []),
                    details=getattr(msg, "details", None),
                    isError=getattr(msg, "is_error", False),
                    timestamp=getattr(msg, "timestamp", int(time.time() * 1000)),
                )
            )

            raw_text = _format_tool_content(trm.content)

            if trm.is_error:
                trm.content = [TextContent(text=f"[Tool error]\n{raw_text}")]
            elif trm.tool_name == "bash":
                trm.content = [TextContent(
                    text=_format_bash_tool_result(trm.details, raw_text)
                )]
            else:
                trm.content = [TextContent(text=raw_text)]

            result.append(trm)

        elif role == "compactionSummary":
            summary = getattr(msg, "summary", "")
            result.append(
                UserMessage(
                    content=COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX,
                    timestamp=getattr(msg, "timestamp", int(time.time() * 1000)),
                )
            )

        # Unknown roles (branchSummary, custom, bashExecution, etc.) are
        # silently dropped — they only exist in harness-level metadata flows
        # and should never reach the LLM.

    return result
