"""
All Pydantic v2 models for the pi-agent core engine.

Matches the TS types from:
  - packages/agent/src/types.ts
  - packages/agent/src/harness/types.ts
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


# ============================================================================
# Content blocks
# ============================================================================

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64 data URL or URL
    media_type: str = Field(default="image/png", alias="mediaType")


# ============================================================================
# Usage & Cost
# ============================================================================

class CostBreakdown(BaseModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: CostBreakdown = Field(default_factory=CostBreakdown)


# ============================================================================
# Content blocks for assistant messages
# ============================================================================

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolCallBlock(BaseModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# LLM-level messages
# ============================================================================

class _UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[TextContent | ImageContent]
    timestamp: int  # epoch ms


class _AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ThinkingBlock | ToolCallBlock]
    provider: str = ""
    model: str = ""
    api: str = "chat"
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str = Field(default="end_turn", alias="stopReason")
    error_message: str | None = Field(default=None, alias="errorMessage")
    timestamp: int


class _ToolResultMessage(BaseModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    details: Any = None
    is_error: bool = Field(default=False, alias="isError")
    timestamp: int


# ============================================================================
# Custom agent-level messages (beyond LLM standard roles)
# ============================================================================

class _BashExecutionMessage(BaseModel):
    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str = ""
    exit_code: int | None = Field(default=None, alias="exitCode")
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = Field(default=None, alias="fullOutputPath")
    timestamp: int
    exclude_from_context: bool = Field(default=False, alias="excludeFromContext")


class _CompactionSummaryMessage(BaseModel):
    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int = Field(alias="tokensBefore")
    timestamp: int


class _CustomMessage(BaseModel):
    role: Literal["custom"] = "custom"
    custom_type: str = Field(alias="customType")
    content: str | list[TextContent | ImageContent]
    display: bool = True
    details: Any = None
    timestamp: int


# Union type alias (consumers use this)
AgentMessage = Union[
    _UserMessage, _AssistantMessage, _ToolResultMessage,
    _BashExecutionMessage, _CompactionSummaryMessage, _CustomMessage,
]

# Re-export as usable classes
UserMessage = _UserMessage
AssistantMessage = _AssistantMessage
ToolResultMessage = _ToolResultMessage
BashExecutionMessage = _BashExecutionMessage
CompactionSummaryMessage = _CompactionSummaryMessage
CustomMessage = _CustomMessage


# Factory helpers
def make_user_message(text: str, timestamp: int | None = None) -> UserMessage:
    import time
    return UserMessage(
        content=text,
        timestamp=timestamp or int(time.time() * 1000),
    )


def make_text_content(text: str) -> TextContent:
    return TextContent(text=text)


def make_image_content(data: str, media_type: str = "image/png") -> ImageContent:
    return ImageContent(data=data, mediaType=media_type)


def make_tool_result_message(
    tool_call_id: str, tool_name: str, content: str, *,
    details: Any = None, is_error: bool = False,
) -> ToolResultMessage:
    import time
    return ToolResultMessage(
        toolCallId=tool_call_id,
        toolName=tool_name,
        content=[TextContent(text=content)],
        details=details,
        isError=is_error,
        timestamp=int(time.time() * 1000),
    )


def make_compaction_summary_message(
    summary: str, tokens_before: int, timestamp: int | None = None,
) -> CompactionSummaryMessage:
    import time
    return CompactionSummaryMessage(
        summary=summary,
        tokensBefore=tokens_before,
        timestamp=timestamp or int(time.time() * 1000),
    )


# ============================================================================
# Tool definition
# ============================================================================

ToolExecutionMode = Literal["sequential", "parallel"]
QueueMode = Literal["all", "one-at-a-time"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


class AgentToolResult(BaseModel):
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    details: Any = None
    terminate: bool = False


class AgentTool(BaseModel):
    """Tool registered with the agent."""
    name: str
    description: str = ""
    label: str = ""
    parameters_schema: dict[str, Any] = Field(default_factory=dict, alias="parametersSchema")
    execution_mode: ToolExecutionMode | None = Field(default=None, alias="executionMode")
    execute: Any = Field(default=None, exclude=True)
    prepare_arguments: Any = Field(default=None, exclude=True)
    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Context & Loop Config
# ============================================================================

class AgentContext(BaseModel):
    system_prompt: str = Field(default="", alias="systemPrompt")
    messages: list[AgentMessage] = Field(default_factory=list)
    tools: list[AgentTool] = Field(default_factory=list)


class BeforeToolCallContext(BaseModel):
    assistant_message: AssistantMessage = Field(alias="assistantMessage")
    tool_call: ToolCallBlock = Field(alias="toolCall")
    args: Any
    context: AgentContext


class AfterToolCallContext(BaseModel):
    assistant_message: AssistantMessage = Field(alias="assistantMessage")
    tool_call: ToolCallBlock = Field(alias="toolCall")
    args: Any
    result: AgentToolResult
    is_error: bool = Field(alias="isError")
    context: AgentContext


class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str | None = None


class AfterToolCallResult(BaseModel):
    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = Field(default=None, alias="isError")
    terminate: bool | None = None


class ShouldStopAfterTurnContext(BaseModel):
    message: AssistantMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list, alias="toolResults")
    context: AgentContext
    new_messages: list[AgentMessage] = Field(default_factory=list, alias="newMessages")


class AgentLoopTurnUpdate(BaseModel):
    context: AgentContext | None = None
    model: Any | None = None
    thinking_level: ThinkingLevel | None = Field(default=None, alias="thinkingLevel")


class AgentLoopConfig(BaseModel):
    """Configuration for the stateless agent loop."""
    model: Any = Field(default=None)  # Model from llm module
    reasoning: ThinkingLevel | None = None
    api_key: str | None = Field(default=None, alias="apiKey")
    session_id: str | None = Field(default=None, alias="sessionId")
    tool_execution: ToolExecutionMode = Field(default="parallel", alias="toolExecution")

    # Callbacks
    convert_to_llm: Any = Field(default=None, alias="convertToLlm")
    transform_context: Any = Field(default=None)
    get_api_key: Any = Field(default=None)
    should_stop_after_turn: Any = Field(default=None)
    prepare_next_turn: Any = Field(default=None)
    get_steering_messages: Any = Field(default=None)
    get_follow_up_messages: Any = Field(default=None)
    before_tool_call: Any = Field(default=None)
    after_tool_call: Any = Field(default=None)

    # Stream options
    transport: str = "auto"
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    max_retries: int = Field(default=3, alias="maxRetries")
    max_retry_delay_ms: int | None = Field(default=None, alias="maxRetryDelayMs")
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Session entry types (simplified: linear, no tree)
# ============================================================================

class SessionEntry(BaseModel):
    """A single entry in the session JSONL. Simplified from the TS discriminated union."""
    type: str  # "message", "compaction", "thinking_level_change", "model_change", "custom_message", "session_info"
    id: str
    timestamp: str  # ISO 8601

    # Payload fields (only relevant ones are populated based on type)
    message: AgentMessage | None = None
    summary: str | None = None
    tokens_before: int | None = Field(default=None, alias="tokensBefore")
    first_kept_entry_id: str | None = Field(default=None, alias="firstKeptEntryId")
    read_files: list[str] | None = Field(default=None, alias="readFiles")
    modified_files: list[str] | None = Field(default=None, alias="modifiedFiles")
    thinking_level: str | None = Field(default=None, alias="thinkingLevel")
    provider: str | None = None
    model_id: str | None = Field(default=None, alias="modelId")
    custom_type: str | None = Field(default=None, alias="customType")
    content: str | list[TextContent | ImageContent] | None = None
    display: bool | None = None
    details: Any = None


class SessionContext(BaseModel):
    messages: list[AgentMessage] = Field(default_factory=list)
    thinking_level: str = Field(default="off", alias="thinkingLevel")
    model_provider: str | None = None
    model_id: str | None = None


# ============================================================================
# Compaction types
# ============================================================================

class CompactionSettings(BaseModel):
    enabled: bool = True
    reserve_tokens: int = Field(default=16384, alias="reserveTokens")
    keep_recent_tokens: int = Field(default=20000, alias="keepRecentTokens")


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


class FileOperations(BaseModel):
    """File paths touched by a compaction range."""
    read: set[str] = Field(default_factory=set)
    written: set[str] = Field(default_factory=set)
    edited: set[str] = Field(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}


class CutPointResult(BaseModel):
    first_kept_entry_index: int
    turn_start_index: int = -1
    is_split_turn: bool = False


class CompactionPreparation(BaseModel):
    first_kept_entry_id: str
    messages_to_summarize: list[AgentMessage] = Field(default_factory=list)
    turn_prefix_messages: list[AgentMessage] = Field(default_factory=list)
    is_split_turn: bool = False
    tokens_before: int = 0
    previous_summary: str | None = None
    file_ops: FileOperations = Field(default_factory=FileOperations)
    settings: CompactionSettings = Field(default_factory=CompactionSettings)

    model_config = {"arbitrary_types_allowed": True}


class CompactionResult(BaseModel):
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    read_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)


# ============================================================================
# Event type strings
# ============================================================================

AGENT_EVENT_TYPES = Literal[
    "agent_start", "agent_end",
    "turn_start", "turn_end",
    "message_start", "message_end", "message_update",
    "tool_execution_start", "tool_execution_update", "tool_execution_end",
]

# AgentEvent is a plain dict to avoid Pydantic overhead on every emit
# Format: {"type": str, ...event-specific fields}
