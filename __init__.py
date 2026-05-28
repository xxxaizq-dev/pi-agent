"""
pi_agent — a minimal Python agent engine inspired by the pi-agent core.

Usage:
    from pi_agent import Agent, Model, OpenAiProvider

    agent = Agent(
        model=Model(id="gpt-4o", provider="openai", contextWindow=128000),
        system_prompt="You are a helpful assistant.",
    )
    result = await agent.prompt("Hello!")
    await agent.compact()  # trigger compaction
"""

from .agent import Agent, default_convert_to_llm
from .compaction import (
    compact,
    generate_summary,
    find_cut_point,
    prepare_compaction,
    should_compact,
)
from .llm import (
    AnthropicProvider,
    LlmProvider,
    LlmContext,
    Model,
    OpenAiProvider,
    StreamEvent,
)
from .messages import convert_to_llm
from .session import Session, JsonlSessionStorage, build_session_context
from .tools import create_default_tools
from .agent_types import (
    # Message types
    AgentMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    BashExecutionMessage,
    CompactionSummaryMessage,
    CustomMessage,
    # Content blocks
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    TextContent,
    ImageContent,
    # Tool types
    AgentTool,
    AgentToolResult,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    # Config
    CompactionSettings,
    CompactionPreparation,
    CompactionResult,
    CutPointResult,
    FileOperations,
    DEFAULT_COMPACTION_SETTINGS,
    # Session
    SessionEntry,
    SessionContext,
    # Hooks
    BeforeToolCallContext,
    BeforeToolCallResult,
    AfterToolCallContext,
    AfterToolCallResult,
    ShouldStopAfterTurnContext,
    # Enums
    ToolExecutionMode,
    QueueMode,
    ThinkingLevel,
    # Factories
    make_user_message,
    make_tool_result_message,
    make_compaction_summary_message,
    make_text_content,
    make_image_content,
    # Usage
    Usage,
    CostBreakdown,
)
from .utils import (
    estimate_tokens,
    estimate_context_tokens,
    maybe_await,
    serialize_conversation,
    uuid_v7,
)

__all__ = [
    # Agent
    "Agent",
    "default_convert_to_llm",
    "convert_to_llm",
    # LLM
    "Model",
    "LlmProvider",
    "LlmContext",
    "OpenAiProvider",
    "AnthropicProvider",
    "StreamEvent",
    # Session
    "Session",
    "JsonlSessionStorage",
    "build_session_context",
    # Tools
    "create_default_tools",
    # Compaction
    "compact",
    "generate_summary",
    "find_cut_point",
    "prepare_compaction",
    "should_compact",
    # Messages
    "AgentMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "BashExecutionMessage",
    "CompactionSummaryMessage",
    "CustomMessage",
    # Content
    "TextBlock",
    "ThinkingBlock",
    "ToolCallBlock",
    "TextContent",
    "ImageContent",
    # Tools
    "AgentTool",
    "AgentToolResult",
    "AgentContext",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    # Config
    "CompactionSettings",
    "CompactionPreparation",
    "CompactionResult",
    "CutPointResult",
    "FileOperations",
    "DEFAULT_COMPACTION_SETTINGS",
    # Session types
    "SessionEntry",
    "SessionContext",
    # Hook types
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "ShouldStopAfterTurnContext",
    # Enums
    "ToolExecutionMode",
    "QueueMode",
    "ThinkingLevel",
    # Factories
    "make_user_message",
    "make_tool_result_message",
    "make_compaction_summary_message",
    "make_text_content",
    "make_image_content",
    # Usage
    "Usage",
    "CostBreakdown",
    # Utils
    "estimate_tokens",
    "estimate_context_tokens",
    "maybe_await",
    "serialize_conversation",
    "uuid_v7",
]
