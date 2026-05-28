"""
Stateful Agent class wrapping the stateless agent loop.

Usage:
    agent = Agent(model=..., tools=[...], system_prompt="...")
    result = await agent.prompt("fix the bug")
    result = await agent.prompt("now add tests")
    await agent.compact()  # manually trigger compaction
"""

from __future__ import annotations

import asyncio
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .compaction import compact as _do_compact
from .compaction import prepare_compaction, should_compact
from .llm import LlmProvider, Model, OpenAiProvider
from .messages import convert_to_llm as _default_convert_to_llm
from .session import Session, build_session_context
from .agent_types import (
    AgentContext,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    CompactionSettings,
    CompactionResult,
    CompactionPreparation,
    CompactionSummaryMessage,
    DEFAULT_COMPACTION_SETTINGS,
    QueueMode,
    TextBlock,
    TextContent,
    ThinkingLevel,
    ToolCallBlock,
    ToolExecutionMode,
    ToolResultMessage,
    UserMessage,
    make_compaction_summary_message,
    make_tool_result_message,
    make_user_message,
)
from .utils import estimate_context_tokens, maybe_await


# Re-export from messages.py for backward compatibility
default_convert_to_llm = _default_convert_to_llm


# ============================================================================
# Agent class
# ============================================================================

class Agent:
    """Stateful agent wrapping the stateless loop."""

    def __init__(
        self,
        model: Model,
        *,
        tools: list[AgentTool] | None = None,
        system_prompt: str | Callable[[dict[str, Any]], str | Awaitable[str]] = "",
        provider: LlmProvider | None = None,
        cwd: str = ".",
        thinking_level: ThinkingLevel = "off",
        tool_execution: ToolExecutionMode = "parallel",
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        api_key: str | None = None,
        session: Session | None = None,
        compaction_settings: CompactionSettings | None = None,
        # Callbacks
        convert_to_llm: Any = None,
        transform_context: Any = None,
        get_api_key: Any = None,
        before_tool_call: Any = None,
        after_tool_call: Any = None,
        should_stop_after_turn: Any = None,
        prepare_next_turn: Any = None,
    ):
        self._model = model
        self._tools: dict[str, AgentTool] = {}
        for t in (tools or []):
            self._tools[t.name] = t
        self._system_prompt = system_prompt
        self._cwd = str(Path(cwd).resolve())
        self._provider = provider or OpenAiProvider()
        self._thinking_level = thinking_level
        self._tool_execution = tool_execution
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._api_key = api_key
        self._session = session
        self._compaction_settings = compaction_settings or DEFAULT_COMPACTION_SETTINGS

        # Callbacks
        self._convert_to_llm = convert_to_llm or default_convert_to_llm
        self._transform_context = transform_context
        self._get_api_key = get_api_key
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._should_stop_after_turn = should_stop_after_turn
        self._prepare_next_turn = prepare_next_turn

        # State
        self._messages: list[AgentMessage] = []
        self._steer_queue: list[UserMessage] = []
        self._follow_up_queue: list[UserMessage] = []
        self._next_turn_queue: list[AgentMessage] = []
        self._event_listeners: list[Callable[[dict[str, Any]], Awaitable[None]]] = []

        # Concurrency
        self._run_lock = asyncio.Lock()
        self._abort_event: asyncio.Event | None = None
        self._is_busy = False

        # Restore messages from session on init
        if self._session:
            ctx = build_session_context(self._session.entries)
            self._messages = list(ctx.messages)

    # ========================================================================
    # Public API
    # ========================================================================

    async def prompt(self, text: str) -> list[AgentMessage]:
        """Send a text prompt. Returns all new messages from this run."""
        msg = make_user_message(text)
        return await self._run([msg])

    async def continue_(self) -> list[AgentMessage]:
        """Continue from current transcript without adding a prompt."""
        if self._is_busy:
            raise RuntimeError("Agent is busy")
        context = self._build_context()
        config = self._build_loop_config()

        async with self._run_lock:
            self._is_busy = True
            self._abort_event = asyncio.Event()
            try:
                new_msgs = await run_agent_loop_continue(
                    context, config, self._emit,
                    provider=self._provider, signal=self._abort_event,
                )
                self._messages.extend(new_msgs)
                return new_msgs
            finally:
                self._is_busy = False
                self._abort_event = None

    async def prompt_message(self, message: AgentMessage) -> list[AgentMessage]:
        """Send raw agent messages."""
        return await self._run([message])

    def steer(self, text: str) -> None:
        """Queue a steering message for the running agent."""
        msg = make_user_message(text)
        self._steer_queue.append(msg)

    def follow_up(self, text: str) -> None:
        """Queue a follow-up message to extend the agent after it would stop."""
        msg = make_user_message(text)
        self._follow_up_queue.append(msg)

    def next_turn(self, text: str) -> None:
        """Queue a message for the next turn."""
        msg = make_user_message(text)
        self._next_turn_queue.append(msg)

    async def abort(self) -> None:
        """Cancel the current run."""
        if self._abort_event:
            self._abort_event.set()
        self._steer_queue.clear()
        self._follow_up_queue.clear()
        await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        """Wait until the agent is idle."""
        while self._is_busy:
            await asyncio.sleep(0.05)

    def reset(self) -> None:
        """Clear transcript and queues."""
        self._messages.clear()
        self._steer_queue.clear()
        self._follow_up_queue.clear()
        self._next_turn_queue.clear()

    # ========================================================================
    # Compaction
    # ========================================================================

    def check_compaction(self) -> bool:
        """Check if compaction should trigger based on current context.

        Includes anti-thrashing: after a compaction, kept pre-compaction
        assistant messages carry stale usage from the old (large) context.
        If the estimate anchor is from before the last compaction, skip.
        """
        if not self._session:
            return False

        entries = self._session.entries

        # Find the last compaction entry for anti-thrashing
        last_compaction = None
        for entry in reversed(entries):
            if entry.type == "compaction":
                last_compaction = entry
                break

        raw_messages = [
            m.model_dump(by_alias=True, exclude_none=True, mode="json")
            for m in build_session_context(entries).messages
        ]
        tokens = estimate_context_tokens(raw_messages)

        # Anti-thrashing: if the anchor assistant used for estimation is
        # from before the last compaction, its usage reflects old context
        # and would falsely trigger another compaction immediately.
        if last_compaction and last_compaction.first_kept_entry_id:
            # Find which assistant message was used as the token anchor
            anchor_ts = self._find_estimate_anchor_timestamp(raw_messages)
            if anchor_ts is not None:
                comp_ts = self._entry_timestamp_ms(last_compaction)
                if anchor_ts < comp_ts:
                    return False

        return should_compact(
            tokens, self._model.context_window, self._compaction_settings,
        )

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult | None:
        """Manually trigger compaction on the session."""
        if not self._session:
            return None
        if self._is_busy:
            raise RuntimeError("Agent is busy")

        entries = list(self._session.entries)
        preparation = prepare_compaction(entries, self._compaction_settings)
        if preparation is None:
            return None

        result = await _do_compact(
            preparation, self._model, self._provider,
            self._api_key, custom_instructions,
        )
        await self._session.append_compaction(
            result.summary, result.first_kept_entry_id, result.tokens_before,
            read_files=result.read_files, modified_files=result.modified_files,
        )
        return result

    async def auto_compact(self) -> CompactionResult | None:
        """Check and perform compaction if needed."""
        if self.check_compaction():
            return await self.compact()
        return None

    async def _auto_compact_if_needed(self) -> None:
        """Check, compact, and rebuild memory from session context."""
        result = await self.auto_compact()
        if result:
            # Rebuild memory from session to reflect compaction boundary
            ctx = build_session_context(self._session.entries)
            self._messages = list(ctx.messages)

    # ========================================================================
    # Event subscription
    # ========================================================================

    def on_event(
        self, callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> Callable[[], None]:
        """Register an event listener. Returns unsubscribe function."""
        self._event_listeners.append(callback)
        def _unsub() -> None:
            if callback in self._event_listeners:
                self._event_listeners.remove(callback)
        return _unsub

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    @tools.setter
    def tools(self, tools: list[AgentTool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model

    @property
    def is_streaming(self) -> bool:
        return self._is_busy

    @property
    def system_prompt(self) -> str:
        return self._build_system_prompt()

    @system_prompt.setter
    def system_prompt(self, prompt: str | Callable[[dict[str, Any]], str | Awaitable[str]]) -> None:
        self._system_prompt = prompt

    @property
    def session(self) -> Session | None:
        return self._session

    # ========================================================================
    # Private
    # ========================================================================

    def _build_context(self) -> AgentContext:
        return AgentContext(
            systemPrompt=self._build_system_prompt(),
            messages=list(self._messages),
            tools=list(self._tools.values()),
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt, injecting dynamic <env> and <tools> sections.

        If system_prompt is a callable, it receives a context dict for full
        customization and its return value is used directly (no auto-injection).
        """
        if callable(self._system_prompt):
            ctx = {
                "cwd": self._cwd,
                "model": self._model,
                "tools": list(self._tools.values()),
                "thinking_level": self._thinking_level,
                "session": self._session,
            }
            result = self._system_prompt(ctx)
            return result if isinstance(result, str) else str(result)

        return self._system_prompt + "\n\n" + self._generate_dynamic_context()

    def _generate_dynamic_context(self) -> str:
        """Generate the auto-injected <env> and <tools> XML sections."""
        parts: list[str] = []

        # <env> section
        parts.append("<env>")
        parts.append(f"Working directory: {self._cwd}")
        parts.append(f"Current date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        parts.append(f"Operating system: {platform.system()}")
        parts.append("</env>")

        # <tools> section
        if self._tools:
            parts.append("\n<tools>")
            for tool in self._tools.values():
                parts.append(f"- {tool.name}: {tool.description}")
            parts.append("</tools>")

        return "\n".join(parts)

    def _build_loop_config(self) -> AgentLoopConfig:
        return AgentLoopConfig(
            model=self._model,
            reasoning=self._thinking_level if self._thinking_level != "off" else None,
            apiKey=self._api_key,
            toolExecution=self._tool_execution,
            convertToLlm=self._convert_to_llm,
            transformContext=self._transform_context,
            getApiKey=self._get_api_key,
            beforeToolCall=self._before_tool_call,
            afterToolCall=self._after_tool_call,
            shouldStopAfterTurn=self._should_stop_after_turn,
            prepareNextTurn=self._prepare_next_turn,
            getSteeringMessages=self._get_steering_messages,
            getFollowUpMessages=self._get_follow_up_messages,
        )

    async def _get_steering_messages(self) -> list[AgentMessage]:
        if not self._steer_queue:
            return []
        if self._steering_mode == "one-at-a-time":
            msg = self._steer_queue.pop(0)
            return [msg]
        msgs = list(self._steer_queue)
        self._steer_queue.clear()
        return msgs

    async def _get_follow_up_messages(self) -> list[AgentMessage]:
        if not self._follow_up_queue:
            return []
        if self._follow_up_mode == "one-at-a-time":
            msg = self._follow_up_queue.pop(0)
            return [msg]
        msgs = list(self._follow_up_queue)
        self._follow_up_queue.clear()
        return msgs

    async def _emit(self, event: dict[str, Any]) -> None:
        for listener in self._event_listeners:
            try:
                await maybe_await(listener(event))
            except Exception:
                pass

    # ========================================================================
    # Overflow detection
    # ========================================================================

    @staticmethod
    def _is_context_overflow(msg: AgentMessage) -> bool:
        """Check if an assistant message indicates context overflow from the LLM."""
        role = getattr(msg, 'role', None)
        if role != "assistant":
            return False
        stop_reason = getattr(msg, 'stop_reason', '')
        if stop_reason != "error":
            return False
        error_msg = (getattr(msg, 'error_message', '') or '').lower()
        # Match known overflow indicators from OpenAI, Anthropic, etc.
        overflow_markers = [
            "context length", "maximum context", "context window",
            "prompt is too long", "reduce the length", "too many tokens",
            "max_tokens", "token limit",
        ]
        return any(marker in error_msg for marker in overflow_markers)

    # ========================================================================
    # Main run loop
    # ========================================================================

    async def _run(self, prompts: list[AgentMessage]) -> list[AgentMessage]:
        if self._is_busy:
            raise RuntimeError("Agent is busy")

        context = self._build_context()
        config = self._build_loop_config()
        overflow_recovery_attempted = False

        while True:
            async with self._run_lock:
                self._is_busy = True
                self._abort_event = asyncio.Event()
                try:
                    if overflow_recovery_attempted:
                        new_msgs = await run_agent_loop_continue(
                            context, config, self._emit,
                            provider=self._provider, signal=self._abort_event,
                        )
                    else:
                        new_msgs = await run_agent_loop(
                            prompts, context, config, self._emit,
                            provider=self._provider, signal=self._abort_event,
                        )
                    self._messages.extend(new_msgs)
                finally:
                    self._is_busy = False
                    self._abort_event = None

            # Find last assistant from just-returned messages
            last_assistant = self._find_last_assistant_in(new_msgs)

            # Overflow recovery: last assistant returned context-too-long error
            if (
                not overflow_recovery_attempted
                and last_assistant
                and self._is_context_overflow(last_assistant)
                and self._session
                and self._compaction_settings.enabled
            ):
                # Remove the error message from memory (don't persist it)
                self._messages.pop()
                overflow_recovery_attempted = True

                # Compact to free context space
                await self.compact()

                # Rebuild memory from session + re-add prompts for retry
                ctx = build_session_context(self._session.entries)
                self._messages = list(ctx.messages)
                for prompt in prompts:
                    self._messages.append(prompt)
                    await self._session.append_message(prompt)
                context = self._build_context()
                continue

            # Persist non-error messages to session
            if self._session:
                for msg in new_msgs:
                    await self._session.append_message(msg)

            # Threshold auto-compaction
            if self._session:
                await self._auto_compact_if_needed()

            return new_msgs

    def _find_last_assistant_in(self, messages: list[AgentMessage]) -> AgentMessage | None:
        """Find the last assistant message in a message list."""
        for msg in reversed(messages):
            role = getattr(msg, 'role', None)
            if role == "assistant":
                return msg
        return None

    @staticmethod
    def _find_estimate_anchor_timestamp(raw_messages: list[dict[str, Any]]) -> int | None:
        """Find the timestamp of the assistant message used as token estimate anchor.

        Mirrors the anchor-finding logic in utils.estimate_context_tokens:
        the last successful (non-error, non-aborted) assistant with usage data.
        """
        for i in range(len(raw_messages) - 1, -1, -1):
            msg = raw_messages[i]
            if msg.get("role") == "assistant" and msg.get("stopReason") not in ("aborted", "error"):
                usage = msg.get("usage")
                if usage:
                    return msg.get("timestamp")
        return None

    @staticmethod
    def _entry_timestamp_ms(entry: Any) -> int:
        """Convert a session entry's ISO timestamp to epoch milliseconds."""
        ts_str = getattr(entry, "timestamp", "")
        if ts_str:
            try:
                from datetime import datetime
                return int(datetime.fromisoformat(ts_str).timestamp() * 1000)
            except Exception:
                pass
        return 0
