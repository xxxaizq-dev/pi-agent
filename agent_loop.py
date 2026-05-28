"""
Stateless agent loop engine.

Double-loop architecture:
  Outer: follow-up messages keep the agent alive
  Inner: steering messages + assistant turn + tool execution

Matches TS agent-loop.ts:95-746.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Callable, Awaitable

from .utils import maybe_await
from .llm import (
    LlmContext,
    LlmProvider,
    Model,
    OpenAiProvider,
    StreamEvent,
)
from .utils import estimate_context_tokens
from .agent_types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    ShouldStopAfterTurnContext,
    TextBlock,
    TextContent,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
    make_tool_result_message,
)

# ============================================================================
# Event sink
# ============================================================================

AgentEventSink = Callable[[dict[str, Any]], Awaitable[None]]


# ============================================================================
# Public entry points
# ============================================================================

async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    *,
    provider: LlmProvider | None = None,
    signal: asyncio.Event | None = None,
    stream_fn: Callable | None = None,
) -> list[AgentMessage]:
    """Start an agent loop with prompt messages."""
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        systemPrompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=list(context.tools),
    )

    await emit({"type": "agent_start"})
    await emit({"type": "turn_start"})
    for prompt in prompts:
        await emit({"type": "message_start", "message": prompt})
        await emit({"type": "message_end", "message": prompt})

    await _run_loop(
        current_context, new_messages, config, emit,
        provider=provider, signal=signal, stream_fn=stream_fn,
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    *,
    provider: LlmProvider | None = None,
    signal: asyncio.Event | None = None,
    stream_fn: Callable | None = None,
) -> list[AgentMessage]:
    """Continue from context without adding prompts."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last_msg = context.messages[-1]
    if hasattr(last_msg, 'role') and last_msg.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        systemPrompt=context.system_prompt,
        messages=list(context.messages),
        tools=list(context.tools),
    )

    await emit({"type": "agent_start"})
    await emit({"type": "turn_start"})

    await _run_loop(
        current_context, new_messages, config, emit,
        provider=provider, signal=signal, stream_fn=stream_fn,
    )
    return new_messages


# ============================================================================
# Main loop (double-loop)
# ============================================================================

async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    emit: AgentEventSink,
    *,
    provider: LlmProvider | None = None,
    signal: asyncio.Event | None = None,
    stream_fn: Callable | None = None,
) -> None:
    current_context = initial_context
    config = initial_config
    first_turn = True

    # Poll steering messages at start
    pending_messages: list[AgentMessage] = []
    if config.get_steering_messages:
        ms = await maybe_await(config.get_steering_messages())
        pending_messages = ms or []

    # Outer loop: follow-up messages restart after agent would stop
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await emit({"type": "turn_start"})
            else:
                first_turn = False

            # Inject pending messages
            if pending_messages:
                for msg in pending_messages:
                    await emit({"type": "message_start", "message": msg})
                    await emit({"type": "message_end", "message": msg})
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            if signal and signal.is_set():
                await emit({"type": "agent_end", "messages": new_messages})
                return

            # Stream assistant response
            message = await _stream_assistant_response(
                current_context, config, emit,
                provider=provider, signal=signal, stream_fn=stream_fn,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await emit({"type": "turn_end", "message": message, "toolResults": []})
                await emit({"type": "agent_end", "messages": new_messages})
                return

            # Extract tool calls
            tool_calls = [
                b for b in message.content
                if isinstance(b, ToolCallBlock)
            ]

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                batch = await _execute_tool_calls(
                    current_context, message, tool_calls, config, emit,
                    signal=signal, provider=provider,
                )
                tool_results = batch["messages"]
                has_more_tool_calls = not batch["terminate"]
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await emit({"type": "turn_end", "message": message, "toolResults": tool_results})

            # prepareNextTurn
            if config.prepare_next_turn:
                turn_ctx = {
                    "message": message,
                    "toolResults": tool_results,
                    "context": current_context,
                    "newMessages": new_messages,
                }
                update = await maybe_await(config.prepare_next_turn(turn_ctx))
                if update:
                    if update.get("context"):
                        current_context = update["context"]
                    if update.get("model"):
                        config.model = update["model"]
                    if update.get("thinkingLevel") is not None:
                        level = update["thinkingLevel"]
                        config.reasoning = level if level != "off" else None

            # shouldStopAfterTurn
            if config.should_stop_after_turn:
                stop_ctx = {
                    "message": message,
                    "toolResults": tool_results,
                    "context": current_context,
                    "newMessages": new_messages,
                }
                if await maybe_await(config.should_stop_after_turn(stop_ctx)):
                    await emit({"type": "agent_end", "messages": new_messages})
                    return

            # Poll steering
            if config.get_steering_messages:
                ms = await maybe_await(config.get_steering_messages())
                pending_messages = ms or []
            else:
                pending_messages = []

        # Agent would stop. Check follow-up.
        if config.get_follow_up_messages:
            follow_up = await maybe_await(config.get_follow_up_messages())
            if follow_up:
                pending_messages = follow_up
                continue

        break

    await emit({"type": "agent_end", "messages": new_messages})


# ============================================================================
# Stream assistant response
# ============================================================================

async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    *,
    provider: LlmProvider | None = None,
    signal: asyncio.Event | None = None,
    stream_fn: Callable | None = None,
) -> AssistantMessage:
    """Stream assistant response. AgentMessage[] → LlmContext at this boundary."""
    messages = list(context.messages)

    # Transform context (optional pre-processing)
    if config.transform_context:
        messages = await maybe_await(config.transform_context(messages, signal))

    # Convert to LLM messages
    llm_messages = await maybe_await(config.convert_to_llm(messages))

    # Build LLM context
    llm_ctx = LlmContext(
        systemPrompt=context.system_prompt,
        messages=llm_messages,
        tools=list(context.tools),
    )

    # Resolve API key
    api_key = config.api_key
    if config.get_api_key:
        api_key = await maybe_await(config.get_api_key(config.model.provider)) or api_key

    # Pre-flight context window check
    estimated = estimate_context_tokens([
        m.model_dump(by_alias=True, exclude_none=True, mode="json")
        for m in llm_messages
    ])
    if estimated > config.model.context_window:
        return AssistantMessage(
            content=[TextBlock(
                type="text",
                text=(
                    f"Context window exceeded: estimated {estimated} tokens "
                    f"but model supports {config.model.context_window}. "
                    f"Trigger compaction via agent.compact() or auto_compact()."
                ),
            )],
            provider=config.model.provider,
            model=config.model.id,
            stopReason="error",
            errorMessage="Context window exceeded",
            timestamp=int(time.time() * 1000),
        )

    # Use provider
    prov = provider or OpenAiProvider()

    # Use custom stream_fn or provider.stream
    if stream_fn:
        # Custom stream function gets model + llm_ctx + options
        response = await stream_fn(config.model, llm_ctx, {
            "apiKey": api_key,
            "signal": signal,
            "headers": config.headers,
            "maxTokens": config.model.max_tokens,
        })
    else:
        response = prov.stream(
            config.model, llm_ctx,
            api_key=api_key,
            signal=signal,
            headers=config.headers,
            max_tokens=config.model.max_tokens,
            reasoning=config.reasoning,
        )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if signal and signal.is_set():
            break

        if event.type == "start":
            partial_message = event.partial
            if partial_message:
                context.messages.append(partial_message)
                added_partial = True
                await emit({
                    "type": "message_start",
                    "message": partial_message.model_copy(deep=True),
                })

        elif event.type in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial_message and event.partial:
                partial_message = event.partial
                if added_partial:
                    context.messages[-1] = partial_message
                await emit({
                    "type": "message_update",
                    "assistantMessageEvent": event.model_dump(mode="json", exclude_none=True),
                    "message": partial_message.model_copy(deep=True),
                })

        elif event.type in ("done", "error"):
            if event.partial:
                final = event.partial
            else:
                final = partial_message or AssistantMessage(
                    content=[], provider=config.model.provider,
                    model=config.model.id, stopReason="error",
                    errorMessage="No response", timestamp=int(time.time() * 1000),
                )
            if added_partial:
                context.messages[-1] = final
            else:
                context.messages.append(final)
                await emit({"type": "message_start", "message": final.model_copy(deep=True)})
            await emit({"type": "message_end", "message": final})
            return final

    # Fallback (should not reach here)
    if partial_message:
        await emit({"type": "message_end", "message": partial_message})
        return partial_message
    empty = AssistantMessage(
        content=[TextBlock(type="text", text="")],
        provider=config.model.provider, model=config.model.id,
        stopReason="end_turn", timestamp=int(time.time() * 1000),
    )
    await emit({"type": "message_end", "message": empty})
    return empty


# ============================================================================
# Tool execution dispatch
# ============================================================================

async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    *,
    signal: asyncio.Event | None = None,
    provider: LlmProvider | None = None,
) -> dict[str, Any]:
    """Dispatch to sequential or parallel execution based on config and tool modes."""
    has_sequential = any(
        _find_tool(current_context, tc.name) is not None
        and _find_tool(current_context, tc.name).execution_mode == "sequential"
        for tc in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(
            current_context, assistant_message, tool_calls, config, emit, signal,
        )
    return await _execute_parallel(
        current_context, assistant_message, tool_calls, config, emit, signal,
    )


def _find_tool(context: AgentContext, name: str) -> AgentTool | None:
    for t in context.tools:
        if t.name == name:
            return t
    return None


# ============================================================================
# Sequential execution
# ============================================================================

async def _execute_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> dict[str, Any]:
    messages: list[ToolResultMessage] = []
    finalized_calls: list[_Finalized] = []

    for tc in tool_calls:
        await emit({
            "type": "tool_execution_start",
            "toolCallId": tc.id, "toolName": tc.name, "args": tc.arguments,
        })

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tc, config, signal,
        )
        if preparation["kind"] == "immediate":
            finalized: _Finalized = {
                "toolCall": tc,
                "result": preparation["result"],
                "isError": preparation["isError"],
            }
        else:
            executed = await _execute_prepared(preparation, signal, emit)
            finalized = await _finalize_tool_call(
                current_context, assistant_message, preparation, executed, config, signal,
            )

        await _emit_tool_execution_end(finalized, emit)
        trm = _create_tool_result_message(finalized)
        await emit({"type": "message_start", "message": trm})
        await emit({"type": "message_end", "message": trm})
        finalized_calls.append(finalized)
        messages.append(trm)

        if signal and signal.is_set():
            break

    return {
        "messages": messages,
        "terminate": _should_terminate(finalized_calls),
    }


# ============================================================================
# Parallel execution
# ============================================================================

async def _execute_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallBlock],
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> dict[str, Any]:
    entries: list[_Finalized | Callable[[], Awaitable[_Finalized]]] = []

    for tc in tool_calls:
        await emit({
            "type": "tool_execution_start",
            "toolCallId": tc.id, "toolName": tc.name, "args": tc.arguments,
        })

        preparation = await _prepare_tool_call(
            current_context, assistant_message, tc, config, signal,
        )
        if preparation["kind"] == "immediate":
            finalized: _Finalized = {
                "toolCall": tc,
                "result": preparation["result"],
                "isError": preparation["isError"],
            }
            await _emit_tool_execution_end(finalized, emit)
            entries.append(finalized)
            if signal and signal.is_set():
                break
            continue

        async def _thunk(p=preparation) -> _Finalized:
            execd = await _execute_prepared(p, signal, emit)
            final = await _finalize_tool_call(
                current_context, assistant_message, p, execd, config, signal,
            )
            await _emit_tool_execution_end(final, emit)
            return final

        entries.append(_thunk)
        if signal and signal.is_set():
            break

    # Resolve all (sync + async) in parallel
    ordered: list[_Finalized] = []
    for entry in entries:
        if callable(entry):
            ordered.append(await entry())
        else:
            ordered.append(entry)

    messages: list[ToolResultMessage] = []
    for fin in ordered:
        trm = _create_tool_result_message(fin)
        await emit({"type": "message_start", "message": trm})
        await emit({"type": "message_end", "message": trm})
        messages.append(trm)

    return {
        "messages": messages,
        "terminate": _should_terminate(ordered),
    }


# ============================================================================
# Internal types for tool execution pipeline
# ============================================================================

class _Prepared:
    kind: str = "prepared"
    toolCall: ToolCallBlock
    tool: AgentTool
    args: Any


class _Immediate:
    kind: str = "immediate"
    result: AgentToolResult
    isError: bool


_Preparation = _Prepared | _Immediate


class _Executed:
    result: AgentToolResult
    isError: bool


class _Finalized:
    toolCall: ToolCallBlock
    result: AgentToolResult
    isError: bool


# ============================================================================
# Tool call pipeline: prepare → execute → finalize
# ============================================================================

def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
    )


def _create_tool_result_message(finalized: _Finalized) -> ToolResultMessage:
    return make_tool_result_message(
        finalized["toolCall"].id,
        finalized["toolCall"].name,
        "".join(
            c.text if hasattr(c, 'text') else str(c)
            for c in finalized["result"].content
        ),
        details=finalized["result"].details,
        is_error=finalized["isError"],
    )


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallBlock,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> dict[str, Any]:
    tool = _find_tool(current_context, tool_call.name)
    if tool is None:
        return {
            "kind": "immediate",
            "result": _create_error_tool_result(f"Tool {tool_call.name} not found"),
            "isError": True,
        }

    try:
        # prepareArguments
        args = tool_call.arguments
        if hasattr(tool, 'prepare_arguments') and tool.prepare_arguments:
            prep_args = tool.prepare_arguments(args)
            if prep_args is not args:
                args = prep_args

        if signal and signal.is_set():
            return {
                "kind": "immediate",
                "result": _create_error_tool_result("Operation aborted"),
                "isError": True,
            }

        if config.before_tool_call:
            before_ctx = {
                "assistantMessage": assistant_message,
                "toolCall": tool_call,
                "args": args,
                "context": current_context,
            }
            before_result = await maybe_await(config.before_tool_call(before_ctx, signal))
            if signal and signal.is_set():
                return {
                    "kind": "immediate",
                    "result": _create_error_tool_result("Operation aborted"),
                    "isError": True,
                }
            if before_result and before_result.get("block"):
                return {
                    "kind": "immediate",
                    "result": _create_error_tool_result(
                        before_result.get("reason", "Tool execution was blocked")
                    ),
                    "isError": True,
                }

        if signal and signal.is_set():
            return {
                "kind": "immediate",
                "result": _create_error_tool_result("Operation aborted"),
                "isError": True,
            }

        prep = _Prepared()
        prep.kind = "prepared"
        prep.toolCall = tool_call
        prep.tool = tool
        prep.args = args
        return {"kind": "prepared", "data": prep}

    except Exception as e:
        return {
            "kind": "immediate",
            "result": _create_error_tool_result(str(e)),
            "isError": True,
        }


async def _execute_prepared(
    preparation: dict[str, Any],
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> dict[str, Any]:
    prep = preparation["data"]
    update_events: list[Awaitable[None]] = []

    async def _on_update(partial_result: Any) -> None:
        update_events.append(
            emit({
                "type": "tool_execution_update",
                "toolCallId": prep.toolCall.id,
                "toolName": prep.toolCall.name,
                "args": prep.toolCall.arguments,
                "partialResult": partial_result,
            })
        )

    try:
        execute_fn = prep.tool.execute
        # Support both sync and async execute
        if asyncio.iscoroutinefunction(execute_fn) or hasattr(execute_fn, '__call__'):
            result = await execute_fn(
                prep.toolCall.id, prep.args, signal, _on_update,
            )
        else:
            result = execute_fn(
                prep.toolCall.id, prep.args, signal, _on_update,
            )
        # Ensure update events are processed
        for ue in update_events:
            await ue
        return {"result": result, "isError": False}
    except Exception as e:
        for ue in update_events:
            await ue
        return {
            "result": _create_error_tool_result(str(e)),
            "isError": True,
        }


async def _finalize_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    preparation: dict[str, Any],
    executed: dict[str, Any],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> dict[str, Any]:
    prep = preparation["data"]
    result = executed["result"]
    is_error = executed["isError"]

    if config.after_tool_call:
        try:
            after_ctx = {
                "assistantMessage": assistant_message,
                "toolCall": prep.toolCall,
                "args": prep.args,
                "result": result,
                "isError": is_error,
                "context": current_context,
            }
            after_result = await maybe_await(config.after_tool_call(after_ctx, signal))
            if after_result:
                if after_result.get("content") is not None:
                    result.content = after_result["content"]
                if after_result.get("details") is not None:
                    result.details = after_result["details"]
                if after_result.get("terminate") is not None:
                    result.terminate = after_result["terminate"]
                if after_result.get("isError") is not None:
                    is_error = after_result["isError"]
        except Exception as e:
            result = _create_error_tool_result(str(e))
            is_error = True

    return {
        "toolCall": prep.toolCall,
        "result": result,
        "isError": is_error,
    }


async def _emit_tool_execution_end(finalized: _Finalized, emit: AgentEventSink) -> None:
    await emit({
        "type": "tool_execution_end",
        "toolCallId": finalized["toolCall"].id,
        "toolName": finalized["toolCall"].name,
        "result": finalized["result"],
        "isError": finalized["isError"],
    })


def _should_terminate(finalized_calls: list[_Finalized]) -> bool:
    return (
        len(finalized_calls) > 0
        and all(f["result"].terminate for f in finalized_calls)
    )
