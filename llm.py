"""
LLM provider abstraction.

Minimal interface: a Provider implements stream() and complete().
Two built-in implementations: OpenAI-compatible and Anthropic.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel, Field

from .agent_types import (
    AgentTool,
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)


# ============================================================================
# Model descriptor
# ============================================================================

class Model(BaseModel):
    id: str
    name: str = ""
    provider: str = "openai"
    base_url: str = Field(default="https://api.openai.com/v1", alias="baseUrl")
    api: str = "chat"
    reasoning: bool = False
    context_window: int = Field(default=128000, alias="contextWindow")
    max_tokens: int = Field(default=4096, alias="maxTokens")


# ============================================================================
# LLM context (what gets sent to the provider)
# ============================================================================

class LlmContext(BaseModel):
    system_prompt: str = Field(default="", alias="systemPrompt")
    messages: list[UserMessage | AssistantMessage | ToolResultMessage]
    tools: list[AgentTool] | None = None


# ============================================================================
# Stream event (mirrors TS AssistantMessageEvent union)
# ============================================================================

class StreamEvent(BaseModel):
    type: str  # start, text_start, text_delta, text_end, thinking_*, toolcall_*, done, error
    partial: Any = None  # partial AssistantMessage
    text: str | None = None
    tool_call_id: str | None = Field(default=None, alias="toolCallId")
    tool_name: str | None = Field(default=None, alias="toolName")
    arguments: str | None = None  # JSON string accumulated via deltas

    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Provider interface
# ============================================================================

class LlmProvider(ABC):
    """Minimal provider interface."""

    @abstractmethod
    async def stream(
        self, model: Model, context: LlmContext, *,
        api_key: str | None = None,
        signal: asyncio.Event | None = None,
        headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream assistant response. Must yield 'done' or 'error' at end."""
        ...

    @abstractmethod
    async def complete(
        self, model: Model, context: LlmContext, *,
        api_key: str | None = None,
        signal: asyncio.Event | None = None,
        headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
    ) -> AssistantMessage:
        """Non-streaming completion."""
        ...


# ============================================================================
# OpenAI-compatible provider
# ============================================================================

class OpenAiProvider(LlmProvider):
    """Works with OpenAI, Azure, and any OpenAI-compatible endpoint."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    def _build_chat_messages(
        self, messages: list[UserMessage | AssistantMessage | ToolResultMessage],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role
            if role == "user":
                result.append({"role": "user", "content": msg.content})
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                tool_calls: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingBlock):
                        pass  # OpenAI doesn't render thinking in requests
                    elif isinstance(block, ToolCallBlock):
                        blocks.append({"type": "text", "text": f"Calling tool: {block.name}"})
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.arguments)},
                        })
                payload: dict[str, Any] = {"role": "assistant"}
                if blocks:
                    payload["content"] = blocks if len(blocks) > 1 else blocks[0]["text"]
                else:
                    payload["content"] = None
                if tool_calls:
                    payload["tool_calls"] = tool_calls
                result.append(payload)
            elif role == "toolResult":
                content = msg.content
                text_parts = [
                    c.text for c in content
                    if hasattr(c, 'text')
                ]
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": "\n".join(text_parts) if text_parts else "(empty)",
                })
        return result

    def _build_tools(self, tools: list[AgentTool] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in tools
        ]

    async def stream(
        self, model: Model, context: LlmContext, *,
        api_key: str | None = None,
        signal: asyncio.Event | None = None,
        headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        url = f"{model.base_url.rstrip('/')}/chat/completions"

        payload: dict[str, Any] = {
            "model": model.id,
            "messages": self._build_chat_messages(context.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if context.tools:
            payload["tools"] = self._build_tools(context.tools)

        req_headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            req_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            req_headers.update(headers)

        ts = int(time.time() * 1000)
        partial = AssistantMessage(
            content=[],
            provider=model.provider,
            model=model.id,
            timestamp=ts,
        )

        # Build a partial assistant message gradually from stream deltas
        text_idx: dict[int, TextBlock] = {}  # content index -> TextBlock
        tool_idx: dict[int, ToolCallBlock] = {}  # content index -> ToolCallBlock
        next_idx = 0

        yield StreamEvent(type="start", partial=partial.model_copy(deep=True))

        try:
            async with client.stream(
                "POST", url, json=payload, headers=req_headers,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield StreamEvent(type="error", text=f"HTTP {response.status_code}: {body.decode()[:500]}")
                    return

                finish_reason: str | None = None
                usage_data: dict[str, Any] | None = None
                async for line in response.aiter_lines():
                    if signal and signal.is_set():
                        finish_reason = "aborted"
                        break
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason") or finish_reason
                    if data.get("usage"):
                        usage_data = data["usage"]

                    # Handle text
                    if "content" in delta and delta["content"]:
                        text = delta["content"]
                        idx = next_idx
                        # Check if previous block was also text
                        if idx > 0 and (idx - 1) in text_idx:
                            tb = text_idx[idx - 1]
                            new_tb = TextBlock(type="text", text=tb.text + text)
                            text_idx[idx - 1] = new_tb
                        else:
                            text_idx[idx] = TextBlock(type="text", text=text)
                            next_idx += 1

                        # Build partial state
                        content = list(text_idx.values()) + list(tool_idx.values())
                        partial.content = content
                        yield StreamEvent(
                            type="text_delta",
                            partial=partial.model_copy(deep=True),
                            text=text,
                        )

                    # Handle tool calls
                    tool_calls_delta = delta.get("tool_calls") or []
                    for tc in tool_calls_delta:
                        idx = tc.get("index", 0)
                        func = tc.get("function", {})
                        if idx in tool_idx:
                            existing = tool_idx[idx]
                            if "arguments" in func:
                                try:
                                    args = json.loads(existing.arguments_str + func["arguments"])
                                except json.JSONDecodeError:
                                    args = {}
                                new_args = {**existing.arguments, **args}
                                tool_idx[idx] = ToolCallBlock(
                                    type="toolCall",
                                    id=existing.id,
                                    name=existing.name,
                                    arguments=new_args,
                                )
                                tool_idx[idx].arguments_str = existing.arguments_str + func["arguments"]
                        else:
                            tool_idx[idx] = ToolCallBlock(
                                type="toolCall",
                                id=tc.get("id", ""),
                                name=func.get("name", ""),
                                arguments={},
                            )
                            tool_idx[idx].arguments_str = func.get("arguments", "")
                            if idx >= next_idx:
                                next_idx = idx + 1

                        content = list(text_idx.values()) + list(tool_idx.values())
                        partial.content = content
                        yield StreamEvent(
                            type="toolcall_delta",
                            partial=partial.model_copy(deep=True),
                            toolCallId=tool_idx[idx].id,
                            toolName=tool_idx[idx].name,
                            arguments=tool_idx[idx].arguments_str if hasattr(tool_idx[idx], 'arguments_str') else "",
                        )

            # Build final message
            content = list(text_idx.values()) + list(tool_idx.values())
            # Clean up arguments_str hack
            for tc_block in tool_idx.values():
                if hasattr(tc_block, 'arguments_str'):
                    delattr(tc_block, 'arguments_str')

            stop_reason = "end_turn"
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "length":
                stop_reason = "max_tokens"
            elif finish_reason == "aborted" or (signal and signal.is_set()):
                stop_reason = "aborted"

            usage = Usage()
            if usage_data:
                usage = Usage(
                    input=usage_data.get("prompt_tokens", 0),
                    output=usage_data.get("completion_tokens", 0),
                    cacheRead=usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    totalTokens=usage_data.get("total_tokens", 0),
                )
            final = AssistantMessage(
                content=content,
                provider=model.provider,
                model=model.id,
                stopReason=stop_reason,
                usage=usage,
                timestamp=int(time.time() * 1000),
            )
            yield StreamEvent(type="done", partial=final)

        except asyncio.CancelledError:
            yield StreamEvent(type="error", text="Cancelled")
        except Exception as e:
            final = AssistantMessage(
                content=[],
                provider=model.provider,
                model=model.id,
                stopReason="error",
                errorMessage=str(e),
                timestamp=int(time.time() * 1000),
            )
            yield StreamEvent(type="error", text=str(e), partial=final)

    async def complete(
        self, model: Model, context: LlmContext, *,
        api_key: str | None = None,
        signal: asyncio.Event | None = None,
        headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
    ) -> AssistantMessage:
        """Non-streaming completion."""
        ts = int(time.time() * 1000)
        final_msg: AssistantMessage | None = None
        async for event in self.stream(
            model, context, api_key=api_key, signal=signal,
            headers=headers, max_tokens=max_tokens, reasoning=reasoning,
        ):
            if event.type == "done" and event.partial:
                final_msg = event.partial
            elif event.type == "error":
                return AssistantMessage(
                    content=[],
                    provider=model.provider,
                    model=model.id,
                    stopReason="error",
                    errorMessage=event.text or "Unknown error",
                    timestamp=ts,
                )
        if final_msg:
            return final_msg
        return AssistantMessage(
            content=[TextBlock(type="text", text="")],
            provider=model.provider,
            model=model.id,
            stopReason="error",
            errorMessage="No response received",
            timestamp=ts,
        )


# ============================================================================
# Anthropic provider (minimal)
# ============================================================================

class AnthropicProvider(LlmProvider):
    """Minimal Anthropic Messages API provider."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._client

    def _build_messages(
        self, messages: list[UserMessage | AssistantMessage | ToolResultMessage],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.role
            if role == "user":
                result.append({"role": "user", "content": msg.content})
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingBlock):
                        blocks.append({"type": "thinking", "thinking": block.thinking})
                    elif isinstance(block, ToolCallBlock):
                        blocks.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments,
                        })
                result.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif role == "toolResult":
                text_parts = [
                    c.text for c in msg.content
                    if hasattr(c, 'text')
                ]
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": "\n".join(text_parts) if text_parts else "(empty)",
                    }],
                })
        return result

    def _build_tools(self, tools: list[AgentTool] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters_schema,
            }
            for t in tools
        ]

    async def stream(
        self, model: Model, context: LlmContext, *,
        api_key: str | None = None,
        signal: asyncio.Event | None = None,
        headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self._get_client()
        url = f"{model.base_url.rstrip('/')}/messages"

        payload: dict[str, Any] = {
            "model": model.id,
            "messages": self._build_messages(context.messages),
            "max_tokens": max_tokens or model.max_tokens,
            "stream": True,
        }
        if context.system_prompt:
            payload["system"] = context.system_prompt
        if context.tools:
            payload["tools"] = self._build_tools(context.tools)

        req_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            req_headers["x-api-key"] = api_key
        if headers:
            req_headers.update(headers)

        ts = int(time.time() * 1000)
        partial = AssistantMessage(
            content=[],
            provider=model.provider,
            model=model.id,
            timestamp=ts,
        )

        text_buf = ""
        thinking_buf = ""
        tool_blocks: dict[int, dict[str, Any]] = {}
        stop_reason = "end_turn"
        input_tokens = 0
        output_tokens = 0

        yield StreamEvent(type="start", partial=partial.model_copy(deep=True))

        try:
            async with client.stream(
                "POST", url, json=payload, headers=req_headers,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield StreamEvent(type="error", text=f"HTTP {response.status_code}: {body.decode()[:500]}")
                    return

                async for line in response.aiter_lines():
                    if signal and signal.is_set():
                        stop_reason = "aborted"
                        break
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type", "")
                    if event_type == "message_stop":
                        break

                    delta = data.get("delta", {}) or data
                    stop_reason_map = {
                        "end_turn": "end_turn", "tool_use": "tool_use",
                        "max_tokens": "max_tokens", "stop_sequence": "end_turn",
                    }
                    if "message" in data:
                        sr = data["message"].get("stop_reason")
                        if sr:
                            stop_reason = stop_reason_map.get(sr, sr)
                        msg_usage = data["message"].get("usage", {})
                        if msg_usage:
                            input_tokens = msg_usage.get("input_tokens", 0)
                    elif event_type == "message_delta":
                        delta_usage = data.get("usage", {})
                        if delta_usage:
                            output_tokens = delta_usage.get("output_tokens", 0)

                    if event_type == "content_block_start":
                        block = data.get("content_block", {})
                        bt = block.get("type", "")
                        idx = block.get("index", data.get("index", 0))
                        if bt == "text":
                            pass  # wait for content_block_delta
                        elif bt == "tool_use":
                            tool_blocks[idx] = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "input": "",
                            }
                    elif event_type == "content_block_delta":
                        bt = delta.get("type", "")
                        idx = data.get("index", 0)
                        if bt == "text_delta":
                            text_buf += delta.get("text", "")
                        elif bt == "thinking_delta":
                            thinking_buf += delta.get("thinking", "")
                        elif bt == "input_json_delta":
                            if idx in tool_blocks:
                                tool_blocks[idx]["input"] += delta.get("partial_json", "")

                        # Build partial
                        content: list[Any] = []
                        if thinking_buf:
                            content.append(ThinkingBlock(type="thinking", thinking=thinking_buf))
                        if text_buf:
                            content.append(TextBlock(type="text", text=text_buf))
                        for tb in tool_blocks.values():
                            try:
                                args = json.loads(tb["input"])
                            except json.JSONDecodeError:
                                args = {}
                            content.append(ToolCallBlock(
                                type="toolCall", id=tb["id"], name=tb["name"], arguments=args,
                            ))
                        partial.content = content
                        yield StreamEvent(type="text_delta", partial=partial.model_copy(deep=True))

            # Build final
            content = []
            if thinking_buf:
                content.append(ThinkingBlock(type="thinking", thinking=thinking_buf))
            if text_buf:
                content.append(TextBlock(type="text", text=text_buf))
            for tb in tool_blocks.values():
                try:
                    args = json.loads(tb["input"])
                except json.JSONDecodeError:
                    args = {}
                content.append(ToolCallBlock(type="toolCall", id=tb["id"], name=tb["name"], arguments=args))

            final = AssistantMessage(
                content=content,
                provider=model.provider,
                model=model.id,
                stopReason=stop_reason,
                usage=Usage(input=input_tokens, output=output_tokens, totalTokens=input_tokens + output_tokens),
                timestamp=int(time.time() * 1000),
            )
            yield StreamEvent(type="done", partial=final)

        except asyncio.CancelledError:
            yield StreamEvent(type="error", text="Cancelled")
        except Exception as e:
            yield StreamEvent(
                type="error", text=str(e),
                partial=AssistantMessage(
                    content=[], provider=model.provider, model=model.id,
                    stopReason="error", errorMessage=str(e), timestamp=int(time.time() * 1000),
                ),
            )

    async def complete(
        self, model: Model, context: LlmContext, *,
        api_key: str | None = None,
        signal: asyncio.Event | None = None,
        headers: dict[str, str] | None = None,
        max_tokens: int | None = None,
        reasoning: str | None = None,
    ) -> AssistantMessage:
        ts = int(time.time() * 1000)
        final_msg: AssistantMessage | None = None
        async for event in self.stream(
            model, context, api_key=api_key, signal=signal,
            headers=headers, max_tokens=max_tokens, reasoning=reasoning,
        ):
            if event.type == "done" and event.partial:
                final_msg = event.partial
            elif event.type == "error":
                return AssistantMessage(
                    content=[],
                    provider=model.provider,
                    model=model.id,
                    stopReason="error",
                    errorMessage=event.text or "Unknown error",
                    timestamp=ts,
                )
        if final_msg:
            return final_msg
        return AssistantMessage(
            content=[TextBlock(type="text", text="")],
            provider=model.provider, model=model.id,
            stopReason="error", errorMessage="No response received", timestamp=ts,
        )
