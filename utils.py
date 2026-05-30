"""
Token estimation, UUIDv7 generation, and conversation serialization utilities.
"""

import math
import os
import json
import time
import inspect
from typing import Any, Awaitable


# ---------------------------------------------------------------------------
# UUIDv7 (time-ordered, matches TS behaviour)
# ---------------------------------------------------------------------------

def sanitize_surrogates(obj: Any) -> Any:
    """Recursively replace lone surrogates in strings with U+FFFD."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_surrogates(item) for item in obj]
    return obj


async def maybe_await(fn_result: Any) -> Any:
    """Await if coroutine, pass through otherwise. Mirrors TS `await` on non-Promise."""
    if inspect.iscoroutine(fn_result) or inspect.isawaitable(fn_result):
        return await fn_result
    return fn_result


def uuid_v7() -> str:
    """Generate a time-ordered UUIDv7 string without external dependencies."""
    # 48-bit timestamp in ms (big-endian)
    ts = int(time.time() * 1000)
    # 74 random bits (12 from rand_a, 62 from rand_b)
    rand = os.urandom(10)
    rand_a = rand[:2]
    rand_b = rand[2:]

    ts_bytes = ts.to_bytes(6, "big")
    # version 7: set high nibble of byte 6
    b6 = ts_bytes[6] if len(ts_bytes) > 6 else rand_a[0]  # unused
    _ = b6  # unused — ts fills exactly 6 bytes
    # Build: ts (6) + rand_a (2) + rand_b (6) = 14 bytes
    combined = ts_bytes + rand_a + rand_b
    # Version (4 bits) at byte 6 high nibble: 0x70
    combined = combined[:6] + bytes([(combined[6] & 0x0F) | 0x70]) + combined[7:]
    # Variant (2 bits) at byte 8 high bits: 10xx_xxxx
    combined = combined[:8] + bytes([(combined[8] & 0x3F) | 0x80]) + combined[9:]

    hex_str = combined.hex()
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


# ---------------------------------------------------------------------------
# Token estimation (chars/4 heuristic, mirrors TS estimateTokens)
# ---------------------------------------------------------------------------

def estimate_tokens(message: dict[str, Any]) -> int:
    """Conservative token estimate: total_character_count / 4 (ceil)."""
    role = message.get("role", "")

    if role == "user":
        content = message.get("content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            chars = sum(
                len(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            chars = 0
        return math.ceil(chars / 4)

    if role == "assistant":
        chars = 0
        for block in message.get("content", []) or []:
            t = block.get("type", "")
            if t == "text":
                chars += len(block.get("text", ""))
            elif t == "thinking":
                chars += len(block.get("thinking", ""))
            elif t == "toolCall":
                chars += len(block.get("name", "")) + len(_safe_json(block.get("arguments", {})))
        return math.ceil(chars / 4)

    if role in ("custom", "toolResult"):
        content = message.get("content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            chars = 0
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        chars += len(block.get("text", ""))
                    elif block.get("type") == "image":
                        chars += 4800
        else:
            chars = 0
        return math.ceil(chars / 4)

    if role == "bashExecution":
        chars = len(message.get("command", "")) + len(message.get("output", ""))
        return math.ceil(chars / 4)

    if role in ("branchSummary", "compactionSummary"):
        chars = len(message.get("summary", ""))
        return math.ceil(chars / 4)

    return 0


def estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    """
    Estimate total context tokens.
    Uses the last successful assistant's usage.totalTokens as an anchor,
    plus estimated tokens for messages after it.
    """
    last_usage = None
    last_usage_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("stopReason") not in ("aborted", "error"):
            usage = msg.get("usage")
            if usage:
                last_usage = usage
                last_usage_idx = i
                break

    if last_usage is None:
        return sum(estimate_tokens(m) for m in messages)

    total = last_usage.get("totalTokens", 0) or (
        last_usage.get("input", 0) + last_usage.get("output", 0)
        + last_usage.get("cacheRead", 0) + last_usage.get("cacheWrite", 0)
    )
    for i in range(last_usage_idx + 1, len(messages)):
        total += estimate_tokens(messages[i])
    return total


# ---------------------------------------------------------------------------
# Conversation serialization (for compaction prompts)
# ---------------------------------------------------------------------------

TOOL_RESULT_MAX_CHARS = 2000


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "[unserializable]"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    removed = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {removed} more characters truncated]"


def serialize_conversation(messages: list[dict[str, Any]]) -> str:
    """Convert LLM messages to plain text for summarization prompts."""
    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            if text:
                parts.append(f"[User]: {text}")

        elif role == "assistant":
            texts: list[str] = []
            thinkings: list[str] = []
            tool_calls: list[str] = []
            for block in msg.get("content", []) or []:
                t = block.get("type", "")
                if t == "text":
                    texts.append(block.get("text", ""))
                elif t == "thinking":
                    thinkings.append(block.get("thinking", ""))
                elif t == "toolCall":
                    args = block.get("arguments", {})
                    args_str = ", ".join(f"{k}={_safe_json(v)}" for k, v in (args.items() if isinstance(args, dict) else {}))
                    tool_calls.append(f"{block.get('name', 'unknown')}({args_str})")

            if thinkings:
                parts.append(f"[Assistant thinking]: {'\n'.join(thinkings)}")
            if texts:
                parts.append(f"[Assistant]: {'\n'.join(texts)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")

        elif role == "toolResult":
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                text = ""
            if text:
                parts.append(f"[Tool result]: {_truncate(text, TOOL_RESULT_MAX_CHARS)}")

    return "\n\n".join(parts)
