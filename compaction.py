"""
Context compaction engine.

Pure functions for: shouldCompact, findCutPoint, generateSummary, prepareCompaction, compact.
Matches the TS harness/compaction/compaction.ts logic.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from .llm import LlmProvider, Model
from .session import build_session_context, SessionEntry
from .agent_types import (
    AgentMessage,
    AssistantMessage,
    CompactionPreparation,
    CompactionResult,
    CompactionSettings,
    CutPointResult,
    DEFAULT_COMPACTION_SETTINGS,
    FileOperations,
    TextBlock,
    TextContent,
    ToolCallBlock,
    UserMessage,
    make_compaction_summary_message,
)
from .utils import (
    estimate_context_tokens,
    estimate_tokens,
    serialize_conversation,
)


# ============================================================================
# Summarization prompts (exact copies from TS)
# ============================================================================

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = (
    "The messages above are a conversation to summarize. Create a structured context "
    "checkpoint summary that another LLM will use to continue the work.\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the session "
    "covers different tasks.]\n\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    "- [Or \"(none)\" if none were mentioned]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    "- [Or \"(none)\" if not applicable]\n\n"
    "Keep each section concise. Preserve exact file paths, function names, and error messages."
)

UPDATE_SUMMARIZATION_PROMPT = (
    "The messages above are NEW conversation messages to incorporate into the existing "
    "summary provided in <previous-summary> tags.\n\n"
    "Update the existing structured summary with new information. RULES:\n"
    "- PRESERVE all existing information from the previous summary\n"
    "- ADD new progress, decisions, and context from the new messages\n"
    "- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n"
    "- UPDATE \"Next Steps\" based on what was accomplished\n"
    "- PRESERVE exact file paths, function names, and error messages\n"
    "- If something is no longer relevant, you may remove it\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[Preserve existing goals, add new ones if the task expanded]\n\n"
    "## Constraints & Preferences\n"
    "- [Preserve existing, add new ones discovered]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Include previously done items AND newly completed items]\n\n"
    "### In Progress\n"
    "- [ ] [Current work - update based on progress]\n\n"
    "### Blocked\n"
    "- [Current blockers - remove if resolved]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n"
    "## Next Steps\n"
    "1. [Update based on current state]\n\n"
    "## Critical Context\n"
    "- [Preserve important context, add new if needed]\n\n"
    "Keep each section concise. Preserve exact file paths, function names, and error messages."
)

TURN_PREFIX_SUMMARIZATION_PROMPT = (
    "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\n"
    "Summarize the prefix to provide context for the retained suffix:\n\n"
    "## Original Request\n"
    "[What did the user ask for in this turn?]\n\n"
    "## Early Progress\n"
    "- [Key decisions and work done in the prefix]\n\n"
    "## Context for Suffix\n"
    "- [Information needed to understand the retained recent work]\n\n"
    "Be concise. Focus on what's needed to understand the kept suffix."
)


# ============================================================================
# Token helpers
# ============================================================================

def should_compact(
    context_tokens: int, context_window: int, settings: CompactionSettings,
) -> bool:
    """Check if compaction should trigger based on context token usage."""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


def _safe_json(obj: Any) -> str:
    import json as _json
    try:
        return _json.dumps(obj, default=str)
    except Exception:
        return "[unserializable]"


# ============================================================================
# File operation extraction
# ============================================================================

def _create_file_ops() -> FileOperations:
    return FileOperations(read=set(), written=set(), edited=set())


def _extract_file_ops_from_message(msg: Any, file_ops: FileOperations) -> None:
    """Scrape assistant tool call blocks for file paths."""
    role = getattr(msg, 'role', None)
    if role != "assistant":
        return
    content = getattr(msg, 'content', []) or []
    for block in content:
        if not isinstance(block, ToolCallBlock):
            continue
        name = block.name if hasattr(block, 'name') else getattr(block, 'name', '')
        args = block.arguments if hasattr(block, 'arguments') else getattr(block, 'arguments', {}) or {}
        path = args.get("path") if isinstance(args, dict) else None
        if not path or not isinstance(path, str):
            continue
        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def _extract_file_operations(
    messages: list[AgentMessage],
    entries: list[SessionEntry],
    prev_compaction_index: int,
) -> FileOperations:
    file_ops = _create_file_ops()
    # Inherit from previous compaction's details
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        read_files = prev.read_files or []
        modified_files = prev.modified_files or []
        for f in read_files:
            file_ops.read.add(f)
        for f in modified_files:
            file_ops.edited.add(f)
    for msg in messages:
        _extract_file_ops_from_message(msg, file_ops)
    return file_ops


def _compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    modified = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    return read_only, sorted(modified)


def _format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ============================================================================
# Cut point detection
# ============================================================================

def _find_valid_cut_points(
    entries: list[SessionEntry], start_index: int, end_index: int,
) -> list[int]:
    """Enumerate safe cut indices (user, assistant, bashExecution, custom, branch_summary)."""
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.type == "message" and entry.message is not None:
            role = getattr(entry.message, 'role', None)
            if role in ("user", "assistant", "bashExecution", "custom", "branchSummary", "compactionSummary"):
                cut_points.append(i)
        elif entry.type in ("branch_summary", "custom_message"):
            cut_points.append(i)
    return cut_points


def _find_turn_start(
    entries: list[SessionEntry], entry_index: int, start_index: int,
) -> int:
    """Find the user message that starts the turn containing the given entry index."""
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.type in ("branch_summary", "custom_message"):
            return i
        if entry.type == "message" and entry.message is not None:
            role = getattr(entry.message, 'role', None)
            if role in ("user", "bashExecution"):
                return i
    return -1


def find_cut_point(
    entries: list[SessionEntry], start_index: int, end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """
    Find the compaction cut point that keeps approximately keep_recent_tokens
    of recent context. Walk backwards from newest, accumulating token estimates.
    """
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index)

    accumulated = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.type != "message" or entry.message is None:
            continue
        accumulated += estimate_tokens(entry.message.model_dump(by_alias=True, exclude_none=True))
        if accumulated >= keep_recent_tokens:
            for cp in cut_points:
                if cp >= i:
                    cut_index = cp
                    break
            break

    # Slide backwards past non-message entries
    while cut_index > start_index:
        prev = entries[cut_index - 1]
        if prev.type == "compaction":
            break
        if prev.type == "message":
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_msg = (cut_entry.type == "message" and cut_entry.message is not None
                   and getattr(cut_entry.message, 'role', None) == "user")
    turn_start = -1 if is_user_msg else _find_turn_start(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=(not is_user_msg and turn_start != -1),
    )


# ============================================================================
# Message extraction from entries
# ============================================================================

def _get_message_from_entry(entry: SessionEntry) -> AgentMessage | None:
    if entry.type == "message" and entry.message is not None:
        return entry.message
    if entry.type == "custom_message" and entry.content is not None:
        ts = 0
        if entry.timestamp:
            try:
                from datetime import datetime as _dt
                ts = int(_dt.fromisoformat(entry.timestamp).timestamp() * 1000)
            except Exception:
                pass
        from .agent_types import CustomMessage as CM
        return CM(
            customType=entry.custom_type or "custom",
            content=entry.content,
            display=entry.display if entry.display is not None else True,
            details=entry.details,
            timestamp=ts,
        )
    if entry.type == "branch_summary" and entry.summary:
        ts = 0
        if entry.timestamp:
            try:
                from datetime import datetime as _dt
                ts = int(_dt.fromisoformat(entry.timestamp).timestamp() * 1000)
            except Exception:
                pass
        # Use compaction summary as placeholder for branch summary
        return make_compaction_summary_message(entry.summary, 0, ts)
    if entry.type == "compaction" and entry.summary:
        ts = 0
        if entry.timestamp:
            try:
                from datetime import datetime as _dt
                ts = int(_dt.fromisoformat(entry.timestamp).timestamp() * 1000)
            except Exception:
                pass
        return make_compaction_summary_message(entry.summary, entry.tokens_before or 0, ts)
    return None


def _get_message_for_compaction(entry: SessionEntry) -> AgentMessage | None:
    if entry.type == "compaction":
        return None
    return _get_message_from_entry(entry)


# ============================================================================
# Summarization
# ============================================================================

def _convert_to_llm_messages(agent_messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Convert AgentMessage objects to plain dicts for serialization."""
    result: list[dict[str, Any]] = []
    for msg in agent_messages:
        d: dict[str, Any] = {}
        role = getattr(msg, 'role', None)
        if role is None:
            continue
        d["role"] = role
        if role == "user":
            d["content"] = getattr(msg, 'content', "")
        elif role == "assistant":
            content = getattr(msg, 'content', []) or []
            d["content"] = []
            for block in content:
                block_type = getattr(block, 'type', None)
                if block_type == "text":
                    d["content"].append({"type": "text", "text": getattr(block, 'text', "")})
                elif block_type == "thinking":
                    d["content"].append({"type": "thinking", "thinking": getattr(block, 'thinking', "")})
                elif block_type == "toolCall":
                    d["content"].append({
                        "type": "toolCall",
                        "id": getattr(block, 'id', ""),
                        "name": getattr(block, 'name', ""),
                        "arguments": getattr(block, 'arguments', {}) or {},
                    })
        elif role == "toolResult":
            content = getattr(msg, 'content', []) or []
            d["content"] = []
            for c in content:
                if hasattr(c, 'type') and c.type == "text":
                    d["content"].append({"type": "text", "text": c.text if hasattr(c, 'text') else str(c)})
        elif role in ("compactionSummary", "branchSummary"):
            d["summary"] = getattr(msg, 'summary', "")
        result.append(d)
    return result


async def generate_summary(
    messages: list[AgentMessage],
    model: Model,
    provider: LlmProvider,
    api_key: str | None = None,
    reserve_tokens: int = 16384,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """
    Generate a structured summary of the given messages using the LLM.
    If previous_summary is provided, uses iterative update prompt.
    """
    max_tokens = min(
        math.floor(0.8 * reserve_tokens),
        model.max_tokens if model.max_tokens > 0 else float("inf"),
    )
    max_tokens = int(max_tokens)

    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    llm_messages = _convert_to_llm_messages(messages)
    conversation_text = serialize_conversation(llm_messages)

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    ts = int(time.time() * 1000)
    summary_user_msg = UserMessage(
        content=[TextContent(text=prompt_text)],
        timestamp=ts,
    )

    from .llm import LlmContext
    ctx = LlmContext(
        systemPrompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[summary_user_msg],
        tools=None,
    )

    response = await provider.complete(
        model, ctx, api_key=api_key, max_tokens=max_tokens,
    )

    if response.stop_reason == "error":
        raise RuntimeError(f"Summarization failed: {response.error_message or 'Unknown error'}")

    text = ""
    for block in response.content:
        if isinstance(block, TextBlock):
            text += block.text + "\n"
        elif hasattr(block, 'text'):
            text += block.text + "\n"

    return text.strip()


async def generate_turn_prefix_summary(
    messages: list[AgentMessage],
    model: Model,
    provider: LlmProvider,
    api_key: str | None = None,
    reserve_tokens: int = 16384,
) -> str:
    """Generate a summary for the prefix of a split turn."""
    max_tokens = min(
        math.floor(0.5 * reserve_tokens),  # Smaller budget for turn prefix
        model.max_tokens if model.max_tokens > 0 else float("inf"),
    )
    max_tokens = int(max_tokens)

    llm_messages = _convert_to_llm_messages(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"

    ts = int(time.time() * 1000)
    summary_user_msg = UserMessage(
        content=[TextContent(text=prompt_text)],
        timestamp=ts,
    )

    from .llm import LlmContext
    ctx = LlmContext(
        systemPrompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[summary_user_msg],
        tools=None,
    )

    response = await provider.complete(
        model, ctx, api_key=api_key, max_tokens=max_tokens,
    )

    if response.stop_reason == "error":
        raise RuntimeError(f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}")

    text = ""
    for block in response.content:
        if isinstance(block, TextBlock):
            text += block.text + "\n"
        elif hasattr(block, 'text'):
            text += block.text + "\n"

    return text.strip()


# ============================================================================
# Preparation
# ============================================================================

def prepare_compaction(
    entries: list[SessionEntry],
    settings: CompactionSettings | None = None,
) -> CompactionPreparation | None:
    """
    Prepare session entries for compaction.
    Returns None when compaction is not applicable (empty or last entry is already compaction).
    """
    if not entries:
        return None
    if entries[-1].type == "compaction":
        return None

    settings = settings or DEFAULT_COMPACTION_SETTINGS

    # Find previous compaction
    prev_compaction_index = -1
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].type == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        previous_summary = prev.summary
        first_kept_id = prev.first_kept_entry_id
        if first_kept_id:
            fidx = next((i for i, e in enumerate(entries) if e.id == first_kept_id), -1)
            boundary_start = fidx if fidx >= 0 else prev_compaction_index + 1
        else:
            boundary_start = prev_compaction_index + 1

    boundary_end = len(entries)

    # Estimate tokens
    ctx = build_session_context(entries)
    tokens_before = estimate_context_tokens([
        m.model_dump(by_alias=True, exclude_none=True, mode="json")
        for m in ctx.messages
    ])

    # Find cut point
    cut_point = find_cut_point(entries, boundary_start, boundary_end, settings.keep_recent_tokens)
    first_kept_entry = entries[cut_point.first_kept_entry_index]
    if not first_kept_entry.id:
        return None

    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index

    # Messages to summarize
    messages_to_summarize: list[AgentMessage] = []
    for i in range(boundary_start, history_end):
        msg = _get_message_for_compaction(entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    # Turn prefix messages (split turn)
    turn_prefix_messages: list[AgentMessage] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_for_compaction(entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    # File operations
    file_ops = _extract_file_operations(messages_to_summarize, entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            _extract_file_ops_from_message(msg, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry.id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


# ============================================================================
# Main compact function
# ============================================================================

async def compact(
    preparation: CompactionPreparation,
    model: Model,
    provider: LlmProvider,
    api_key: str | None = None,
    custom_instructions: str | None = None,
) -> CompactionResult:
    """Generate compaction summaries from prepared data."""
    (
        first_kept_entry_id, messages_to_summarize, turn_prefix_messages,
        is_split_turn, tokens_before, previous_summary, file_ops, settings,
    ) = (
        preparation.first_kept_entry_id, preparation.messages_to_summarize,
        preparation.turn_prefix_messages, preparation.is_split_turn,
        preparation.tokens_before, preparation.previous_summary,
        preparation.file_ops, preparation.settings,
    )

    if not first_kept_entry_id:
        raise ValueError("First kept entry has no ID")

    if is_split_turn and turn_prefix_messages:
        # Generate both summaries in parallel
        history_task = None
        if messages_to_summarize:
            history_task = asyncio.create_task(generate_summary(
                messages_to_summarize, model, provider, api_key,
                settings.reserve_tokens, previous_summary, custom_instructions,
            ))
        else:
            history_task = None

        turn_task = asyncio.create_task(generate_turn_prefix_summary(
            turn_prefix_messages, model, provider, api_key, settings.reserve_tokens,
        ))

        history_result = (await history_task) if history_task else "No prior history."
        turn_result = await turn_task

        summary = f"{history_result}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_result}"
    else:
        summary = await generate_summary(
            messages_to_summarize, model, provider, api_key,
            settings.reserve_tokens, previous_summary, custom_instructions,
        )

    # Append file operations metadata
    read_files, modified_files = _compute_file_lists(file_ops)
    summary += _format_file_operations(read_files, modified_files)

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=first_kept_entry_id,
        tokens_before=tokens_before,
        read_files=read_files,
        modified_files=modified_files,
    )
