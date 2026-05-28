"""
Session management: JSONL file storage + buildSessionContext.

Simplified from the TS session.ts + harness/session/session.ts:
  - Linear entries only (no tree structure, no parentId, no fork)
  - JSONL file backing
  - buildSessionContext handles compaction boundaries
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

from .agent_types import (
    AgentMessage,
    CompactionSummaryMessage,
    CustomMessage,
    make_compaction_summary_message,
    SessionContext,
    SessionEntry,
)
from .utils import uuid_v7


# ============================================================================
# JSONL Storage
# ============================================================================

class JsonlSessionStorage:
    """Read/write SessionEntry objects as JSONL lines."""

    def __init__(self, file_path: str | Path):
        self._path = Path(file_path)
        self._entries: list[SessionEntry] = []
        self._metadata: dict[str, Any] = {}

    @classmethod
    async def create(cls, file_path: str | Path, cwd: str, session_id: str) -> "JsonlSessionStorage":
        file_path = Path(file_path)
        storage = cls(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        storage._metadata = {
            "id": session_id,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "cwd": cwd,
            "path": str(file_path),
        }
        # Write header as first line
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps({"_metadata": storage._metadata}, ensure_ascii=False) + "\n")
        return storage

    @classmethod
    async def open(cls, file_path: str | Path) -> "JsonlSessionStorage":
        file_path = Path(file_path)
        storage = cls(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {file_path}")
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "_metadata" in data:
                    storage._metadata = data["_metadata"]
                else:
                    storage._entries.append(SessionEntry(**data))
        return storage

    async def append_entry(self, entry: SessionEntry) -> str:
        """Append one JSON line to the file, return the entry ID."""
        self._entries.append(entry)
        line = entry.model_dump(by_alias=True, exclude_none=True, mode="json")
        async with aiofiles.open(self._path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return entry.id

    @property
    def entries(self) -> list[SessionEntry]:
        return list(self._entries)

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def create_entry_id(self) -> str:
        return uuid_v7()


# ============================================================================
# Session
# ============================================================================

class Session:
    """Linear session backed by JSONL."""

    def __init__(self, storage: JsonlSessionStorage):
        self._storage = storage

    @staticmethod
    async def create(file_path: str | Path, cwd: str = ".", session_id: str | None = None) -> "Session":
        file_path = Path(file_path)
        sid = session_id or uuid_v7()
        storage = await JsonlSessionStorage.create(file_path, cwd, sid)
        return Session(storage)

    @staticmethod
    async def open(file_path: str | Path) -> "Session":
        file_path = Path(file_path)
        storage = await JsonlSessionStorage.open(file_path)
        return Session(storage)

    @property
    def entries(self) -> list[SessionEntry]:
        return self._storage.entries

    @property
    def metadata(self) -> dict[str, Any]:
        return self._storage.metadata

    async def append_message(self, message: AgentMessage) -> str:
        entry = SessionEntry(
            type="message",
            id=self._storage.create_entry_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=message,
        )
        return await self._storage.append_entry(entry)

    async def append_compaction(
        self, summary: str, first_kept_entry_id: str, tokens_before: int,
        read_files: list[str] | None = None,
        modified_files: list[str] | None = None,
    ) -> str:
        entry = SessionEntry(
            type="compaction",
            id=self._storage.create_entry_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary,
            firstKeptEntryId=first_kept_entry_id,
            tokensBefore=tokens_before,
            readFiles=read_files or [],
            modifiedFiles=modified_files or [],
        )
        return await self._storage.append_entry(entry)

    async def append_thinking_level_change(self, level: str) -> str:
        entry = SessionEntry(
            type="thinking_level_change",
            id=self._storage.create_entry_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            thinkingLevel=level,
        )
        return await self._storage.append_entry(entry)

    async def append_model_change(self, provider: str, model_id: str) -> str:
        entry = SessionEntry(
            type="model_change",
            id=self._storage.create_entry_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            modelId=model_id,
        )
        return await self._storage.append_entry(entry)

    # ========================================================================
    # Context building (handles compaction boundaries)
    # ========================================================================

    async def build_context(self) -> SessionContext:
        return build_session_context(self.entries)


# ============================================================================
# buildSessionContext — mirrors TS session.ts:21-76
# ============================================================================

def build_session_context(entries: list[SessionEntry]) -> SessionContext:
    """
    Walk entries linearly and build the model-facing message list.
    When a compaction entry exists, replaces older history with
    the compaction summary, keeping messages from firstKeptEntryId onward.
    """
    thinking_level = "off"
    model_provider: str | None = None
    model_id: str | None = None
    compaction: SessionEntry | None = None

    for entry in entries:
        if entry.type == "thinking_level_change":
            thinking_level = entry.thinking_level or "off"
        elif entry.type == "model_change":
            model_provider = entry.provider
            model_id = entry.model_id
        elif entry.type == "message" and entry.message is not None:
            msg = entry.message
            if hasattr(msg, 'role') and msg.role == "assistant":
                model_provider = getattr(msg, 'provider', None) or model_provider
                model_id = getattr(msg, 'model', None) or model_id
        elif entry.type == "compaction":
            compaction = entry

    messages: list[AgentMessage] = []

    def _append_message(e: SessionEntry) -> None:
        if e.type == "message" and e.message is not None:
            messages.append(e.message)
        elif e.type == "custom_message" and e.content is not None:
            messages.append(CustomMessage(
                customType=e.custom_type or "custom",
                content=e.content,
                display=e.display if e.display is not None else True,
                details=e.details,
                timestamp=int(datetime.fromisoformat(e.timestamp).timestamp() * 1000) if e.timestamp else 0,
            ))

    if compaction:
        # 1. Insert compaction summary
        ts = int(datetime.fromisoformat(compaction.timestamp).timestamp() * 1000) if compaction.timestamp else 0
        messages.append(make_compaction_summary_message(
            summary=compaction.summary or "",
            tokens_before=compaction.tokens_before or 0,
            timestamp=ts,
        ))

        # 2. Find compaction index
        comp_idx = next((i for i, e in enumerate(entries) if e.id == compaction.id), -1)

        # 3. Walk entries before compaction, only include those >= firstKeptEntryId
        first_kept_id = compaction.first_kept_entry_id
        found_first_kept = False
        if comp_idx > 0 and first_kept_id:
            for i in range(comp_idx):
                e = entries[i]
                if e.id == first_kept_id:
                    found_first_kept = True
                if found_first_kept:
                    _append_message(e)

        # 4. Walk entries after compaction
        if comp_idx >= 0:
            for i in range(comp_idx + 1, len(entries)):
                _append_message(entries[i])
    else:
        for e in entries:
            _append_message(e)

    return SessionContext(
        messages=messages,
        thinkingLevel=thinking_level,
        modelProvider=model_provider or "",
        modelId=model_id or "",
    )
