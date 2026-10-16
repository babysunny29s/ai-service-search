"""Search history and chat messages, persisted in SQLite.

One history row = one candidate the user opened after a search; chat
messages hang off that row (like onesearch's SearchHistory/ChatMessage)."""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.core.models import utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_history (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    query TEXT NOT NULL,
    filters TEXT NOT NULL,
    candidate TEXT NOT NULL,
    search_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_history_user ON search_history (user_id, created_at);
CREATE TABLE IF NOT EXISTS chat_message (
    id TEXT PRIMARY KEY,
    history_id TEXT NOT NULL REFERENCES search_history (id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_message_history ON chat_message (history_id, created_at);
"""


class HistoryStore:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    async def _run(self, fn, *args):
        def locked():
            with self._lock:
                out = fn(*args)
                self._db.commit()
                return out

        return await asyncio.to_thread(locked)

    async def create(self, user_id: str | None, query: str, filters: dict[str, str], candidate: dict[str, Any], search_id: str | None) -> str:
        hid = f"hist_{secrets.token_hex(8)}"
        now = utcnow().isoformat()
        await self._run(
            self._db.execute,
            "INSERT INTO search_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (hid, user_id, query[:200], json.dumps(filters, ensure_ascii=False), json.dumps(candidate, ensure_ascii=False), search_id, now, now),
        )
        return hid

    async def get(self, history_id: str) -> dict[str, Any] | None:
        def q():
            row = self._db.execute("SELECT * FROM search_history WHERE id = ?", (history_id,)).fetchone()
            return _history(row) if row else None

        return await self._run(q)

    async def list(self, user_id: str | None, limit: int = 50) -> list[dict[str, Any]]:
        def q():
            rows = self._db.execute(
                "SELECT h.*, (SELECT COUNT(*) FROM chat_message m WHERE m.history_id = h.id) AS message_count "
                "FROM search_history h WHERE h.user_id IS ? ORDER BY h.created_at DESC LIMIT ?",
                (user_id, max(1, min(limit, 100))),
            ).fetchall()
            return [{**_history(r), "messageCount": r["message_count"]} for r in rows]

        return await self._run(q)

    async def update_candidate(self, history_id: str, candidate: dict[str, Any]) -> None:
        await self._run(
            self._db.execute,
            "UPDATE search_history SET candidate = ?, updated_at = ? WHERE id = ?",
            (json.dumps(candidate, ensure_ascii=False), utcnow().isoformat(), history_id),
        )

    async def delete(self, history_id: str) -> None:
        await self._run(self._db.execute, "DELETE FROM search_history WHERE id = ?", (history_id,))

    async def messages(self, history_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Oldest first; with `limit`, only the most recent `limit` messages."""

        def q():
            rows = self._db.execute(
                "SELECT * FROM chat_message WHERE history_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (history_id, limit if limit else -1),
            ).fetchall()
            return [_message(r) for r in reversed(rows)]

        return await self._run(q)

    async def append(self, history_id: str, messages: list[dict[str, Any]]) -> None:
        def q():
            now = utcnow().isoformat()
            self._db.executemany(
                "INSERT INTO chat_message VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (f"msg_{secrets.token_hex(8)}", history_id, m["role"], m["content"], json.dumps(m["citations"]) if m.get("citations") is not None else None, now)
                    for m in messages
                ],
            )
            self._db.execute("UPDATE search_history SET updated_at = ? WHERE id = ?", (now, history_id))

        await self._run(q)


def _history(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "userId": row["user_id"],
        "query": row["query"],
        "filters": json.loads(row["filters"]),
        "candidate": json.loads(row["candidate"]),
        "searchId": row["search_id"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "citations": json.loads(row["citations"]) if row["citations"] else None,
        "createdAt": row["created_at"],
    }
