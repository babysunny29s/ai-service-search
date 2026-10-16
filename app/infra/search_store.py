"""Persist into the existing onesearch tables. No DDL.

SearchHistory.id is the client searchId. ChatSession.id is the client sessionId.
Candidate.id is assigned here (cnd_…). ChatHistory.id is assigned here (msg_…).
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

from app.core.models import ClientCandidate, ClientMeta, ClientSource, DeepSearchResponse, PeopleSearchResponse, utcnow
from app.core.urls import fold

log = logging.getLogger(__name__)

_ID_ATTEMPTS = 8


class ChatSessionConflict(Exception):
    """session_id is already bound to a different candidate_id."""


def _fresh_candidate_id(taken: set[str]) -> str:
    """64-bit id, skipped when it is already in `taken` (this batch or the table)."""
    for _ in range(_ID_ATTEMPTS):
        cid = f"cnd_{secrets.token_hex(8)}"
        if cid not in taken:
            taken.add(cid)
            return cid
    raise RuntimeError("could not allocate a unique candidate id")


async def _insert_candidate(conn: Any, search_id: str, person_key: str, data: dict[str, Any], reserved: set[str]) -> str:
    """Insert a new row. An id already in Candidate, or lost to a concurrent insert, is discarded."""
    for _ in range(_ID_ATTEMPTS):
        cid = _fresh_candidate_id(reserved)
        if await conn.fetchval('SELECT 1 FROM "Candidate" WHERE id = $1', cid):
            continue
        data["id"] = cid
        inserted = await conn.fetchval(
            """
            INSERT INTO "Candidate" (id, "searchId", "personKey", "candidateInformation", "createdAt", "updatedAt")
            VALUES ($1, $2, $3, $4::jsonb, NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            cid,
            search_id,
            person_key,
            json.dumps(data, ensure_ascii=False),
        )
        if inserted:
            return inserted
    raise RuntimeError("could not allocate a unique candidate id")


def _now() -> str:
    return utcnow().isoformat()


def _chat_rec(session_id: str, candidate_id: str, rec: dict[str, Any], created: str, updated: str) -> dict[str, Any]:
    return {
        "id": session_id,
        "sessionId": session_id,
        "candidateId": candidate_id,
        "userId": rec.get("userId"),
        "query": rec.get("query", ""),
        "filters": rec.get("filters") or {},
        "candidate": rec["candidate"],
        "searchId": rec.get("searchId"),
        "createdAt": created,
        "updatedAt": updated,
    }


def _session(candidate_id: str, search_id: str, query: str, filters: dict, user_id: str | None, metadata: dict, created: str, updated: str) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "userId": user_id,
        "query": query,
        "filters": filters or {},
        "candidate": metadata,
        "searchId": search_id,
        "createdAt": created,
        "updatedAt": updated,
    }


class MemorySearchStore:
    """Used in tests and when Postgres is not configured."""

    def __init__(self) -> None:
        self.searches: dict[str, dict[str, Any]] = {}
        self.candidates: dict[str, dict[str, Any]] = {}
        self._by_person: dict[tuple[str, str], str] = {}
        self.chat_sessions: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}

    async def close(self) -> None:
        return None

    async def ensure_search(self, search_id: str, query: str, filters: dict[str, str], mode: str = "searchprofile", user_id: str | None = None) -> None:
        now = _now()
        row = self.searches.get(search_id)
        if row is None:
            self.searches[search_id] = {
                "search_id": search_id,
                "query": query,
                "filters": dict(filters or {}),
                "user_id": user_id,
                "mode": mode,
                "created_at": now,
                "updated_at": now,
            }
            return
        row["query"] = query or row["query"]
        row["filters"] = dict(filters or row["filters"])
        if user_id is not None:
            row["user_id"] = user_id
        row["mode"] = mode or row["mode"]
        row["updated_at"] = now

    async def save_candidates(self, search_id: str, candidates: list[ClientCandidate]) -> list[ClientCandidate]:
        out: list[ClientCandidate] = []
        now = _now()
        reserved = set(self.candidates)
        for c in candidates:
            person_key = c.id
            cid = self._by_person.get((search_id, person_key)) or _fresh_candidate_id(reserved)
            data = json.loads(c.model_dump_json())
            data["id"] = cid
            self.candidates[cid] = {
                "candidate_id": cid,
                "search_id": search_id,
                "person_key": person_key,
                "candidate_information": data,
                "created_at": self.candidates.get(cid, {}).get("created_at", now),
                "updated_at": now,
            }
            self._by_person[(search_id, person_key)] = cid
            out.append(c.model_copy(update={"id": cid}))
        if search_id in self.searches:
            self.searches[search_id]["updated_at"] = now
        return out

    async def update_candidate(self, candidate_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        row = self.candidates.get(candidate_id)
        if not row:
            raise KeyError(candidate_id)
        data = dict(metadata)
        data["id"] = candidate_id
        row["candidate_information"] = data
        row["updated_at"] = _now()
        return data

    async def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = self.candidates.get(candidate_id)
        if not row:
            return None
        search = self.searches.get(row["search_id"], {})
        return _session(
            candidate_id,
            row["search_id"],
            search.get("query", ""),
            search.get("filters", {}),
            search.get("user_id"),
            row.get("candidate_information") or row.get("metadata"),
            row["created_at"],
            row["updated_at"],
        )

    async def session(self, candidate_id: str, user_id: str | None = None) -> dict[str, Any]:
        rec = await self.get_candidate(candidate_id)
        if rec is None:
            raise KeyError(candidate_id)
        if rec["userId"] and user_id is not None and rec["userId"] != user_id:
            raise PermissionError("candidate belongs to another user")
        return rec

    async def ensure_chat_session(self, session_id: str, candidate_id: str, user_id: str | None = None) -> dict[str, Any]:
        rec = await self.session(candidate_id, user_id)
        now = _now()
        row = self.chat_sessions.get(session_id)
        if row is None:
            self.chat_sessions[session_id] = {
                "session_id": session_id,
                "candidate_id": candidate_id,
                "candidate": rec["candidate"],
                "user_id": user_id or rec["userId"],
                "search_id": rec["searchId"],
                "query": rec["query"],
                "filters": rec["filters"],
                "created_at": now,
                "updated_at": now,
            }
        else:
            if row["candidate_id"] != candidate_id:
                raise ChatSessionConflict(session_id)
            row["candidate"] = rec["candidate"]
            if user_id is not None:
                row["user_id"] = user_id
            row["search_id"] = rec["searchId"]
            row["query"] = rec["query"]
            row["filters"] = rec["filters"]
            row["updated_at"] = now
        row = self.chat_sessions[session_id]
        rec = {**rec, "userId": row["user_id"], "candidate": row["candidate"]}
        return _chat_rec(session_id, candidate_id, rec, row["created_at"], row["updated_at"])

    async def messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self._messages.get(session_id, [])
        if limit:
            rows = rows[-limit:]
        return list(rows)

    async def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        now = _now()
        bucket = self._messages.setdefault(session_id, [])
        for m in messages:
            bucket.append(
                {
                    "id": f"msg_{secrets.token_hex(8)}",
                    "role": m["role"],
                    "content": m["content"],
                    "citations": m.get("citations"),
                    "createdAt": now,
                }
            )
        if session_id in self.chat_sessions:
            self.chat_sessions[session_id]["updated_at"] = now

    async def person_key(self, candidate_id: str) -> str | None:
        row = self.candidates.get(candidate_id)
        return row["person_key"] if row else None

    async def recent_deep(self, person_key: str, max_age_s: float) -> dict[str, Any] | None:
        since = _since(max_age_s)
        rows = [
            r
            for r in self.candidates.values()
            if r["person_key"] == person_key and "socialSources" in (r.get("candidate_information") or {}) and r["updated_at"] > since
        ]
        return max(rows, key=lambda r: r["updated_at"])["candidate_information"] if rows else None

    async def recent_search(self, query: str, filters: dict[str, str], mode: str, max_age_s: float) -> list[tuple[str, dict[str, Any]]]:
        since, key = _since(max_age_s), search_key(query, filters)
        for row in sorted(self.searches.values(), key=lambda r: r["updated_at"], reverse=True):
            if row["updated_at"] <= since or row["mode"] != mode or search_key(row["query"], row["filters"]) != key:
                continue
            found = sorted((c for c in self.candidates.values() if c["search_id"] == row["search_id"]), key=lambda c: c["created_at"])
            if found:
                return [(c["person_key"], c["candidate_information"]) for c in found]
        return []


class PostgresSearchStore:
    def __init__(self, pool: Any) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, settings: Any) -> PostgresSearchStore:
        import asyncpg

        dsn = (
            f"postgresql://{quote_plus(settings.db_user)}:{quote_plus(settings.db_password)}"
            f"@{settings.db_host}:{int(settings.db_port)}/{settings.db_name}"
        )
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        log.info("Postgres search store ready db=%s host=%s", settings.db_name, settings.db_host)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def ensure_search(self, search_id: str, query: str, filters: dict[str, str], mode: str = "searchprofile", user_id: str | None = None) -> None:
        q = """
        INSERT INTO "SearchHistory" (id, "userId", query, filters, result, mode, "createdAt", "updatedAt")
        VALUES ($1, $2, $3, $4::jsonb, '{"candidates":[]}'::jsonb, $5, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE SET
            query = COALESCE(NULLIF(EXCLUDED.query, ''), "SearchHistory".query),
            filters = EXCLUDED.filters,
            "userId" = COALESCE(EXCLUDED."userId", "SearchHistory"."userId"),
            mode = EXCLUDED.mode,
            "updatedAt" = NOW()
        """
        await self.pool.execute(q, search_id, user_id, query or "", json.dumps(filters or {}), mode)

    async def save_candidates(self, search_id: str, candidates: list[ClientCandidate]) -> list[ClientCandidate]:
        out: list[ClientCandidate] = []
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                reserved: set[str] = set()
                for c in candidates:
                    person_key = c.id
                    existing = await conn.fetchval(
                        'SELECT id FROM "Candidate" WHERE "searchId" = $1 AND "personKey" = $2',
                        search_id,
                        person_key,
                    )
                    data = json.loads(c.model_dump_json())
                    if existing:
                        cid = existing
                        data["id"] = cid
                        await conn.execute(
                            """
                            UPDATE "Candidate"
                            SET "candidateInformation" = $2::jsonb, "updatedAt" = NOW()
                            WHERE id = $1
                            """,
                            cid,
                            json.dumps(data, ensure_ascii=False),
                        )
                    else:
                        cid = await _insert_candidate(conn, search_id, person_key, data, reserved)
                    out.append(c.model_copy(update={"id": cid}))
                rows = await conn.fetch(
                    'SELECT "candidateInformation" FROM "Candidate" WHERE "searchId" = $1 ORDER BY "createdAt", id',
                    search_id,
                )
                packed = [_json(r["candidateInformation"]) for r in rows]
                await conn.execute(
                    """
                    UPDATE "SearchHistory"
                    SET result = $2::jsonb, "updatedAt" = NOW()
                    WHERE id = $1
                    """,
                    search_id,
                    json.dumps({"candidates": packed}, ensure_ascii=False),
                )
        return out

    async def update_candidate(self, candidate_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        data = dict(metadata)
        data["id"] = candidate_id
        result = await self.pool.execute(
            """
            UPDATE "Candidate"
            SET "candidateInformation" = $2::jsonb, "updatedAt" = NOW()
            WHERE id = $1
            """,
            candidate_id,
            json.dumps(data, ensure_ascii=False),
        )
        if result == "UPDATE 0":
            raise KeyError(candidate_id)
        await self.pool.execute(
            """
            UPDATE "SearchHistory"
            SET "selectedCandidateId" = $1, "updatedAt" = NOW()
            WHERE id = (SELECT "searchId" FROM "Candidate" WHERE id = $1)
            """,
            candidate_id,
        )
        return data

    async def person_key(self, candidate_id: str) -> str | None:
        return await self.pool.fetchval('SELECT "personKey" FROM "Candidate" WHERE id = $1', candidate_id)

    async def recent_deep(self, person_key: str, max_age_s: float) -> dict[str, Any] | None:
        info = await self.pool.fetchval(
            """
            SELECT "candidateInformation" FROM "Candidate"
            WHERE "personKey" = $1 AND "candidateInformation" ? 'socialSources'
              AND "updatedAt" > NOW() - make_interval(secs => $2)
            ORDER BY "updatedAt" DESC
            LIMIT 1
            """,
            person_key,
            float(max_age_s),
        )
        return _json(info) if info else None

    async def recent_search(self, query: str, filters: dict[str, str], mode: str, max_age_s: float) -> list[tuple[str, dict[str, Any]]]:
        """Candidates of the newest search with this query + filters (compared folded:
        case and diacritics ignored), any user."""
        key = search_key(query, filters)
        rows = await self.pool.fetch(
            """
            SELECT id, query, filters FROM "SearchHistory"
            WHERE mode = $1 AND "updatedAt" > NOW() - make_interval(secs => $2)
              AND jsonb_array_length(COALESCE(result->'candidates', '[]'::jsonb)) > 0
            ORDER BY "updatedAt" DESC
            LIMIT 500
            """,
            mode,
            float(max_age_s),
        )
        for r in rows:
            if search_key(r["query"], _json(r["filters"]) or {}) != key:
                continue
            found = await self.pool.fetch(
                'SELECT "personKey", "candidateInformation" FROM "Candidate" WHERE "searchId" = $1 ORDER BY "createdAt", id',
                r["id"],
            )
            if found:
                return [(c["personKey"], _json(c["candidateInformation"])) for c in found]
        return []

    async def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = await self.pool.fetchrow(
            """
            SELECT c.id, c."searchId", c."candidateInformation", c."createdAt", c."updatedAt",
                   s.query, s.filters, s."userId"
            FROM "Candidate" c
            JOIN "SearchHistory" s ON s.id = c."searchId"
            WHERE c.id = $1
            """,
            candidate_id,
        )
        if not row:
            return None
        return _session(
            row["id"],
            row["searchId"],
            row["query"],
            _json(row["filters"]) or {},
            row["userId"],
            _json(row["candidateInformation"]),
            _ts(row["createdAt"]),
            _ts(row["updatedAt"]),
        )

    async def session(self, candidate_id: str, user_id: str | None = None) -> dict[str, Any]:
        rec = await self.get_candidate(candidate_id)
        if rec is None:
            raise KeyError(candidate_id)
        if rec["userId"] and user_id is not None and rec["userId"] != user_id:
            raise PermissionError("candidate belongs to another user")
        return rec

    async def ensure_chat_session(self, session_id: str, candidate_id: str, user_id: str | None = None) -> dict[str, Any]:
        rec = await self.session(candidate_id, user_id)
        existing = await self.pool.fetchrow('SELECT id, "candidateId" FROM "ChatSession" WHERE id = $1', session_id)
        snap = json.dumps(rec["candidate"], ensure_ascii=False)
        uid = user_id if user_id is not None else rec["userId"]
        if existing is None:
            await self.pool.execute(
                """
                INSERT INTO "ChatSession" (id, "candidateId", "userId", "searchId", metadata, "createdAt", "updatedAt")
                VALUES ($1, $2, $3, $4, $5::jsonb, NOW(), NOW())
                """,
                session_id,
                candidate_id,
                uid,
                rec["searchId"],
                snap,
            )
        else:
            if existing["candidateId"] != candidate_id:
                raise ChatSessionConflict(session_id)
            await self.pool.execute(
                """
                UPDATE "ChatSession" SET
                    metadata = $2::jsonb,
                    "userId" = COALESCE($3, "userId"),
                    "searchId" = $4,
                    "updatedAt" = NOW()
                WHERE id = $1
                """,
                session_id,
                snap,
                uid,
                rec["searchId"],
            )
        row = await self.pool.fetchrow(
            """
            SELECT cs.metadata, cs."userId", cs."searchId", cs."createdAt", cs."updatedAt",
                   s.query, s.filters
            FROM "ChatSession" cs
            JOIN "SearchHistory" s ON s.id = cs."searchId"
            WHERE cs.id = $1
            """,
            session_id,
        )
        packed = {
            "userId": row["userId"],
            "query": row["query"],
            "filters": _json(row["filters"]) or {},
            "candidate": _json(row["metadata"]),
            "searchId": row["searchId"],
        }
        return _chat_rec(session_id, candidate_id, packed, _ts(row["createdAt"]), _ts(row["updatedAt"]))

    async def messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT id, role, content, citations, "createdAt"
            FROM "ChatHistory" WHERE "sessionId" = $1
            ORDER BY "createdAt" DESC, id DESC
        """
        if limit:
            rows = await self.pool.fetch(sql + " LIMIT $2", session_id, limit)
        else:
            rows = await self.pool.fetch(sql, session_id)
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "citations": _json(r["citations"]),
                "createdAt": _ts(r["createdAt"]),
            }
            for r in reversed(rows)
        ]

    async def append(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for m in messages:
                    cites = m.get("citations")
                    await conn.execute(
                        """
                        INSERT INTO "ChatHistory" (id, "sessionId", role, content, citations, "createdAt")
                        VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
                        """,
                        f"msg_{secrets.token_hex(8)}",
                        session_id,
                        m["role"],
                        m["content"],
                        json.dumps(cites) if cites is not None else None,
                    )
                await conn.execute('UPDATE "ChatSession" SET "updatedAt" = NOW() WHERE id = $1', session_id)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _ts(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value) if value else ""


def search_key(query: str, filters: dict[str, str] | None) -> tuple:
    return fold(" ".join((query or "").split())), tuple(sorted((k, fold(str(v))) for k, v in (filters or {}).items() if str(v).strip()))


def _since(max_age_s: float) -> str:
    return (utcnow() - timedelta(seconds=max_age_s)).isoformat()


async def reuse_search(
    store: MemorySearchStore | PostgresSearchStore, query: str, filters: dict[str, str], mode: str, max_age_h: float
) -> PeopleSearchResponse | None:
    """The newest saved search (any user) with the same query + filters, as a free cache hit.
    The candidates keep their personKey as id, so saving them gives this search its own rows."""
    if max_age_h <= 0:
        return None
    try:
        rows = await store.recent_search(query, filters, mode, max_age_h * 3600)
    except Exception:
        log.exception("reuse lookup for %r failed", query)
        return None
    candidates = [ClientCandidate.model_validate({**info, "id": pk}) for pk, info in rows]
    if not candidates:
        return None
    return PeopleSearchResponse(
        mode=mode,  # type: ignore[arg-type]
        status="resolved" if len(candidates) == 1 else "ambiguous",
        candidates=candidates,
        meta=ClientMeta(searchCalls=0, cacheHit=True, totalSources=sum(len(c.sources) for c in candidates)),
    )


async def reuse_deep(store: MemorySearchStore | PostgresSearchStore, candidate: ClientCandidate, max_age_h: float) -> DeepSearchResponse | None:
    """The newest deep search of the same person (Candidate.personKey, any user), as a free cache hit."""
    if max_age_h <= 0:
        return None
    try:
        person = await store.person_key(candidate.id) or candidate.id
        info = await store.recent_deep(person, max_age_h * 3600)
    except Exception:
        log.exception("reuse lookup for %s failed", candidate.id)
        return None
    if not info:
        return None
    social = [ClientSource.model_validate(s) for s in info.get("socialSources") or []]
    articles = [ClientSource.model_validate(s) for s in info.get("articleSources") or []]
    return DeepSearchResponse(
        candidate=ClientCandidate.model_validate({**info, "id": candidate.id}),
        socialSources=social,
        articleSources=articles,
        warnings=list(info.get("warnings") or []),
        meta=ClientMeta(searchCalls=0, cacheHit=True, totalSources=len(social) + len(articles)),
    )


def deep_metadata(out: Any) -> dict[str, Any]:
    meta = json.loads(out.candidate.model_dump_json())
    meta["socialSources"] = [json.loads(s.model_dump_json()) for s in out.socialSources]
    meta["articleSources"] = [json.loads(s.model_dump_json()) for s in out.articleSources]
    meta["warnings"] = list(out.warnings or [])
    return meta


async def persist_people(
    store: MemorySearchStore | PostgresSearchStore,
    *,
    search_id: str,
    query: str,
    filters: dict[str, str],
    mode: str,
    candidates: list[ClientCandidate],
    user_id: str | None = None,
) -> list[ClientCandidate]:
    uid = str(user_id).strip()[:128] if user_id not in (None, "") else None
    if uid is None and isinstance(store, PostgresSearchStore):
        # "SearchHistory"."userId" is NOT NULL and references "User": anonymous searches are not saved.
        return list(candidates)
    await store.ensure_search(search_id, query, filters, mode=mode, user_id=uid)
    return await store.save_candidates(search_id, candidates)
