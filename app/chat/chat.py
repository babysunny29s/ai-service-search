"""Chat about one saved candidate (a search-history row).

The model gets a `web_search` tool: it answers right away from the saved
profile, the conversation or its own knowledge when it can, and calls the
tool only when it does not know. Searches run (in parallel) and a second,
tool-free call answers from the results. So a question costs 1 LLM call when
the model knows, or 2 LLM calls + 1..CHAT_MAX_SEARCHES searches when it does not.

The prompt pins the person's identity (name, headline, key facts, known
sources) so the model does not drift to namesakes, and carries the last few
turns so follow-ups ("where did he study?") work."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.core import tracing
from app.config import Settings
from app.chat.history import HistoryStore
from app.core.llm import LLMProvider, ToolCall
from app.core.models import ToolCallLog
from app.infra.search_locale import locale_for
from app.infra.fetcher import excerpt
from app.core.urls import fold, social_platform

log = logging.getLogger(__name__)

_BASE_RULES = """You answer questions about ONE specific person, described in the PERSON block.
Stay on this person: namesakes are different people. Never invent facts, dates or numbers.
Answer like a well-informed journalist: open with the direct answer, then add the relevant context in 1-3 sentences.
Answer the intent, not only the literal words: asked about a husband, say whether the person is married and what is publicly known about their partner or relationship; asked about age, email, or a phone number, give the value the sources state, and for age give the birth date too.
When only part is known, state what is known and say plainly what has not been made public.
Answer in the language of the user's latest question (e.g. Vietnamese question -> Vietnamese answer), in plain text (no markdown, no [1] markers), at most 150 words.
Keep names, titles and quoted source text as they are."""

SYSTEM_AGENT = _BASE_RULES + """
First decide whether you already know the answer: from the PERSON block, the earlier conversation or your own knowledge of THIS person.
If you know it, answer directly. Do not search to double-check or to add detail to something you know.
Call web_search only when you do not know the answer, or cannot tell whether what you know is about this person rather than a namesake.
Search queries must contain the person's full name plus a detail that tells them apart from namesakes (employer, country, field), written in the language the sources are likely to use."""

SYSTEM_GROUNDED = _BASE_RULES + """
Use the SEARCH RESULTS (snippets and page passages), the PERSON block and the earlier conversation; you may add your own reliable knowledge of this person when it is consistent with them. Ignore results about other people.
Say that nothing relevant was found only when the results contain nothing related to the question."""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web (Google) for information about the person. Use only when you cannot answer reliably without it.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Google query with the person's name and a distinguishing detail"}},
            "required": ["query"],
        },
    },
}

PROFILE_CHARS = 600
SNIPPET_CHARS = 300


class ChatError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def person_block(candidate: dict[str, Any]) -> str:
    lines = [f"Name: {candidate.get('displayName', '')}"]
    if candidate.get("aliases"):
        lines.append("Also known as: " + ", ".join(candidate["aliases"][:4]))
    for key in ("headline", "occupation", "knownFor", "nationality", "location"):
        if candidate.get(key):
            lines.append(f"{key}: {_clip(candidate[key], 160)}")
    for f in (candidate.get("keyFacts") or [])[:8]:
        lines.append(f"{f.get('label') or f.get('key')}: {_clip(f.get('value', ''), 160)}")
    urls = [s["url"] for s in (candidate.get("sources") or []) if s.get("url")][:3]
    if urls:
        lines.append("Known profiles: " + ", ".join(urls))
    profile = " ".join(candidate.get("profile") or [])
    if profile:
        lines.append("Bio: " + _clip(profile, PROFILE_CHARS))
    return "\n".join(lines)


def search_query(call: ToolCall, name: str, question: str) -> str:
    """The model's query, made to name the person; the question when the arguments are unusable."""
    try:
        query = str(json.loads(call.arguments or "{}").get("query") or "")
    except (json.JSONDecodeError, AttributeError):
        query = ""
    query = " ".join(query.split()) or question
    if fold(name) not in fold(query):
        query = f"{name} {query}"
    return _clip(query, 200)


class ChatService:
    def __init__(self, settings: Settings, store: HistoryStore, search: Any, llm: LLMProvider, fetcher: Any = None):
        self.s = settings
        self.store = store
        self.search = search
        self.llm = llm
        self.fetcher = fetcher

    async def load(self, history_id: str, user_id: str | None) -> dict[str, Any]:
        rec = await self.store.get(history_id)
        if rec is None:
            raise ChatError(404, "history not found")
        if rec["userId"] and rec["userId"] != user_id:
            raise ChatError(403, "history belongs to another user")
        return rec

    async def _search(self, rec: dict[str, Any], queries: list[str], logs: list[ToolCallLog]) -> list[dict[str, str]]:
        name = rec["candidate"].get("displayName", "")
        locale = locale_for(rec["filters"], (self.s.search_locale_gl, self.s.search_locale_hl), name)
        outs = await asyncio.gather(*(self.search.search(q, "web", logs, locale) for q in queries), return_exceptions=True)
        seen: set[str] = set()
        evidence: list[dict[str, str]] = []
        for out in outs:
            if isinstance(out, BaseException):
                log.warning("chat search failed: %s", out)
                continue
            kept = 0
            for h in out[0]:
                if kept >= self.s.chat_search_results:
                    break
                if h.url and h.url not in seen:
                    seen.add(h.url)
                    kept += 1
                    evidence.append({"url": h.url, "title": _clip(h.title, 120), "snippet": _clip(h.snippet, SNIPPET_CHARS)})
        await self._read_pages(evidence, name)
        return evidence

    async def _read_pages(self, evidence: list[dict[str, str]], name: str) -> None:
        """Add passages around the name from the first readable result pages
        (social networks are skipped: they do not serve their content to bots)."""
        if not self.fetcher or self.s.chat_fetch_pages <= 0:
            return
        targets = [e for e in evidence if not social_platform(e["url"])][: self.s.chat_fetch_pages]
        if not targets:
            return
        try:
            pages = await asyncio.wait_for(
                asyncio.gather(*(self.fetcher.fetch(e["url"]) for e in targets), return_exceptions=True),
                timeout=self.s.chat_fetch_timeout_s,
            )
        except asyncio.TimeoutError:
            log.info("chat page fetch timed out")
            return
        for e, page in zip(targets, pages):
            if not isinstance(page, BaseException) and page.status == "fetched" and page.text:
                e["passage"] = excerpt(page.text, name, self.s.chat_page_chars, window=300)

    def _messages(self, system: str, rec: dict[str, Any], prior: list[dict[str, Any]], question: str) -> list[dict[str, str]]:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": "PERSON:\n" + person_block(rec["candidate"])},
            {"role": "assistant", "content": "Understood. I will answer only about this person."},
        ]
        msgs += [{"role": m["role"], "content": _clip(m["content"], self.s.chat_turn_chars)} for m in prior]
        msgs.append({"role": "user", "content": question})
        return msgs

    async def stream(self, history_id: str, question: str, user_id: str | None) -> AsyncIterator[dict[str, Any]]:
        """Events: status → delta* → done, or error. The turn is saved only when complete."""
        try:
            rec = await self.load(history_id, user_id)
        except ChatError as e:
            yield {"event": "error", "status": e.status, "message": str(e)}
            return
        async for ev in self._run_turn(rec, history_id, question, self.store):
            yield ev

    async def stream_with(self, rec: dict[str, Any], thread_id: str, question: str, store: Any) -> AsyncIterator[dict[str, Any]]:
        async for ev in self._run_turn(rec, thread_id, question, store):
            yield ev

    async def ask_with(self, rec: dict[str, Any], thread_id: str, question: str, store: Any | None = None) -> dict[str, Any]:
        async for ev in self._run_turn(rec, thread_id, question, store or self.store):
            if ev["event"] == "error":
                raise ChatError(ev["status"], ev["message"])
            if ev["event"] == "done":
                return {k: v for k, v in ev.items() if k != "event"}
        raise ChatError(502, "no answer")

    async def _run_turn(self, rec: dict[str, Any], thread_id: str, question: str, store: Any) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        prior = await store.messages(thread_id, limit=self.s.chat_max_turns * 2)
        session_id = rec.get("sessionId") or thread_id
        candidate_id = rec.get("candidateId") or rec.get("id")
        trace = {"langfuse_session_id": session_id, "sessionId": session_id, "candidateId": candidate_id, "searchId": rec.get("searchId")}
        can_search = self.s.chat_max_searches > 0 and self.search is not None
        parts: list[str] = []
        calls: list[ToolCall] = []
        logs: list[ToolCallLog] = []
        evidence: list[dict[str, str]] = []
        queries: list[str] = []
        llm_calls = 1
        yield {"event": "status", "stage": "thinking"}
        obs, stats_tok = tracing.begin("chat", input={"question": question, "sessionId": session_id, "candidateId": candidate_id}, session_id=session_id, tags=["chat"])
        try:
            async for out in self.llm.chat_stream(
                self._messages(SYSTEM_AGENT, rec, prior, question),
                tools=[WEB_SEARCH_TOOL] if can_search else None,
                max_tokens=self.s.chat_max_tokens,
                trace_name="llm:chat",
                trace_metadata=trace,
            ):
                if isinstance(out, ToolCall):
                    calls.append(out)
                else:
                    parts.append(out)
                    yield {"event": "delta", "text": out}

            searches = [c for c in calls if c.name == WEB_SEARCH_TOOL["function"]["name"]][: self.s.chat_max_searches]
            if searches:
                name = rec["candidate"].get("displayName", "")
                queries = list(dict.fromkeys(search_query(c, name, question) for c in searches))
                yield {"event": "status", "stage": "searching", "queries": queries}
                evidence = await self._search(rec, queries, logs)
                results = "\n".join(
                    f"[{i}] {e['title']} ({e['url']})\n{e['snippet']}" + (f"\nPage: {e['passage']}" if e.get("passage") else "")
                    for i, e in enumerate(evidence, 1)
                )
                msgs = self._messages(SYSTEM_GROUNDED, rec, prior, question)
                msgs[-1]["content"] = f"SEARCH RESULTS:\n{results or '(nothing found)'}\n\nQUESTION: {question}"
                if parts:
                    yield {"event": "reset"}
                    parts = []
                yield {"event": "status", "stage": "answering"}
                llm_calls += 1
                async for out in self.llm.chat_stream(msgs, max_tokens=self.s.chat_max_tokens, trace_name="llm:chat-grounded", trace_metadata=trace):
                    if isinstance(out, str):
                        parts.append(out)
                        yield {"event": "delta", "text": out}
            answer = "".join(parts).strip()
            if not answer:
                yield {"event": "error", "status": 502, "message": "the model returned an empty answer"}
                return
            citations = [{"url": e["url"], "title": e["title"]} for e in evidence]
            await store.append(
                thread_id,
                [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "citations": citations}],
            )
            st = tracing.current_stats()
            yield {
                "event": "done",
                "sessionId": session_id,
                "candidateId": candidate_id or thread_id,
                "searchId": rec.get("searchId"),
                "answer": answer,
                "answeredFrom": "web" if queries else "knowledge",
                "citations": citations,
                "meta": {
                    "searchQueries": queries,
                    "searchCalls": sum(1 for c in logs if not c.cached),
                    "llmCalls": llm_calls,
                    "latencyMs": int((time.perf_counter() - started) * 1000),
                    **tracing.client_cost(st),
                },
            }
        except Exception as e:
            log.warning("chat llm failed: %s", e)
            yield {"event": "error", "status": 502, "message": f"LLM error: {type(e).__name__}"}
        finally:
            tracing.finish(obs, stats_tok)

    async def ask(self, history_id: str, question: str, user_id: str | None) -> dict[str, Any]:
        async for ev in self.stream(history_id, question, user_id):
            if ev["event"] == "error":
                raise ChatError(ev["status"], ev["message"])
            if ev["event"] == "done":
                return {k: v for k, v in ev.items() if k != "event"}
        raise ChatError(502, "no answer")
