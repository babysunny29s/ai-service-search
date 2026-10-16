"""Single search steps for one known person, exposed as building blocks so a
client (onesearch) can compose its own flow: social profiles, public web
pages, raw image search. No LLM call in this module."""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.client_format import sort_social_first
from app.config import Settings
from app.core.models import (
    ClientPhoto,
    ClientSource,
    ImageSearchRequest,
    ImageSearchResponse,
    SocialSourcesRequest,
    SourcesResponse,
    StepMeta,
    ToolCallLog,
    WebSourcesRequest,
)
from app.search.preview import Mention, classify_source
from app.infra.search_adapter import SearchHit
from app.infra.search_locale import locale_for
from app.core.urls import (
    PROFILE_SITES,
    handle_matches_name,
    identity_words,
    image_key,
    is_news,
    link_key,
    looks_like_fan_account,
    mentions_identity,
    name_from_title,
    name_match,
    name_tokens,
    profile_of,
)


class SourceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _billable(logs: list[ToolCallLog]) -> int:
    return sum(1 for c in logs if not c.cached and not c.error)


def _check(logs: list[ToolCallLog]) -> None:
    if logs and all(c.error for c in logs):
        raise SourceError(502, f"search failed: {logs[0].error}")


def mentions_name(title: str, snippet: str, name: str) -> bool:
    """The page is about the name: its title names the person, or the title/snippet holds every name word."""
    if name_from_title(title, name):
        return True
    need = set(name_tokens(name))
    return bool(need) and any(need <= set(name_tokens(text)) for text in (title, snippet))


def _kg_mentions(kgs: list[dict[str, Any]]) -> list[Mention]:
    out = []
    for kg in kgs:
        url = kg.get("website") or kg.get("descriptionLink")
        if kg.get("title") and isinstance(url, str) and url.startswith("http"):
            out.append(Mention(title=kg["title"], url=url, snippet=kg.get("description") or "", knowledge_graph=bool(kg.get("website"))))
    return out


class SourceLookup:
    def __init__(self, settings: Settings, search: Any):
        self.s = settings
        self.search = search

    def _locale(self, filters: dict[str, str], name: str) -> tuple[str, str]:
        return locale_for(filters, (self.s.search_locale_gl, self.s.search_locale_hl), name)

    async def _run(self, query: str, kind: str, logs: list[ToolCallLog], locale: tuple[str, str]) -> tuple[list[SearchHit], list[dict]]:
        try:
            return await self.search.search(query, kind, logs, locale)
        except RuntimeError as e:
            raise SourceError(503, f"search unavailable: {e}") from e

    async def social(self, req: SocialSourcesRequest) -> SourcesResponse:
        """One profile per network: the top-ranked account whose title names the person.
        The identity query (name + occupation) finds accounts the name query missed;
        those count only when they also describe the person (occupation words)."""
        names = [req.name, *(a for a in req.aliases if a.strip())]
        id_words = identity_words(req.occupation, req.knownFor, req.headline)
        locale = self._locale(req.filters, req.name)
        logs: list[ToolCallLog] = []
        by_name_q = f"{req.name} {PROFILE_SITES}"
        # Accounts usually carry the stage name (HIEUTHUHAI), not the legal name.
        id_q = f"{names[1] if len(names) > 1 else req.name} {id_words[0]} youtube instagram tiktok" if req.withIdentity and id_words else None
        by_name, by_identity = await asyncio.gather(
            self._run(by_name_q, "web", logs, locale),
            self._run(id_q, "web", logs, locale) if id_q else asyncio.sleep(0, result=([], [])),
        )
        _check(logs)

        sources: list[ClientSource] = []
        have: set[str] = set()

        def take(hits: list[SearchHit], need_identity: bool) -> None:
            for h in sorted(hits, key=lambda h: h.position):
                profile = profile_of(h.url)
                if not profile or profile.platform in have or looks_like_fan_account(profile.url, h.title):
                    continue
                if not handle_matches_name(profile.url, *names):
                    continue
                text = f"{h.title} {h.snippet}"
                if need_identity:
                    if not (max(name_match(n, text) for n in names) >= 0.6 and mentions_identity(text, id_words)):
                        continue
                elif not any(name_from_title(h.title, n) for n in names):
                    continue
                have.add(profile.platform)
                sources.append(ClientSource(url=profile.url, title=h.title, type=profile.platform, snippet=h.snippet or None, verified=True))

        take(by_name[0], need_identity=False)
        take(by_identity[0], need_identity=True)
        return SourcesResponse(sources=sort_social_first(sources), meta=StepMeta(searchCalls=_billable(logs)))

    async def web(self, req: WebSourcesRequest) -> SourcesResponse:
        """Pages about the name (official site, Wikipedia, directories...); news only on request."""
        query = " ".join(x for x in (req.name, req.filters.get("location")) if x)
        locale = self._locale(req.filters, req.name)
        logs: list[ToolCallLog] = []
        hits, kgs = await self._run(query, "web", logs, locale)
        _check(logs)
        mentions = _kg_mentions(kgs) + [Mention(title=h.title, url=h.url, snippet=h.snippet) for h in sorted(hits, key=lambda h: h.position)]
        seen: set[str] = set()
        sources: list[ClientSource] = []
        for m in mentions:
            key = link_key(m.url)
            if key in seen or (not req.includeNews and is_news(m.url)) or not mentions_name(m.title, m.snippet, req.name):
                continue
            seen.add(key)
            sources.append(classify_source(m, req.name))
            if len(sources) >= req.limit:
                break
        return SourcesResponse(sources=sort_social_first(sources), meta=StepMeta(searchCalls=_billable(logs)))

    async def images(self, req: ImageSearchRequest) -> ImageSearchResponse:
        locale = self._locale(req.filters, req.name or "")
        logs: list[ToolCallLog] = []
        hits, _ = await self._run(req.query, "image", logs, locale)
        _check(logs)
        seen: set[str] = set()
        photos: list[ClientPhoto] = []
        for h in sorted(hits, key=lambda h: h.position):
            if not h.image_url or not h.image_url.startswith(("http://", "https://")):
                continue
            if req.name and name_match(req.name, f"{h.title} {h.url}") < 0.6:
                continue
            key = image_key(h.image_url)
            if key in seen:
                continue
            seen.add(key)
            photos.append(ClientPhoto(url=h.image_url, pageUrl=h.url, title=h.title or None, fromSource=False, fallbackUrl=h.image_fallback))
            if len(photos) >= req.limit:
                break
        return ImageSearchResponse(photos=photos, meta=StepMeta(searchCalls=_billable(logs)))
