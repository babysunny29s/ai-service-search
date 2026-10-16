"""Preview search: search results only, no LLM.

Candidates are social profiles whose result title carries the searched name.
Facts are read verbatim from result titles/snippets; nothing is inferred.
Same rules as the onesearch web preview, so both produce the same candidates.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field

from app.core.cache import TTLCache
from app.core.client_format import sort_social_first
from app.config import Settings
from app.media.faces import FaceVerifier, distinct_urls, select_photos
from app.core.models import ClientPhoto, PeopleSearchResponse, ToolCallLog, ClientCandidate, ClientKeyFact, ClientMeta, ClientSource
from app.infra.search_adapter import SearchAdapter, SearchHit
from app.infra.search_locale import locale_for
from app.media.source_photos import SourcePhotoFinder
from app.core.urls import (
    PROFILE_SITES,
    SOCIAL_ORDER,
    fold,
    image_key,
    image_on_sources,
    is_news,
    name_from_title,
    name_tokens,
    official_website_of,
    platform_of_url,
    profile_of,
    source_image_query,
)

log = logging.getLogger(__name__)


@dataclass
class Mention:
    title: str
    url: str
    snippet: str = ""
    knowledge_graph: bool = False


# ---------- facts from result titles/snippets ----------

FACT_LABELS = (
    ("role", "Role"),
    ("headline", "Headline"),
    ("company", "Company"),
    ("education", "Education"),
    ("location", "Location"),
    ("audience", "Audience"),
)
_MAX_VALUE = 80
_STOP = r"(?=\s*[·•|;\n]|\.\s|\s+(?:Experience|Education|Location|Kinh nghiệm|Học vấn|Vị trí|Địa điểm)\s*:|$)"
# LinkedIn snippet labels and Facebook intro lines; first match per fact wins.
_SNIPPET_RULES = [
    ("company", re.compile(rf"(?:Experience|Kinh nghiệm)\s*:\s*(.+?){_STOP}", re.I)),
    ("education", re.compile(rf"(?:Education|Học vấn)\s*:\s*(.+?){_STOP}", re.I)),
    ("location", re.compile(rf"(?:Location|Vị trí|Địa điểm)\s*:\s*(.+?){_STOP}", re.I)),
    ("location", re.compile(rf"(?:Lives in|Sống tại|Sống ở)\s+(.+?){_STOP}", re.I)),
    ("company", re.compile(rf"(?:Works at|Làm việc tại)\s+(.+?){_STOP}", re.I)),
    ("education", re.compile(rf"(?:Studied at|Went to|Đã học tại|Học tại)\s+(.+?){_STOP}", re.I)),
]
_AUDIENCE = re.compile(r"(\d[\d.,]*\s*[KMB]?)\s+(Followers|Subscribers|người theo dõi|người đăng ký)", re.I)
_AUDIENCE_PLATFORMS = {"instagram", "tiktok", "x", "youtube", "facebook", "pinterest"}
# Specific enough to tie two results to one person; a city is not.
_IDENTIFYING = ("company", "education")


def _clean(raw: str | None) -> str | None:
    if not raw:
        return None
    v = re.sub(r"\s*(\.\.\.|…)\s*$", "", raw)
    v = re.sub(r"[\s,;:.·•|-]+$", "", " ".join(v.split())).strip()
    if len(v) < 2 or len(v) > _MAX_VALUE:
        return None
    if re.search(r"https?:|@|\d+\+?\s*(connections|kết nối)", v, re.I):
        return None
    return v


def _linkedin_title(title: str, name: str) -> dict[str, str | None]:
    """"Name - Role - Company | LinkedIn" -> role + company; "Name - Something | LinkedIn" -> headline."""
    body = re.sub(r"\s*[|–—-]\s*LinkedIn\s*$", "", title, flags=re.I)
    segs = [s.strip() for s in re.split(r"\s+[-–—|]\s+", body)]
    segs = [s for s in segs if s and fold(s) != fold(name)]
    if len(segs) >= 2:
        return {"role": _clean(segs[0]), "company": _clean(segs[1])}
    if segs:
        return {"headline": _clean(segs[0])}
    return {}


def extract_snippet_facts(m: Mention, platform: str, name: str) -> dict[str, str]:
    out = _linkedin_title(m.title, name) if platform == "linkedin" else {}
    for key, rule in _SNIPPET_RULES:
        if out.get(key):
            continue
        hit = rule.search(m.snippet or "")
        if v := _clean(hit.group(1) if hit else None):
            out[key] = v
    if platform in _AUDIENCE_PLATFORMS and (a := _AUDIENCE.search(f"{m.title} {m.snippet}")):
        kind = "subscribers" if re.search(r"subscri|đăng ký", a.group(2), re.I) else "followers"
        count = re.sub(r"\s+", "", a.group(1))
        out["audience"] = f"{count} {kind}"
    return {k: v for k, v in out.items() if v}


def shares_fact(a: dict[str, str], b: dict[str, str]) -> bool:
    for k in _IDENTIFYING:
        x, y = fold(a.get(k) or ""), fold(b.get(k) or "")
        if not x or not y:
            continue
        if x == y or (min(len(x), len(y)) >= 4 and (x in y or y in x)):
            return True
    return False


def mentions_fact(text: str, facts: dict[str, str]) -> bool:
    """Whole words only, so short names (FPT, VNG) match but never inside another word."""
    words = lambda s: f" {' '.join(name_tokens(s))} "  # noqa: E731
    t = words(text)
    for k in _IDENTIFYING:
        v = facts.get(k) and words(facts[k])
        if v and len(v.strip()) >= 3 and v in t:
            return True
    return False


# ---------- candidates ----------


def classify_source(m: Mention, name: str) -> ClientSource:
    """Social type only for the person's own profile page; posts on social networks are plain web."""
    snippet = m.snippet.strip() or None
    profile = profile_of(m.url)
    if profile and name_from_title(m.title, name):
        return ClientSource(url=profile.url, title=m.title, type=profile.platform, snippet=snippet, verified=True)
    if site := official_website_of(m.url, m.title, name, m.knowledge_graph):
        return ClientSource(url=site, title=m.title, type="website", snippet=snippet, verified=True)
    p = platform_of_url(m.url)
    return ClientSource(url=m.url, title=m.title, type="wikipedia" if p == "wikipedia" else "web", snippet=snippet, verified=True)


@dataclass
class _Draft:
    key: str
    name: str
    name_key: str
    confidence: float
    order: int
    facts: dict[str, str]
    handles: set[str] = field(default_factory=set)
    platforms: set[str] = field(default_factory=set)
    sources: list[ClientSource] = field(default_factory=list)
    brief: list[str] = field(default_factory=list)


def _handle_key(handle: str) -> str | None:
    """Comparable across networks ("Son.LM" / "son_lm"); numeric ids and paths are not handles."""
    if "/" in handle:
        return None
    k = re.sub(r"[@._-]", "", handle.lower())
    return k if len(k) >= 4 and not k.isdigit() else None


def _brief(snippet: str) -> str | None:
    s = " ".join((snippet or "").split())
    return s if len(s) >= 20 else None


def build_preview(query: str, mentions: list[Mention], web_mentions: list[Mention] | None = None, max_candidates: int = 10) -> tuple[list[ClientCandidate], int]:
    """Each social profile anchors a candidate. Another profile joins it only with a
    real shared signal (same handle, or same company/school), never on the name
    alone. Public-web pages are attached only when they mention a candidate's fact."""
    q_tokens = list(dict.fromkeys(name_tokens(query)))
    seen_urls: set[str] = set()
    seen_profiles: set[str] = set()
    drafts: list[_Draft] = []
    web_pages: list[tuple[Mention, str]] = []

    for m in [*mentions, *(web_mentions or [])]:
        name = name_from_title(m.title, query)
        if not name:
            continue
        url_key = re.split(r"[?#]", m.url)[0].rstrip("/")
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        name_key = " ".join(sorted(set(name_tokens(name))))
        profile = profile_of(m.url)
        if not profile:
            web_pages.append((m, name_key))
            continue
        key = f"{profile.platform}:{profile.handle}"
        if key in seen_profiles:
            continue
        seen_profiles.add(key)

        facts = extract_snippet_facts(m, profile.platform, name)
        handle = _handle_key(profile.handle)
        source = ClientSource(url=profile.url, title=m.title, type=profile.platform, snippet=m.snippet or None, verified=True)
        brief = _brief(m.snippet)
        owner = next(
            (
                d
                for d in drafts
                if d.name_key == name_key
                and profile.platform not in d.platforms
                and ((handle and handle in d.handles) or shares_fact(d.facts, facts))
            ),
            None,
        )
        if owner:
            owner.platforms.add(profile.platform)
            if handle:
                owner.handles.add(handle)
            owner.sources.append(source)
            if brief:
                owner.brief.append(brief)
            for k, v in facts.items():
                owner.facts.setdefault(k, v)
            continue
        if len(drafts) >= max_candidates:
            continue
        n_tokens = name_tokens(name)
        drafts.append(
            _Draft(
                key=key,
                name=name,
                name_key=name_key,
                confidence=sum(1 for t in q_tokens if t in n_tokens) / len(q_tokens),
                order=len(drafts),
                facts=facts,
                handles={handle} if handle else set(),
                platforms={profile.platform},
                sources=[source],
                brief=[brief] if brief else [],
            )
        )

    for m, name_key in web_pages:
        d = next((x for x in drafts if x.name_key == name_key and mentions_fact(f"{m.title} {m.snippet}", x.facts)), None)
        if not d:
            continue
        d.sources.append(classify_source(m, d.name))
        if brief := _brief(m.snippet):
            d.brief.append(brief)

    drafts.sort(key=lambda d: (-d.confidence, -len(d.platforms), d.order))
    candidates = [
        ClientCandidate(
            id="wp-" + hashlib.sha1(d.key.encode()).hexdigest()[:12],
            displayName=d.name,
            confidence=round(d.confidence, 2),
            location=d.facts.get("location"),
            platforms=[p for p in SOCIAL_ORDER if p in d.platforms],
            sources=sort_social_first(d.sources),
            keyFacts=[ClientKeyFact(key=k, label=label, value=d.facts[k], verified=True) for k, label in FACT_LABELS if d.facts.get(k)],
            imageSearchQuery=" ".join(x for x in (d.name, d.facts.get("company") or d.facts.get("location")) if x),
            profile=list(dict.fromkeys(d.brief)),
            profileSource="snippets" if d.brief else None,
        )
        for d in drafts
    ]
    return candidates, len(seen_urls)


# ---------- runner ----------


def _filter_terms(query: str, filters: dict[str, str]) -> list[str]:
    parts = [query, filters.get("location"), filters.get("company"), filters.get("jobTitle"), filters.get("education")]
    return [p.strip() for p in parts if p and p.strip()]


def social_query(query: str, filters: dict[str, str]) -> str:
    # Unquoted: Serper's free plan rejects quoted queries with num > 10.
    return " ".join([*_filter_terms(query, filters), PROFILE_SITES])


def public_web_query(query: str, filters: dict[str, str]) -> str:
    # Social bios rarely name an industry, articles and staff pages do.
    return " ".join([*_filter_terms(query, filters), *([filters["field"]] if filters.get("field") else [])])


def _mentions(hits: list[SearchHit], kgs: list[dict]) -> list[Mention]:
    out: list[Mention] = []
    for kg in kgs:
        url = kg.get("website") or kg.get("descriptionLink")
        if kg.get("title") and url:
            out.append(Mention(title=kg["title"], url=url, snippet=kg.get("description") or "", knowledge_graph=bool(kg.get("website"))))
    out.extend(Mention(title=h.title, url=h.url, snippet=h.snippet) for h in hits)
    return [m for m in out if not is_news(m.url)]


class SearchUnavailable(RuntimeError):
    pass


class PreviewSearch:
    def __init__(self, settings: Settings, search: SearchAdapter, faces: FaceVerifier | None = None, source_photos: SourcePhotoFinder | None = None):
        self.s = settings
        self.search = search
        self.faces = faces
        self.source_photos = source_photos
        self.cache: TTLCache[PeopleSearchResponse] = TTLCache(settings.preview_cache_ttl_s, max_items=1000)

    async def run(self, query: str, filters: dict[str, str], max_queries: int | None = None) -> PeopleSearchResponse:
        budget = max(1, min(3, max_queries or self.s.preview_max_queries))
        key = (fold(query), tuple(sorted((k, fold(v)) for k, v in filters.items())), budget)
        if cached := self.cache.get(key):
            return cached.model_copy(update={"meta": cached.meta.model_copy(update={"cacheHit": True, "searchCalls": 0})})

        # Snippets are shown verbatim (no LLM to translate them): local country, English results.
        gl, _ = locale_for(filters, (self.s.search_locale_gl, self.s.search_locale_hl), query)
        locale = (gl, "en")
        logs: list[ToolCallLog] = []
        social_task = self.search.search(social_query(query, filters), "web", logs, locale)
        web_task = self.search.search(public_web_query(query, filters), "web", logs, locale) if budget >= 2 else None
        try:
            social, web = await asyncio.gather(social_task, web_task or asyncio.sleep(0, result=None))
        except RuntimeError as e:
            raise SearchUnavailable(str(e)) from e
        if not social[0] and logs and all(c.error for c in logs):
            raise SearchUnavailable(logs[0].error or "search failed")

        candidates, total = build_preview(
            query,
            _mentions(*social),
            _mentions(*web) if web else None,
            max_candidates=self.s.preview_max_candidates,
        )
        candidates = [c for c in candidates if c.sources]
        # Free photos from the candidates' own pages first (they become the profile photo);
        # the image search only fills what is missing.
        if self.source_photos:
            await asyncio.gather(*(self._source_photos(c) for c in candidates[: self.s.source_photo_candidates]))
        if budget >= 3:
            short = [c for c in candidates[: self.s.preview_photo_candidates] if len(c.photos) < self._limit]
            await asyncio.gather(*(self._photos(c, logs, locale) for c in short))

        out = PeopleSearchResponse(
            mode="preview",
            status="no_match" if not candidates else "resolved" if len(candidates) == 1 else "ambiguous",
            candidates=candidates,
            meta=ClientMeta(searchCalls=sum(1 for c in logs if not c.cached and not c.error), totalSources=total),
        )
        self.cache.set(key, out)
        return out

    @property
    def _limit(self) -> int:
        return max(1, self.s.initial_photos)

    async def _source_photos(self, c: ClientCandidate) -> None:
        """No search credit: images of the candidate's own pages (Wikipedia, GitHub, named
        portraits, og:image)."""
        try:
            photos, _ = await self.source_photos.find(c.displayName, c.sources, limit=self._limit)
        except Exception as e:
            log.warning("source photos for %s failed: %s", c.displayName, e)
            return
        if photos:
            c.photos = photos
            c.avatarUrl = photos[0].url

    async def _photos(self, c: ClientCandidate, logs: list[ToolCallLog], locale: tuple[str, str]) -> None:
        """Image search after the source photos: adds results up to the limit. With the face
        model, a source photo is the reference face, else the face most results agree on."""
        query = source_image_query(c.displayName, c.sources)
        if not query:
            return
        try:
            hits, _ = await self.search.search(query, "image", logs, locale)
        except RuntimeError:
            return
        own = list(c.photos)
        check = bool(self.faces and await self.faces.ready())
        seen = {image_key(p.url) for p in own}
        found: list[ClientPhoto] = []
        for h in hits:
            if not (h.image_url and (key := image_key(h.image_url)) not in seen):
                continue
            if not image_on_sources(h.url, h.image_url, c.sources, h.title or ""):
                continue
            seen.add(key)
            found.append(ClientPhoto(url=h.image_url, pageUrl=h.url, title=h.title or None, fromSource=True, fallbackUrl=h.image_fallback))
            if len(found) >= (8 if check else self._limit):
                break
        if check and found:
            pool = own + found
            faces, used = await self.faces.faces_or_fallback([p.url for p in pool], [p.fallbackUrl for p in pool])
            pool = [p if u == p.url else p.model_copy(update={"url": u, "fallbackUrl": None}) for p, u in zip(pool, used)]
            kept = select_photos(faces, [True] * len(own) + [False] * len(found), self.s.face_gallery_threshold)
            found = sorted(
                (pool[i].model_copy(update={"faceMatch": score}) for i, score in kept if i >= len(own)), key=lambda p: -(p.faceMatch or 0)
            )
        merged = own + found
        if check:
            # The source photo often comes back from the image search under another URL.
            merged = [merged[i] for i in await distinct_urls(self.faces, [p.url for p in merged])]
        c.photos = merged[: self._limit]
        c.avatarUrl = c.photos[0].url if c.photos else None
