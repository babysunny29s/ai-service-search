"""Name search (searchprofile).

Up to two Serper queries: the name plus filters on Wikipedia/LinkedIn/Facebook,
then the same full name on Instagram/TikTok/X. A Wikipedia article about the
person is a profile and is listed first; the English edition is kept when it exists.
Stage-name titles (e.g. "Độ Mixi" for Phùng Thanh Độ) are kept for the merge;
the article is then scraped to confirm it is this person. Posts and videos are
dropped. One LLM call then drops namesakes and merges the same person across
networks. A candidate is returned only when at least one result is their own
profile page. A Wikipedia article is scraped (text, Wikidata facts, external
profile links, lead image) when it is a source; a mismatch is dropped. A photo
is taken from those profile sources when they have one. Face matching and extra
image search wait until deep search.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from app.core import tracing
from app.core.cache import TTLCache
from app.core.client_format import slug_id, sort_social_first
from app.config import Settings
from app.core.llm import LLMProvider, MergedPerson
from app.core.models import (
    ClientCandidate,
    ClientKeyFact,
    ClientMeta,
    ClientPhoto,
    ClientSource,
    PeopleSearchResponse,
    ToolCallLog,
)
from app.infra.fetcher import PageFetcher
from app.search.preview import FACT_LABELS, Mention, classify_source, extract_snippet_facts
from app.infra.search_adapter import SearchAdapter
from app.infra.search_locale import locale_for
from app.media.source_photos import SourcePhotoFinder
from app.infra.wiki import english_wiki_url, fact_items, one_wikipedia, paragraphs, wiki_host_rank, wiki_target
from app.core.urls import (
    SOCIAL_ORDER,
    fold,
    image_key,
    is_news,
    looks_like_image_url,
    link_key,
    looks_like_fan_account,
    source_image_query,
    name_from_title,
    name_match,
    name_score,
    name_tokens,
    tokens_close,
    official_website_of,
    platform_of_url,
    profile_of,
)

log = logging.getLogger(__name__)

# Two social-only queries. One giant site: OR collapses onto a single network,
# and site:youtube.com / site:facebook.com otherwise fill the page with videos and posts.
_SITE_GROUPS = (
    "(site:en.wikipedia.org OR site:wikipedia.org OR site:linkedin.com/in OR site:facebook.com)",
    "(site:instagram.com OR site:tiktok.com OR site:x.com OR site:twitter.com)",
)
_WIKI_SKIP = ("category:", "file:", "image:", "template:", "wikipedia:", "special:", "help:", "portal:", "draft:", "module:", "talk:")
_SURNAMES = {"nguyen", "tran", "pham", "hoang", "huynh", "phan", "dang", "bui", "ngo", "duong", "truong", "dinh", "trinh", "doan", "luu", "luong", "trieu", "phung", "vuong"}
# These show up on the profile itself. Occupation and field rarely do, so they
# only go on the last query (channels and GitHub bios).
_ON_PROFILE = ("location", "company", "jobTitle", "education")
_ON_ROLE = ("occupation", "field", "jobTitle")
_PROFILE_HOSTS = ("about.me", "linktr.ee", "bio.link", "beacons.ai", "campsite.bio", "lnk.bio", "behance.net", "carrd.co")
_PROFILE_PATH = ("/about", "/profile", "/team/", "/people/", "/staff/", "/author/", "/bio")
_MAX_HITS = 24
_SNIPPET_CHARS = 400


class SearchProfileError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class ProfileHit:
    id: str
    title: str
    url: str
    snippet: str = ""
    knowledge_graph: bool = False
    image: str | None = None
    image_fallback: str | None = None


def profile_queries(name: str, filters: dict[str, str]) -> list[str]:
    """Two queries, both with the full typed name: Wikipedia/LinkedIn/Facebook,
    then Instagram/TikTok/X (plus role filters). Never more than two."""
    base = " ".join([name, *[filters[k] for k in _ON_PROFILE if filters.get(k)]])
    social = " ".join(dict.fromkeys([
        name,
        *[filters[k] for k in (*_ON_PROFILE, *_ON_ROLE) if filters.get(k)],
    ]))
    return [f"{text} {group}" for text, group in zip((base, social), _SITE_GROUPS)]


def more_profile_queries(name: str, filters: dict[str, str], page: int = 2) -> list[tuple[str, int]]:
    """The same two queries as the first round, on the next Serper page."""
    page = max(1, page)
    return [(q, page) for q in profile_queries(name, filters)]


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return re.sub(r"^(www|m|mobile)\.", "", host)


def social_account(url: str, title: str, name: str):
    """The person's own social profile, or None for posts, videos, fan pages
    and accounts whose title does not name them."""
    profile = profile_of(url)
    if not profile or looks_like_fan_account(profile.url, title):
        return None
    # The title has to name them. A matching handle alone is not enough
    # (fan pages and namesakes), and abbreviated slugs ("sonlm") are normal.
    if not name_from_title(title, name) and name_score(name, title) < 0.45 and name_match(name, title) < 0.6:
        return None
    return profile


def _wiki_names(title: str, *names: str) -> bool:
    """The article title is about this person. A stage-name title ("Jennie (singer)",
    "Độ Mixi") still matches a fuller legal name when aliases or shared given-name
    tokens overlap. A surname alone does not."""
    head = re.split(r"\s+[-|–—]\s+|\(", title, maxsplit=1)[0]
    head_tokens = name_tokens(head)
    for name in names:
        name = " ".join((name or "").split())
        if len(name) < 2:
            continue
        if name_from_title(title, name) or name_match(name, title) >= 0.8:
            return True
        query_tokens = set(name_tokens(name))
        # len >= 4 keeps short stage tokens ("mixi") while surnames stay out.
        shared = [
            t
            for t in head_tokens
            if len(t) >= 4 and t not in _SURNAMES and any(tokens_close(t, q) for q in query_tokens)
        ]
        if shared:
            return True
    return False


def wiki_person_url(url: str) -> str | None:
    """Canonical /wiki/ article URL, or None for File:/Category:/disambiguation/…"""
    if platform_of_url(url) != "wikipedia":
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    path = unquote(parts.path or "")
    if not path.lower().startswith("/wiki/"):
        return None
    slug = path[len("/wiki/") :]
    if not slug or slug.lower().startswith(_WIKI_SKIP) or "disambiguation" in slug.lower():
        return None
    host = parts.hostname or "en.wikipedia.org"
    return urlunsplit(("https", host, "/wiki/" + quote(slug, safe="()_,'-"), "", ""))


def wiki_article(url: str, title: str, name: str, *also_names: str) -> str | None:
    """Canonical Wikipedia URL when the title clearly names this person (or an alias)."""
    canon = wiki_person_url(url)
    if not canon:
        return None
    try:
        slug = unquote(urlsplit(canon).path)[len("/wiki/") :]
    except ValueError:
        slug = ""
    if not _wiki_names(title or slug.replace("_", " "), name, *also_names):
        return None
    return canon


def wiki_hit(url: str, title: str, name: str, snippet: str = "") -> str | None:
    """Keep a Wikipedia Serper hit for merge. Stage-name titles are allowed when the
    query still weakly matches title/snippet; `_fill_wiki` scrapes to confirm."""
    if wiki_article(url, title, name):
        return wiki_person_url(url)
    canon = wiki_person_url(url)
    if not canon:
        return None
    blob = f"{title} {snippet}"
    if name_match(name, blob) >= 0.4 or name_match(blob, name) >= 0.4:
        return canon
    return None


def wiki_page_about(title: str, extract: str, *names: str) -> bool:
    """True when scraped Wikipedia text is about one of these names (legal or stage).
    The lead often states the legal name when the title is a stage name."""
    if _wiki_names(title, *names):
        return True
    body = (extract or "").strip()
    for name in names:
        name = " ".join((name or "").split())
        if len(name) < 2:
            continue
        if body and name_match(name, body) >= 0.5:
            return True
        # No extract yet: only a strong title match counts (not a weak stage-name overlap).
        if not body and (name_from_title(title, name) or name_match(name, title) >= 0.8):
            return True
    return False


def is_profile_page(
    url: str,
    title: str,
    name: str,
    knowledge_graph: bool = False,
    aliases: list[str] | tuple[str, ...] = (),
) -> bool:
    """A page that is this person's own profile: a social account, a personal
    site, a Wikipedia article, or an about/team page titled with their name."""
    if looks_like_fan_account(url, title) or is_news(url):
        return False
    if social_account(url, title, name):
        return True
    if official_website_of(url, title, name, knowledge_graph):
        return True
    also = [a for a in aliases if a and a.strip()]
    if wiki_article(url, title, name, *also):
        return True
    # Stage-name Wikipedia the merge already attached: keep for scrape confirmation.
    if wiki_person_url(url) and (
        name_match(name, title) >= 0.4
        or any(name_match(a, title) >= 0.4 or name_match(title, a) >= 0.4 for a in also)
    ):
        return True
    host = _host(url)
    blob = f"{title} {url}"
    if any(host == h or host.endswith("." + h) for h in _PROFILE_HOSTS):
        return name_from_title(title, name) is not None or name_match(name, blob) >= 0.8
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        return False
    if any(seg in path for seg in _PROFILE_PATH):
        return name_from_title(title, name) is not None
    return False


def _image_query(name: str, raw: str | None, facts: dict[str, str]) -> str:
    text = re.sub(r"https?://\S+", " ", raw or "")
    text = " ".join(re.sub(r"[\[\]()]", " ", text).split())
    if len(text) < 3 or name_match(name, text) < 0.4:
        extra = facts.get("company") or facts.get("role") or facts.get("location") or ""
        text = " ".join(p for p in (name, extra) if p)
    return text[:80]


def _fits_name(title: str, person: str, query: str) -> bool:
    """Drop a result whose title leads with a different person's name.
    Handle-only titles ("@son.lm") stay: they often omit the legal name."""
    if name_from_title(title, person) or name_from_title(title, query):
        return True
    if name_match(person, title) >= 0.4 or name_match(query, title) >= 0.4:
        return True
    lead = re.split(r"\s+[-|–—·•]\s+", title, maxsplit=1)[0]
    lead_tokens = [t for t in re.split(r"[^a-z0-9]+", fold(lead)) if t]
    if 2 <= len(lead_tokens) <= 5 and "@" not in lead:
        return False
    return True


def assemble_candidates(query: str, hits: list[ProfileHit], people: list[MergedPerson], max_candidates: int) -> list[ClientCandidate]:
    """Keep the model's groups, then drop anyone without a real profile page."""
    by_id = {h.id: h for h in hits}
    out: list[ClientCandidate] = []
    seen_profiles: set[str] = set()
    for person in people:
        name = " ".join(person.displayName.split())
        if len(name) < 2 or (name_score(query, name) < 0.4 and name_match(query, name) < 0.4 and name_match(name, query) < 0.4):
            continue
        aliases = [a.strip() for a in person.aliases if a.strip() and fold(a) != fold(name)]
        chosen = [by_id[i] for i in dict.fromkeys(person.sourceIds) if i in by_id and _fits_name(by_id[i].title, name, query)]
        profiles: list[ProfileHit] = []
        seen_local: set[str] = set()
        for h in chosen:
            key = link_key(h.url)
            if key in seen_local:
                continue
            # Wikipedia the model attached is kept even when the title is a stage
            # name; `_fill_wiki` scrapes the article and drops a mismatch.
            if is_profile_page(h.url, h.title, name, h.knowledge_graph, aliases) or wiki_person_url(h.url):
                seen_local.add(key)
                profiles.append(h)
        profiles = [h for h in profiles if link_key(h.url) not in seen_profiles]
        if not profiles:
            continue
        for h in profiles:
            seen_profiles.add(link_key(h.url))
        facts: dict[str, str] = {}
        sources: list[ClientSource] = []
        briefs: list[str] = []
        platforms: set[str] = set()
        photos: list[ClientPhoto] = []
        seen_images: set[str] = set()
        for h in profiles:
            mention = Mention(h.title, h.url, h.snippet, h.knowledge_graph)
            src = classify_source(mention, name)
            if profile_of(h.url) and src.type not in SOCIAL_ORDER and src.type != "website":
                src = src.model_copy(update={"type": profile_of(h.url).platform})
            sources.append(src)
            if src.type in SOCIAL_ORDER:
                platforms.add(src.type)
            for k, v in extract_snippet_facts(mention, src.type if src.type in SOCIAL_ORDER else "", name).items():
                facts.setdefault(k, v)
            brief = " ".join(h.snippet.split())
            if len(brief) >= 20:
                briefs.append(brief)
            if h.image and h.image.startswith("http") and (key := image_key(h.image)) not in seen_images:
                seen_images.add(key)
                photos.append(ClientPhoto(url=h.image, pageUrl=h.url, title=h.title or None, fromSource=True, fallbackUrl=h.image_fallback))
        # Non-profile results the model attached (a staff page that failed the
        # profile test still stays out). Only profile pages are returned here.
        sources = one_wikipedia(sort_social_first(list({link_key(s.url): s for s in sources}.values())))
        sources.sort(key=lambda s: s.type != "wikipedia")
        loc = person.location or facts.get("location")
        occupation = person.occupation
        headline = " ".join((person.headline or "").split())[:160]
        if headline and headline.lower() == name.lower():
            headline = None
        if not headline and facts.get("role"):
            headline = " · ".join(p for p in (facts.get("role"), facts.get("company")) if p)
        key_facts = [ClientKeyFact(key=k, label=label, value=facts[k], verified=True) for k, label in FACT_LABELS if facts.get(k)]
        anchor = sources[0].url if sources else name
        confidence = min(1.0, max(0.0, person.confidence)) or 0.5
        out.append(
            ClientCandidate(
                id=slug_id(name, link_key(anchor)),
                displayName=name,
                aliases=aliases[:8],
                headline=headline or None,
                confidence=round(confidence, 2),
                disambiguationNote=(" ".join((person.disambiguationNote or "").split())[:400] or None),
                occupation=occupation,
                location=loc,
                platforms=[p for p in SOCIAL_ORDER if p in platforms],
                sources=sources,
                keyFacts=key_facts,
                photos=photos[:3],
                avatarUrl=photos[0].url if photos else None,
                imageSearchQuery=source_image_query(name, sources) or _image_query(name, person.imageSearchQuery, facts),
                profile=list(dict.fromkeys(briefs))[:3],
                profileSource="snippets" if briefs else None,
            )
        )
    out.sort(key=lambda c: (-c.confidence, -name_score(query, c.displayName), -len(c.platforms)))
    return out[:max_candidates]


class SearchProfile:
    def __init__(
        self,
        settings: Settings,
        search: SearchAdapter,
        llm: LLMProvider | None,
        source_photos: SourcePhotoFinder | None = None,
        fetcher: PageFetcher | None = None,
    ):
        self.s = settings
        self.search = search
        self.llm = llm
        self.source_photos = source_photos
        self.fetcher = fetcher
        self.cache: TTLCache[PeopleSearchResponse] = TTLCache(settings.preview_cache_ttl_s, max_items=1000)

    async def run(
        self,
        query: str,
        filters: dict[str, str],
        exclude_urls: list[str] | None = None,
        exclude_names: list[str] | None = None,
        page: int = 1,
    ) -> PeopleSearchResponse:
        if self.llm is None:
            raise SearchProfileError(503, "LLM provider is not configured")
        skip_urls = {link_key(u) for u in (exclude_urls or []) if u}
        skip_names = {fold(n) for n in (exclude_names or []) if n and n.strip()}
        page = max(1, page)
        if page == 1 and (skip_urls or skip_names):
            page = 2
        more = page > 1 or bool(skip_urls or skip_names)
        key = ("v24", fold(query), tuple(sorted((k, fold(v)) for k, v in filters.items())))
        if not more:
            if cached := self.cache.get(key):
                return cached.model_copy(update={"meta": cached.meta.model_copy(update={"cacheHit": True, "searchCalls": 0, "llmCalls": 0})})

        with tracing.span(
            "search-profile",
            input={"query": query, "filters": filters, "page": page},
            tags=["search-profile"],
        ):
            try:
                return await self._run(query, filters, skip_urls, skip_names, page, more, key, exclude_names)
            finally:
                tracing.flush()

    async def _run(
        self,
        query: str,
        filters: dict[str, str],
        skip_urls: set[str],
        skip_names: set[str],
        page: int,
        more: bool,
        key: tuple,
        exclude_names: list[str] | None = None,
    ) -> PeopleSearchResponse:
        locale = locale_for(filters, (self.s.search_locale_gl, self.s.search_locale_hl), query)
        logs: list[ToolCallLog] = []
        planned = more_profile_queries(query, filters, page)
        hits = await self._search(query, filters, logs, locale, planned)
        if skip_urls:
            hits = [h for h in hits if link_key(h.url) not in skip_urls]
        people: list[MergedPerson] = []
        llm_calls = 0
        if hits:
            try:
                merged = await self.llm.merge_profiles(
                    query,
                    filters,
                    [{"sourceId": h.id, "url": h.url, "title": h.title, "snippet": h.snippet[:_SNIPPET_CHARS]} for h in hits],
                    already_listed=list(dict.fromkeys(n for n in (exclude_names or []) if n and n.strip())) or None,
                )
                llm_calls = 1
                people = merged.candidates
            except Exception as e:
                raise SearchProfileError(502, f"profile merge failed: {str(e)[:200]}") from e
        candidates = assemble_candidates(query, hits, people, self.s.max_candidates)
        if skip_names or skip_urls:
            candidates = [
                c
                for c in candidates
                if fold(c.displayName) not in skip_names and not any(link_key(s.url) in skip_urls for s in c.sources)
            ]
        await self._fill_wiki(candidates)
        await self._attach_photos(candidates)
        out = PeopleSearchResponse(
            mode="searchprofile",
            status="no_match" if not candidates else "resolved" if len(candidates) == 1 else "ambiguous",
            candidates=candidates,
            meta=ClientMeta(
                searchCalls=sum(1 for c in logs if not c.cached and not c.error),
                llmCalls=llm_calls,
                totalSources=len(hits),
                **tracing.client_cost(tracing.current_stats()),
            ),
        )
        if not more and not any(c.error for c in logs):
            self.cache.set(key, out)
        return out

    async def _fill_wiki(self, candidates: list[ClientCandidate]) -> None:
        """When a candidate's profile is Wikipedia, read the article (and Wikidata) for biography, facts and links.
        Drop the Wikipedia source when the scraped page is not about this person."""
        if self.fetcher is None:
            return
        await asyncio.gather(*(self._fill_one_wiki(c) for c in candidates if any(s.type == "wikipedia" for s in c.sources)))
        # A rejected wiki may have been the only profile; drop people with nothing left.
        candidates[:] = [
            c
            for c in candidates
            if any(
                s.type == "wikipedia"
                or profile_of(s.url)
                or is_profile_page(s.url, s.title or "", c.displayName, aliases=c.aliases)
                for s in c.sources
            )
        ]

    async def _fill_one_wiki(self, c: ClientCandidate) -> None:
        wiki = next((s for s in c.sources if s.type == "wikipedia"), None)
        if wiki is None or self.fetcher is None:
            return
        try:
            page = await self.fetcher.wikipedia_page(wiki.url)
        except Exception as e:
            log.warning("wikipedia read for %s failed: %s", c.displayName, e)
            return
        names = (c.displayName, *(c.aliases or []))
        if not wiki_page_about(page.title or wiki.title or "", page.extract or "", *names):
            log.info("wikipedia %s is not about %s; dropping", wiki.url, c.displayName)
            c.sources = [s for s in c.sources if s.type != "wikipedia"]
            return
        en = english_wiki_url(page)
        if en and wiki_target(en):
            wiki.url = en
            if page.title:
                wiki.title = page.title
        elif page.title:
            wiki.title = wiki.title or page.title
        if page.extract:
            wiki.snippet = page.extract[:_SNIPPET_CHARS]
            bio = paragraphs(page.extract)
            if bio:
                c.profile = bio
                c.profileSource = "wikipedia"
        have = {f.key: f for f in c.keyFacts}
        extra_facts: list[ClientKeyFact] = []
        for key, label, value in fact_items(page.facts):
            if key in have:
                continue
            fact = ClientKeyFact(key=key, label=label, value=value, verified=True)
            extra_facts.append(fact)
            have[key] = fact
        c.keyFacts = extra_facts + c.keyFacts
        if not c.occupation and page.facts.get("occupation"):
            c.occupation = page.facts["occupation"]
        if not c.location and (page.facts.get("location") or page.facts.get("nationality")):
            c.location = page.facts.get("location") or None
        if not c.nationality and page.facts.get("nationality"):
            c.nationality = page.facts["nationality"]
        if not c.headline and c.occupation:
            c.headline = c.occupation[:160]
        if not c.knownFor and page.facts.get("knownFor"):
            c.knownFor = page.facts["knownFor"]
        if page.image and page.image.startswith("https://") and image_key(page.image) not in {image_key(p.url) for p in c.photos}:
            c.photos = [ClientPhoto(url=page.image, pageUrl=wiki.url, title=wiki.title or None, fromSource=True), *c.photos]
            c.avatarUrl = c.avatarUrl or page.image
        seen = {link_key(s.url) for s in c.sources}
        added: list[ClientSource] = []
        sites = [page.facts["website"]] if page.facts.get("website", "").startswith("http") else []
        # Profile pages written on the article: Instagram, Facebook, X, TikTok,
        # YouTube channel/@/c/user (not a watch URL), LinkedIn, GitHub, Pinterest,
        # Reddit, plus the official website. News citations stay out.
        for url in [*page.links, *sites]:
            if looks_like_image_url(url):
                continue
            if link_key(url) in seen:
                continue
            account = profile_of(url)
            if account:
                seen.add(link_key(account.url))
                added.append(ClientSource(url=account.url, title=f"{c.displayName} · {account.platform}", type=account.platform, snippet="Listed on Wikipedia", verified=True))
                continue
            site = official_website_of(url, "", c.displayName)
            if site and link_key(site) not in seen:
                seen.add(link_key(site))
                added.append(ClientSource(url=site, title=c.displayName, type="website", snippet="Official site on Wikipedia", verified=True))
        site = page.facts.get("website") or ""
        if site.startswith("http") and link_key(site) not in seen:
            added.append(ClientSource(url=site, title=c.displayName, type="website", snippet="Official site on Wikipedia", verified=True))
        c.sources = one_wikipedia(c.sources)
        if added:
            c.sources = one_wikipedia(sort_social_first(c.sources + added))
            c.sources.sort(key=lambda s: s.type != "wikipedia")
            c.platforms = [p for p in SOCIAL_ORDER if p in {s.type for s in c.sources}]

    @property
    def _photo_limit(self) -> int:
        return max(1, self.s.max_photos, self.s.initial_photos)

    async def _attach_photos(self, candidates: list[ClientCandidate]) -> None:
        """Keep photos that already sit on the profile sources (search hit, Wikipedia, GitHub)."""
        if self.source_photos:
            limit = self._photo_limit
            await asyncio.gather(*(self._source_photos(c, limit) for c in candidates))
        for c in candidates:
            if c.photos and not c.avatarUrl:
                c.avatarUrl = c.photos[0].url

    async def _source_photos(self, c: ClientCandidate, limit: int) -> None:
        if not self.source_photos or len(c.photos) >= limit:
            return
        try:
            photos, _ = await self.source_photos.find(c.displayName, c.sources, limit=limit, require_face=False)
        except Exception as e:
            log.warning("source photos for %s failed: %s", c.displayName, e)
            return
        if not photos:
            return
        have = {image_key(p.url) for p in c.photos}
        extra = [p for p in photos if image_key(p.url) not in have]
        c.photos = (c.photos + extra)[:limit]
        c.avatarUrl = c.photos[0].url if c.photos else c.avatarUrl

    async def _search(
        self,
        query: str,
        filters: dict[str, str],
        logs: list[ToolCallLog],
        locale: tuple[str, str],
        planned: list[tuple[str, int]] | None = None,
    ) -> list[ProfileHit]:
        planned = planned or [(q, 1) for q in profile_queries(query, filters)]
        try:
            batches = await asyncio.gather(*(self.search.search(q, "web", logs, locale, page=page) for q, page in planned))
        except RuntimeError as e:
            raise SearchProfileError(503, str(e)[:300]) from e
        if logs and all(c.error for c in logs):
            raise SearchProfileError(503, logs[0].error or "search failed")

        hits: list[ProfileHit] = []
        seen: set[str] = set()

        def add(title: str, url: str, snippet: str, knowledge_graph: bool = False, image: str | None = None, image_fallback: str | None = None) -> None:
            account = social_account(url, title, query)
            article = None if account else wiki_hit(url, title, query, snippet)
            if account:
                url = account.url
            elif article:
                url = article
            else:
                return
            if not title or link_key(url) in seen:
                return
            seen.add(link_key(url))
            hits.append(ProfileHit(id=f"src_{len(hits) + 1}", title=title, url=url, snippet=snippet or "", knowledge_graph=knowledge_graph, image=image, image_fallback=image_fallback))

        for (found, kgs) in batches:
            for kg in kgs:
                url = kg.get("website") or kg.get("descriptionLink")
                if kg.get("title") and url:
                    add(kg["title"], url, kg.get("description") or "", bool(kg.get("website")), kg.get("imageUrl") or kg.get("image"))
            for h in found:
                add(h.title, h.url, h.snippet, image=h.image_url, image_fallback=h.image_fallback)
        hits.sort(
            key=lambda h: (
                platform_of_url(h.url) != "wikipedia",
                wiki_host_rank(h.url) if platform_of_url(h.url) == "wikipedia" else 1,
                not is_profile_page(h.url, h.title, query, h.knowledge_graph),
            )
        )
        for i, h in enumerate(hits[:_MAX_HITS], 1):
            h.id = f"src_{i}"
        return hits[:_MAX_HITS]
