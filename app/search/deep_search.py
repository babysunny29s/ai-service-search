"""Deep search of one person already returned by searchprofile.

The selected candidate JSON is the input. The person's own pages are read
first (Wikipedia, YouTube description, GitHub bio, public sites). Facebook,
Instagram, TikTok, X and LinkedIn are read through Scrape Creators (at most two
profile reads). Every public field on those pages is passed to the model.
One web query then collects articles, and one LLM call writes the dossier.
Photos with a face fill the gallery; otherwise the source avatar still stands.
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.core import tracing
from app.core.client_format import audience_fact, onesearch_facts, social_profiles_text, sort_social_first
from app.config import Settings
from app.search.enrich import _born_supported, _fact_key
from app.media.faces import FaceVerifier, distinct_urls, select_photos
from app.infra.fetcher import PageFetcher, excerpt
from app.infra.about_profiles import github_about, youtube_about
from app.core.llm import DeepProfileOutput, LLMProvider
from app.core.models import (
    ClientCandidate,
    ClientCareer,
    ClientKeyFact,
    ClientMeta,
    ClientPhoto,
    ClientSource,
    DeepSearchResponse,
    SocialSourcesRequest,
    ToolCallLog,
    utcnow,
)
from app.infra.search_adapter import SearchAdapter
from app.infra.search_locale import locale_for
from app.search.search_profile import is_profile_page
from app.search.sources import SourceLookup
from app.media.source_photos import SourcePhotoFinder
from app.core.urls import account_handle, can_fetch, certain_profile, fold, image_key, is_news, link_key, looks_like_fan_account, looks_like_image_url, name_from_title, name_match, name_tokens, official_website_of, platform_of_url, profile_of
from app.infra.social_crawl import CrawlBatch, SocialCrawl, crawl_target
from app.infra.wiki import english_wiki_url, fact_items, is_wikipedia, one_wikipedia, wiki_target

log = logging.getLogger(__name__)

PHOTO_MAX = 10
_MAX_ARTICLES = 8
_READ_PAGES = 4
_PAGE_CHARS = 900
_WIKI_CHARS = 4000
_MAX_PARAGRAPHS = 4
_MAX_FACTS = 18
_SNIPPET = 400
_LABELS = {
    "fullName": "Full name",
    "born": "Born",
    "nationality": "Nationality",
    "location": "Location",
    "jobTitle": "Current role",
    "occupation": "Occupation",
    "company": "Company",
    "education": "Education",
    "knownFor": "Known for",
    "audience": "Audience",
    "industry": "Industry",
    "field": "Field",
    "notableWorks": "Notable works",
    "specialization": "Specialization",
    "awards": "Awards",
    "teams": "Teams",
    "yearsActive": "Years active",
    "website": "Website",
    "age": "Age",
    "email": "Email",
    "phone": "Phone",
}


_LINK = re.compile(r"https?://[^\s<>\"')\]]+")
_ASSET = re.compile(r"\.(?:jpg|jpeg|png|gif|webp|svg|css|js|ico|mp4|mp3)(?:\?|$)", re.I)


def urls_in(text: str) -> list[str]:
    """http(s) links written in a page we already read."""
    out: list[str] = []
    for raw in _LINK.findall(text or ""):
        url = raw.rstrip(".,;)")
        if url.startswith("http") and not _ASSET.search(url) and not looks_like_image_url(url):
            out.append(url)
    return out


def _year_stated(value: str, blob: str) -> bool:
    """The birth year is written in a page we already read."""
    years = re.findall(r"\b(1[89]\d\d|20\d\d)\b", value)
    return bool(years) and all(year in fold(blob) for year in years)


def _read_blob(page_text: dict[str, str], sources: list[ClientSource]) -> str:
    return "\n".join([*page_text.values(), *(f"{s.title} {s.snippet or ''}" for s in sources)])


_STATED = (
    ("born", "Born", r"(?:sinh\s+năm|sinh\s+nam|born(?:\s+in|\s+on)?|date\s+of\s+birth|ngày\s+sinh|ngay\s+sinh)\s*:?\s+((?:\S+\s+){0,3}(?:1[89]\d\d|20\d\d))"),
    ("location", "Location", r"(?:sống\s+tại|song\s+tai|sống\s+ở|song\s+o|lives\s+in|quê\s+quán|que\s+quan|nơi\s+ở|noi\s+o|location|country)\s*:?\s+([^.\n]{2,40})"),
    ("education", "Education", r"(?:học\s+tại|hoc\s+tai|tốt\s+nghiệp|tot\s+nghiep|education)\s*:?\s+([^.\n]{2,60})"),
    ("nationality", "Nationality", r"(?:quốc\s+tịch|quoc\s+tich|nationality)\s*:?\s+([^.\n]{2,30})"),
)
_STREET = re.compile(r"\b(duong|ngo|hem|street|apartment|so nha)\b|^\d")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s.\-]*){9,12}(?!\d)")
_AGE = re.compile(r"(?:(?:\bage|tuổi|tuoi)\s*:?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:tuổi|years?\s+old)\b)", re.I)


def facts_from_pages(page_text: dict[str, str]) -> list[ClientKeyFact]:
    """Personal facts written on a page we read. A street address is not one of them."""
    blob = "\n".join(page_text.values())
    out: list[ClientKeyFact] = []
    seen: set[str] = set()
    for key, label, pattern in _STATED:
        match = re.search(pattern, blob, re.I)
        if not match or key in seen:
            continue
        value = " ".join(match.group(1).split()).strip(" .,-")
        if not value or (key == "location" and _STREET.search(fold(value))):
            continue
        seen.add(key)
        out.append(ClientKeyFact(key=key, label=label, value=value[:240], verified=True))
    if "email" not in seen and (found := _EMAIL.search(blob)):
        out.append(ClientKeyFact(key="email", label="Email", value=found.group(0), verified=True))
    if "phone" not in seen and (found := _PHONE.search(blob)):
        digits = re.sub(r"\D", "", found.group(0))
        if 9 <= len(digits) <= 15:
            out.append(ClientKeyFact(key="phone", label="Phone", value=" ".join(found.group(0).split()), verified=True))
    if "age" not in seen and (found := _AGE.search(blob)):
        years = int(found.group(1) or found.group(2))
        if 1 <= years <= 120:
            out.append(ClientKeyFact(key="age", label="Age", value=str(years), verified=True))
    return out


def classify_links(urls: list[str], name: str, found_on: str) -> tuple[list[ClientSource], list[ClientSource]]:
    """Every link written on this person's own profile, split into profile pages and articles.
    A profile link stays even when the handle does not contain the whole name."""
    profiles: list[ClientSource] = []
    articles: list[ClientSource] = []
    seen: set[str] = set()
    for url in urls:
        url = (url or "").strip()
        if not url.startswith("http"):
            continue
        key = link_key(url)
        if key in seen:
            continue
        seen.add(key)
        if looks_like_image_url(url):
            continue
        account = profile_of(url)
        if account:
            profiles.append(
                ClientSource(
                    url=account.url,
                    title=f"{name} · {account.platform}",
                    type=account.platform,
                    snippet=f"Listed on {found_on}",
                    verified=True,
                )
            )
            continue
        kind = "wikipedia" if platform_of_url(url) == "wikipedia" else "web"
        articles.append(
            ClientSource(url=url, title=name, type=kind, snippet=f"Listed on {found_on}", verified=True)
        )
    return profiles, articles


def linked_profiles(urls: list[str], name: str, found_on: str) -> list[ClientSource]:
    """Profile links from a page that is already this person's. Posts and articles are not included."""
    return classify_links(urls, name, found_on)[0]


def _readable(sources: list[ClientSource]) -> list[ClientSource]:
    """Pages a bot can open. Facebook, Instagram, LinkedIn, TikTok and X refuse."""
    out = []
    seen: set[str] = set()
    for s in sources:
        if not s.url.startswith("http") or not can_fetch(s.url) or link_key(s.url) in seen:
            continue
        seen.add(link_key(s.url))
        out.append(s)
    rank = {"wikipedia": 0, "website": 1, "github": 2, "youtube": 3, "academic": 4}
    out.sort(key=lambda s: rank.get(s.type, 5))
    return out[:_READ_PAGES]


class DeepSearchError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


_MARKER_SKIP = {
    "the", "and", "or", "for", "with", "from", "this", "that", "http", "https",
    "facebook", "instagram", "tiktok", "linkedin", "youtube", "twitter", "github",
    "wikipedia", "profile", "official",
}


def markers(c: ClientCandidate) -> list[str]:
    """Handles, employer, school and other tokens that identify THIS card, not a namesake."""
    raw: list[str] = []
    for s in c.sources:
        if h := account_handle(s.url):
            raw.append(h.lstrip("@"))
        if p := profile_of(s.url):
            raw.append(p.handle.split("/")[-1])
    for f in c.keyFacts:
        if f.value:
            raw.append(f.value)
    for v in (c.occupation, c.headline, c.knownFor, c.location, c.disambiguationNote, c.imageSearchQuery, *(c.aliases or [])):
        if v:
            raw.append(v)
    skip = set(name_tokens(c.displayName)) | _MARKER_SKIP
    out: list[str] = []
    for item in raw:
        for tok in name_tokens(item):
            if len(tok) >= 4 and tok not in skip:
                out.append(tok)
    return list(dict.fromkeys(out))


def fits_selected(c: ClientCandidate, title: str, snippet: str = "", url: str = "") -> bool:
    """A new page is about the chosen card, not someone else with the same name."""
    found = markers(c)
    blob = fold(f"{title} {snippet} {url}")
    if not found:
        return name_match(c.displayName, f"{title} {snippet}") >= 0.6
    return any(fold(m) in blob for m in found)


def article_query(c: ClientCandidate) -> str:
    """Pages about the chosen card, not every namesake. Not the big social networks."""
    extra = ""
    if c.imageSearchQuery and fold(c.imageSearchQuery) != fold(c.displayName):
        extra = c.imageSearchQuery
    if not extra:
        marks = [m for m in markers(c) if len(m) >= 5][:2]
        extra = " ".join(p for p in (c.occupation or c.headline, *marks) if p)
    base = " ".join(p for p in (c.displayName, extra) if p)
    return f"{base} -site:facebook.com -site:instagram.com -site:tiktok.com -site:linkedin.com -site:pinterest.com"


def profiles_copied_from_text(urls: list[str], name: str, page_text: dict[str, str]) -> list[ClientSource]:
    """Social URLs the model copied out of the person's own pages. Kept when
    that text contains the URL. The handle is not checked again."""
    blob = "\n".join(page_text.values())
    written = [raw.strip().rstrip(".,;)") for raw in urls]
    kept = [url for url in written if url.startswith("http") and (url in blob or url.rstrip("/") in blob)]
    return classify_links(kept, name, "profile description")[0]


def _known_url(url: str, known: dict[str, ClientSource]) -> ClientSource | None:
    return known.get(link_key(url))


def _title_names(title: str, name: str) -> bool:
    return name_from_title(title, name) is not None or name_match(name, title) >= 0.8


def is_public_figure(name: str, sources: list[ClientSource]) -> bool:
    """A Wikipedia article titled with this person. That is the signal for a name-based image search."""
    return any(platform_of_url(s.url) == "wikipedia" and _title_names(s.title or "", name) for s in sources)


def allows_name_image_search(has_profile_photo: bool, famous: bool) -> bool:
    """Serper image search until the gallery has a portrait that meets the face standard.
    A logo or a weak match does not count. `famous` is unused; a name search is still
    face-checked, so a result that does not match is not shown."""
    return not has_profile_photo


def _crawlable(source: ClientSource, name: str, trusted: set[str]) -> bool:
    """Crawl a walled profile on the chosen card, or one whose handle is this person.
    Display names on Facebook and Instagram often differ; a source the user already
    selected is still this person."""
    if crawl_target(source.url) is None:
        return False
    if link_key(source.url) in trusted:
        return True
    return certain_profile(source.url, name) or _title_names(source.title or "", name)


def split_sources(
    name: str,
    existing: list[ClientSource],
    extra: list[ClientSource],
    social_urls: list[str],
    article_urls: list[str],
    trusted: set[str] | None = None,
    on_profile: set[str] | None = None,
) -> tuple[list[ClientSource], list[ClientSource]]:
    """Social accounts and personal profile pages in one list; news and other
    articles in the other. Sources the user already had (trusted) stay, including
    Wikipedia. Links written on the person's own profile are only split. A link
    from anywhere else still has to be this person."""
    trusted = trusted or set()
    on_profile = on_profile or set()
    known: dict[str, ClientSource] = {}
    for s in [*existing, *extra]:
        known.setdefault(link_key(s.url), s)

    social: list[ClientSource] = []
    articles: list[ClientSource] = []
    seen: set[str] = set()

    def place(s: ClientSource, as_article: bool) -> None:
        key = link_key(s.url)
        if key in seen:
            return
        account = profile_of(s.url)
        wiki = platform_of_url(s.url) == "wikipedia"
        news = is_news(s.url)
        original = key in trusted
        if account and not original:
            on_their_page = key in on_profile
            titled = key in trusted and _title_names(s.title or "", name)
            if not on_their_page and looks_like_fan_account(account.url, "" if not titled else s.title or ""):
                seen.add(key)
                return
            if not (on_their_page or certain_profile(account.url, name) or titled):
                seen.add(key)
                return
        # Wikipedia editions are collected here; one_wikipedia keeps English if present.
        if wiki and (original or is_profile_page(s.url, s.title or "", name)):
            seen.add(key)
            social.append(s.model_copy(update={"type": "wikipedia"}))
            return
        if (is_profile_page(s.url, s.title, name) or original) and not news:
            seen.add(key)
            social.append(s)
            return
        if account:
            seen.add(key)
            social.append(s)
            return
        if key in on_profile or ((as_article or news) and len(articles) < _MAX_ARTICLES):
            seen.add(key)
            articles.append(s.model_copy(update={"type": "web"}))

    for s in existing:
        place(s, as_article=is_news(s.url))
    for url in social_urls:
        if src := _known_url(url, known):
            place(src, as_article=False)
    for url in article_urls:
        if src := _known_url(url, known):
            place(src, as_article=True)
    social = one_wikipedia(sort_social_first(social))
    social.sort(key=lambda s: s.type != "wikipedia")
    articles = [s for s in articles if not is_wikipedia(s.url, s.type)]
    return social, articles[:_MAX_ARTICLES]


def apply_dossier(
    c: ClientCandidate,
    out: DeepProfileOutput,
    social: list[ClientSource],
    articles: list[ClientSource],
    page_text: dict[str, str] | None = None,
) -> ClientCandidate:
    """Search-verified snippet facts stay. The model fills the gaps.
    A birth year, city, school or nationality is kept when a page we read states it."""
    pack = c.model_copy(update={"sources": social + articles})
    read = _read_blob(page_text or {}, pack.sources)
    facts = list(c.keyFacts)
    have = {f.key for f in facts}
    for f in out.facts:
        key, value = _fact_key(f.key), " ".join(f.value.split())[:240]
        if not key or not value or key in have or len(facts) >= _MAX_FACTS:
            continue
        if key == "born" and not (_born_supported(pack, value) or _year_stated(value, read)):
            continue
        if key == "audience":
            continue
        have.add(key)
        facts.append(ClientKeyFact(key=key, label=_LABELS.get(key) or " ".join(f.label.split())[:60] or key, value=value, verified=False))
    for stated in facts_from_pages(page_text or {}):
        if stated.key in have or len(facts) >= _MAX_FACTS:
            continue
        have.add(stated.key)
        facts.append(stated)
    if social and "socialProfiles" not in have and (text := social_profiles_text(social)):
        facts.append(ClientKeyFact(key="socialProfiles", label="Social profiles", value=text, verified=True))
    if "audience" not in have and (aud := audience_fact(social)):
        facts.append(aud)

    profile = [" ".join(p.split()) for p in out.profile if p.strip()][:_MAX_PARAGRAPHS]
    career = [
        ClientCareer(period=(x.period or "").strip() or None, title=(x.title or "").strip() or None, organization=(x.organization or "").strip() or None)
        for x in out.career
        if (x.title or "").strip() or (x.organization or "").strip()
    ][:8]
    headline = " ".join((out.headline or "").split())[:160]
    by_key = {f.key: f.value for f in facts}
    platforms = [s.type for s in social if s.type not in ("website", "wikipedia", "web", "academic")]
    return c.model_copy(
        update={
            "headline": headline or c.headline,
            "occupation": by_key.get("occupation") or c.occupation,
            "knownFor": by_key.get("knownFor") or c.knownFor,
            "nationality": by_key.get("nationality") or c.nationality,
            "location": by_key.get("location") or c.location,
            "platforms": list(dict.fromkeys(platforms)),
            "sources": social + articles,
            "keyFacts": facts,
            "facts": onesearch_facts(facts, social),
            "profile": profile or c.profile,
            "profileSource": "llm" if profile else c.profileSource,
            "career": career or c.career,
            "tags": [t.strip() for t in out.tags if t.strip()][:6] or c.tags,
            "enrichedAt": utcnow().isoformat(),
        }
    )


class DeepSearch:
    def __init__(
        self,
        settings: Settings,
        search: SearchAdapter,
        llm: LLMProvider | None,
        source_photos: SourcePhotoFinder | None,
        faces: FaceVerifier | None,
        fetcher: PageFetcher | None = None,
        crawl: SocialCrawl | None = None,
        sources: SourceLookup | None = None,
    ):
        self.s = settings
        self.search = search
        self.llm = llm
        self.source_photos = source_photos
        self.faces = faces
        self.fetcher = fetcher
        self.crawl = crawl
        self.sources = sources

    async def run(self, candidate: ClientCandidate, filters: dict[str, str]) -> DeepSearchResponse:
        if self.llm is None:
            raise DeepSearchError(503, "LLM provider is not configured")
        with tracing.span(
            "deep-search",
            input={"id": candidate.id, "displayName": candidate.displayName, "filters": filters},
            tags=["deep-search"],
        ):
            try:
                found_calls = 0
                if not any(is_profile_page(s.url, s.title or "", candidate.displayName) for s in candidate.sources):
                    candidate, found_calls = await self._find_profiles(candidate, filters)
                out = await self._run(candidate, filters)
                out.meta.searchCalls += found_calls
                return out
            finally:
                tracing.flush()

    async def _find_profiles(self, candidate: ClientCandidate, filters: dict[str, str]) -> tuple[ClientCandidate, int]:
        """No profile page among the sources (e.g. a photo-search candidate known only
        from articles): look up the person's accounts by name. Without any, the dossier
        is still written from the sources and the article search."""
        if self.sources is None:
            return candidate, 0
        try:
            found = await self.sources.social(
                SocialSourcesRequest(
                    name=candidate.displayName[:120],
                    aliases=candidate.aliases[:10],
                    occupation=(candidate.occupation or "")[:200] or None,
                    headline=(candidate.headline or "")[:200] or None,
                    knownFor=(candidate.knownFor or "")[:200] or None,
                    filters=filters,
                )
            )
        except Exception as e:
            log.warning("profile lookup for %s failed: %s", candidate.displayName, e)
            return candidate, 0
        have = {link_key(s.url) for s in candidate.sources}
        added = [s for s in found.sources if link_key(s.url) not in have]
        return candidate.model_copy(update={"sources": [*added, *candidate.sources]}), found.meta.searchCalls

    async def _run(self, candidate: ClientCandidate, filters: dict[str, str]) -> DeepSearchResponse:
        logs: list[ToolCallLog] = []
        locale = locale_for(filters, (self.s.search_locale_gl, self.s.search_locale_hl), candidate.displayName)
        if self.faces:
            asyncio.create_task(self.faces.ready())
        trusted = {link_key(s.url) for s in candidate.sources}
        crawl_sources = [s for s in candidate.sources if _crawlable(s, candidate.displayName, trusted)]
        pages_t = asyncio.create_task(self._read_pages(candidate.displayName, candidate.sources))
        crawl_t = asyncio.create_task(self.crawl.read(crawl_sources)) if self.crawl else None
        extra_t = asyncio.create_task(self._articles(candidate, logs, locale))
        discovered, page_text, own_text, listed = await pages_t
        crawled = await crawl_t if crawl_t else None
        extra = await extra_t
        crawled = await self._absorb_crawl(crawled, discovered, listed, page_text, own_text, candidate.displayName, trusted)
        more, more_text, more_own, more_listed = await self._read_pages(candidate.displayName, extra, harvest=False)
        have = {link_key(s.url) for s in discovered}
        discovered.extend(s for s in more if link_key(s.url) not in have)
        listed.extend(s for s in more_listed if link_key(s.url) not in {link_key(a.url) for a in listed})
        for key, text in more_text.items():
            page_text.setdefault(key, text)
        for key, text in more_own.items():
            own_text.setdefault(key, text)
        evidence = {
            "selected": {
                "id": candidate.id,
                "displayName": candidate.displayName,
                "aliases": candidate.aliases,
                "headline": candidate.headline,
                "occupation": candidate.occupation,
                "location": candidate.location,
                "knownFor": candidate.knownFor,
                "disambiguationNote": candidate.disambiguationNote,
                "imageSearchQuery": candidate.imageSearchQuery,
                "keyFacts": {f.key: f.value for f in candidate.keyFacts},
                "sources": [{"type": s.type, "url": s.url, "title": s.title} for s in candidate.sources],
                "profile": (candidate.profile or [])[:3],
            },
            "filters": filters,
            "sources": [
                {
                    "type": s.type,
                    "url": s.url,
                    "title": s.title,
                    "snippet": (s.snippet or "")[:_SNIPPET],
                    **({"pageText": page_text[link_key(s.url)]} if link_key(s.url) in page_text else {}),
                }
                for s in [*candidate.sources, *discovered, *extra][:20]
            ],
        }
        dead_t = asyncio.create_task(self._dead_links([*candidate.sources, *discovered, *listed, *extra]))
        try:
            extracted = await self.llm.extract_profile(evidence)
        except Exception as e:
            dead_t.cancel()
            raise DeepSearchError(502, f"profile extraction failed: {str(e)[:200]}") from e

        described = profiles_copied_from_text(extracted.socialUrls, candidate.displayName, own_text)
        dead = await dead_t
        if described:
            dead |= await self._dead_links(described)
        discovered = [s for s in discovered if link_key(s.url) not in dead]
        listed = [s for s in listed if link_key(s.url) not in dead]
        described = [s for s in described if link_key(s.url) not in dead]
        extra = [s for s in extra if link_key(s.url) not in dead]
        on_profile = {link_key(s.url) for s in [*discovered, *listed, *described]}
        social, articles = split_sources(
            candidate.displayName,
            [s for s in candidate.sources if link_key(s.url) not in dead] + [*discovered, *listed, *described],
            extra,
            extracted.socialUrls,
            extracted.articleUrls,
            trusted=trusted,
            on_profile=on_profile,
        )
        kept = {link_key(s.url) for s in [*social, *articles]}
        for s in candidate.sources:
            key = link_key(s.url)
            if key in dead or key in kept:
                continue
            social.append(s)
            kept.add(key)
        social = one_wikipedia(sort_social_first(social))
        social.sort(key=lambda s: s.type != "wikipedia")
        if not social and not articles:
            raise DeepSearchError(422, "no profile source left after checking the dossier")
        updated = apply_dossier(candidate, extracted, social, articles, page_text)
        photos, checked, photo_note, avatar = await self._photos(
            updated, social + articles, logs, locale, list(crawled.photos) if crawled else []
        )
        warnings = [photo_note] if photo_note else []
        if crawled and crawled.errors:
            warnings.extend(crawled.errors)
        if self.crawl and not self.crawl.api_key and any(crawl_target(s.url) for s in candidate.sources):
            warnings.append("Scrape Creators is not configured, so Facebook, Instagram, TikTok, LinkedIn and X were not read.")
        updated = updated.model_copy(update={"photos": photos, "avatarUrl": avatar})
        return DeepSearchResponse(
            candidate=updated,
            socialSources=social,
            articleSources=articles,
            warnings=warnings,
            meta=ClientMeta(
                searchCalls=sum(1 for c in logs if not c.cached and not c.error),
                llmCalls=1,
                crawlfeedCalls=crawled.calls if crawled else 0,
                crawlfeedCredits=crawled.credits if crawled else 0,
                totalSources=len(social) + len(articles),
                **tracing.client_cost(tracing.current_stats()),
            ),
        )

    async def _absorb_crawl(
        self,
        crawled: CrawlBatch | None,
        discovered: list[ClientSource],
        listed: list[ClientSource],
        page_text: dict[str, str],
        own_text: dict[str, str],
        name: str,
        trusted: set[str],
    ) -> CrawlBatch | None:
        """Fold Crawlfeed text into the evidence pack. If a page found another
        walled profile and a slot is still free, read that one too."""
        if crawled:
            self._fold_crawl_text(crawled, discovered, listed, page_text, own_text, name)
        leftover = 2 - (crawled.calls if crawled else 0)
        if not self.crawl or leftover <= 0:
            return crawled
        have = set(crawled.text) if crawled else set()
        extra = [s for s in discovered if _crawlable(s, name, trusted) and link_key(s.url) not in have]
        if not extra:
            return crawled
        more = await self.crawl.read(extra, max_calls=leftover)
        if crawled is None:
            crawled = more
        else:
            crawled.merge(more)
        self._fold_crawl_text(more, discovered, listed, page_text, own_text, name)
        return crawled

    def _fold_crawl_text(
        self,
        crawled: CrawlBatch,
        discovered: list[ClientSource],
        listed: list[ClientSource],
        page_text: dict[str, str],
        own_text: dict[str, str],
        name: str,
    ) -> None:
        for key, text in crawled.text.items():
            page_text.setdefault(key, text)
            own_text.setdefault(key, text)
            profiles, articles = classify_links([*crawled.links, *urls_in(text)], name, key)
            have = {link_key(s.url) for s in [*discovered, *listed]}
            discovered.extend(s for s in profiles if link_key(s.url) not in have)
            listed.extend(s for s in articles if link_key(s.url) not in have)

    async def _dead_links(self, sources: list[ClientSource]) -> set[str]:
        """Links that no longer open. A Crawlfeed 404 is dead. So is HTTP 404, 410, or a host that does not resolve."""
        urls = [s.url for s in sources if s.url.startswith("http")]
        dead: set[str] = set()
        skip = set()
        if self.crawl:
            dead = {link_key(url) for url in urls if self.crawl.known_missing(url)}
            skip = {link_key(url) for url in urls if self.crawl.known_live(url)}
        if self.fetcher is None:
            return dead
        checked = await self.fetcher.dead_links([url for url in urls if link_key(url) not in dead and link_key(url) not in skip])
        if checked:
            log.info("dropped dead links: %s", ", ".join(sorted(checked)))
        return dead | checked

    async def _read_pages(self, name: str, sources: list[ClientSource], harvest: bool = True) -> tuple[list[ClientSource], dict[str, str], dict[str, str], list[ClientSource]]:
        """Open the person's public pages. Links written there are split into
        profiles and articles when harvest is True (the chosen card's own pages).
        Article-search hits are read for text only so a namesake page cannot add
        its social links. The third dict is own-page text."""
        if self.fetcher is None:
            return [], {}, {}, []
        pages = _readable(sources)
        if not pages:
            return [], {}, {}, []
        texts: dict[str, str] = {}
        own_texts: dict[str, str] = {}
        found: list[ClientSource] = []
        listed: list[ClientSource] = []
        seen = {link_key(s.url) for s in sources}

        async def wiki_bits(source: ClientSource) -> tuple[str, list[str]]:
            assert self.fetcher is not None
            page = await self.fetcher.wikipedia_page(source.url)
            en = english_wiki_url(page)
            if en and wiki_target(en):
                source.url = en
                if page.title:
                    source.title = page.title
            blob = "\n".join(f"{label}: {value}" for _, label, value in fact_items(page.facts))
            text = "\n".join(p for p in (blob, page.extract) if p)
            return text, page.links

        async def site_bits(url: str) -> tuple[str, list[str]]:
            assert self.fetcher is not None
            page = await self.fetcher.fetch(url)
            if page.status != "fetched":
                return "", []
            text = excerpt(page.text, name, _PAGE_CHARS) or (page.description or "")[:_PAGE_CHARS]
            return text, page.profile_links

        async def page_bits(source: ClientSource) -> tuple[str, list[str], str | None]:
            kind = platform_of_url(source.url)
            if kind == "wikipedia":
                text, links = await wiki_bits(source)
                return text, links, None
            # The channel description and the GitHub bio are where people write their other accounts.
            if kind == "youtube" and profile_of(source.url):
                text, audience = await youtube_about(source.url)
                if text:
                    return text, [], audience
            if kind == "github" and profile_of(source.url) and self.fetcher is not None:
                text, audience = await github_about(source.url, self.fetcher.timeout_s, self.fetcher.user_agent)
                if text:
                    return text, [], audience
            text, links = await site_bits(source.url)
            return text, links, None

        bits = await asyncio.gather(*(page_bits(s) for s in pages), return_exceptions=True)
        hop: list[str] = []
        for source, result in zip(pages, bits):
            if isinstance(result, BaseException):
                log.warning("reading %s failed: %s", source.url, result)
                continue
            text, links, audience = result
            if audience and audience.lower() not in (source.snippet or "").lower():
                source.snippet = " · ".join(part for part in (source.snippet, audience) if part)
            own_page = harvest and (
                is_profile_page(source.url, source.title or "", name)
                or (platform_of_url(source.url) == "wikipedia" and _title_names(source.title or "", name))
            )
            if text:
                cap = _WIKI_CHARS if platform_of_url(source.url) == "wikipedia" else _PAGE_CHARS
                texts[link_key(source.url)] = text[:cap]
                if own_page:
                    own_texts[link_key(source.url)] = text[:cap]
            if not harvest or is_news(source.url) or not own_page:
                continue
            profiles, articles = classify_links([*links, *urls_in(text)], name, source.url)
            for account in profiles:
                key = link_key(account.url)
                if key in seen:
                    continue
                seen.add(key)
                found.append(account)
            for article in articles:
                key = link_key(article.url)
                if key in seen:
                    continue
                seen.add(key)
                listed.append(article)
            for url in links:
                if profile_of(url) or not can_fetch(url) or link_key(url) in seen:
                    continue
                if official_website_of(url, name, name) and len(hop) < 2:
                    hop.append(url)
                    seen.add(link_key(url))
        if hop:
            hopped = await asyncio.gather(*(site_bits(url) for url in hop), return_exceptions=True)
            for url, result in zip(hop, hopped):
                if isinstance(result, BaseException):
                    continue
                text, links = result
                if text:
                    texts[link_key(url)] = text[:_PAGE_CHARS]
                    own_texts[link_key(url)] = text[:_PAGE_CHARS]
                profiles, articles = classify_links([*links, *urls_in(text)], name, url)
                for account in profiles:
                    key = link_key(account.url)
                    if key in seen:
                        continue
                    seen.add(key)
                    found.append(account)
                for article in articles:
                    key = link_key(article.url)
                    if key in seen:
                        continue
                    seen.add(key)
                    listed.append(article)
        return found, texts, own_texts, listed

    async def _articles(self, c: ClientCandidate, logs: list[ToolCallLog], locale: tuple[str, str]) -> list[ClientSource]:
        have = {link_key(s.url) for s in c.sources}
        try:
            hits, kgs = await self.search.search(article_query(c), "web", logs, locale)
        except RuntimeError as e:
            log.warning("article search failed: %s", e)
            return []
        out: list[ClientSource] = []
        for kg in kgs:
            url = kg.get("descriptionLink") or kg.get("website")
            title = kg.get("title") or ""
            if url and link_key(url) not in have and fits_selected(c, title, kg.get("description") or "", url):
                have.add(link_key(url))
                out.append(ClientSource(url=url, title=title, type="wikipedia" if "wikipedia.org" in url else "web", snippet=kg.get("description"), verified=True))
        for h in hits:
            if link_key(h.url) in have:
                continue
            if not fits_selected(c, h.title, h.snippet or "", h.url):
                continue
            have.add(link_key(h.url))
            kind = "wikipedia" if platform_of_url(h.url) == "wikipedia" else "web"
            out.append(ClientSource(url=h.url, title=h.title, type=kind, snippet=h.snippet or None, verified=True))
            if len(out) >= 12:
                break
        return out

    async def _photos(
        self,
        c: ClientCandidate,
        sources: list[ClientSource],
        logs: list[ToolCallLog],
        locale: tuple[str, str],
        crawled: list[ClientPhoto] | None = None,
    ) -> tuple[list[ClientPhoto], bool, str | None, str | None]:
        own: list[ClientPhoto] = []
        checked = False
        seen: set[str] = set()
        # Social profile pictures first (avatar, then other images on the page), then
        # Wikipedia/GitHub, then photos already on the search-profile card.
        if self.source_photos:
            try:
                page_photos, checked = await self.source_photos.find(
                    c.displayName, sources, limit=PHOTO_MAX, max_pages=8, require_face=False
                )
            except Exception as e:
                log.warning("source photos for %s failed: %s", c.displayName, e)
                page_photos = []
        else:
            page_photos = []
        for photo in [*(crawled or []), *page_photos, *(p for p in c.photos if p.fromSource)]:
            if image_key(photo.url) not in seen:
                seen.add(image_key(photo.url))
                own.append(photo)
        portraits, checked = await self._keep_faces(c, own, [], checked)
        photos, avatar = gallery_and_avatar(portraits, own, c.avatarUrl)
        return photos[:PHOTO_MAX], checked, None, avatar

    async def _keep_faces(self, c: ClientCandidate, own: list[ClientPhoto], found: list[ClientPhoto], source_checked: bool) -> tuple[list[ClientPhoto], bool]:
        check = bool(self.faces and await self.faces.ready())
        if not check:
            # Without a face model, keep source portraits first, then name-matched images.
            merged = _dedupe_photos(own + found)
            return merged[:PHOTO_MAX], source_checked
        pool = own + [p for p in found if image_key(p.url) not in {image_key(o.url) for o in own}]
        if not pool:
            return [], True
        faces, used = await self.faces.faces_or_fallback([p.url for p in pool], [p.fallbackUrl for p in pool])
        pool = [p if u == p.url else p.model_copy(update={"url": u, "fallbackUrl": None}) for p, u in zip(pool, used)]
        trusted = [True] * len(own) + [p.fromSource for p in pool[len(own) :]]
        kept = select_photos(faces, trusted, self.s.face_gallery_threshold)
        photos = [pool[i].model_copy(update={"faceMatch": score}) for i, score in kept]
        photos.sort(key=lambda p: -(p.faceMatch or 0))
        if self.faces:
            photos = [photos[i] for i in await distinct_urls(self.faces, [p.url for p in photos])]
        if photos:
            return photos[:PHOTO_MAX], True
        # No matching face in the gallery. The avatar still comes from the source photo.
        return [], True


def _dedupe_photos(photos: list[ClientPhoto]) -> list[ClientPhoto]:
    seen: set[str] = set()
    out = []
    for p in photos:
        key = image_key(p.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def gallery_and_avatar(
    face_photos: list[ClientPhoto],
    source_photos: list[ClientPhoto],
    previous: str | None,
) -> tuple[list[ClientPhoto], str | None]:
    """Face photos fill the gallery. If none match, the profile avatar still comes from the source."""
    if face_photos:
        return face_photos, face_photos[0].url
    for p in source_photos:
        if p.url.startswith("http"):
            return [], p.url
    if previous and previous.startswith("http"):
        return [], previous
    return [], None
