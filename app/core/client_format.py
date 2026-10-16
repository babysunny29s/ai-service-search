"""Maps internal FeCandidate records to the client people-search contract."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import unquote, urlsplit

from app.core.urls import SOCIAL_ORDER, account_handle, fold, link_key, name_match, platform_of_url
from app.core.models import (
    ClientCandidate,
    ClientCareer,
    ClientKeyFact,
    ClientMeta,
    ClientPhoto,
    ClientSource,
    FeCandidate,
    FieldValue,
    PeopleSearchResponse,
    SearchResult,
    Source,
)

_PLAIN_URL = re.compile(r"^https?://[^\s\[\]<>\"']+$")
_CITATION = re.compile(r"\[\s*src_?\d+(?:\s*[,;]\s*(?:src_?)?\d+)*\s*\]", re.I)
_TYPE_PRIORITY = {"wiki": 0, "official_website": 1, "academic": 3, "code": 4, "directory": 6}


def _plain_url(u: str | None) -> str | None:
    return u if u and _PLAIN_URL.match(u) and "](" not in u else None


def _about_person(url: str, names: list[str]) -> int:
    """How well a Wikipedia article title names the person: 2 = the title is a
    name or alias ("Lisa (rapper)"), 1 = it contains one or is contained in one
    ("Ronaldo (Brazilian footballer)" for "Ronaldo Luís Nazário de Lima", but also
    "Lisa (rapper) discography"), 0 = not about them (category/Commons pages)."""
    path = unquote(urlsplit(url).path)
    if not path.startswith("/wiki/"):
        return 0
    title = path[len("/wiki/") :].replace("_", " ")
    if ":" in title:
        return 0
    base = re.sub(r"\s*\(.*\)\s*$", "", title)
    if any(n and fold(base) == fold(n) for n in names):
        return 2
    return int(any(n and (name_match(n, base) >= 0.99 or name_match(base, n) >= 0.99) for n in names))


def _wiki_lang_rank(url: str) -> int:
    lang = urlsplit(url).netloc.split(".")[0]
    return {"en": 0, "vi": 1}.get(lang, 2)


def person_wiki(c: FeCandidate) -> str | None:
    """The person's own Wikipedia article: best title match, English first, then Vietnamese."""
    ordered = sorted(c.sources, key=lambda s: (not s.knowledgeGraph, -s.rank))
    names = [c.displayName, *c.aliases]
    scored = [(_about_person(s.url, names), s.url) for s in ordered if s.sourceType == "wiki"]
    scored = [(score, u) for score, u in scored if score]
    return min(scored, key=lambda x: (-x[0], _wiki_lang_rank(x[1])))[1] if scored else None


def official_links(c: FeCandidate) -> list[str]:
    """Official websites, then the person's Wikipedia article."""
    ordered = sorted(c.sources, key=lambda s: (not s.knowledgeGraph, _TYPE_PRIORITY.get(s.sourceType, 5), -s.rank))
    official = [s.url for s in ordered if s.sourceType == "official_website"]
    wiki = [w] if (w := person_wiki(c)) else []
    return [u for u in dict.fromkeys(official + wiki) if _plain_url(u)]


def _strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:])", r"\1", " ".join(_CITATION.sub("", text).split()))


def slug_id(name: str, salt: str = "") -> str:
    """Slug + short hash; the salt tells apart people sharing a display name."""
    base = re.sub(r"[^a-z0-9]+", "-", fold(name)).strip("-")[:80]
    digest = hashlib.sha256(f"{name.lower()}\0{salt}".encode()).hexdigest()[:8]
    return f"{base or 'person'}-{digest}"


def ensure_unique_ids(items: list[ClientCandidate]) -> list[ClientCandidate]:
    seen: set[str] = set()
    for item in items:
        new, n = item.id, 2
        while new in seen:
            new, n = f"{item.id}-{n}", n + 1
        item.id = new
        seen.add(new)
    return items


def _platform_rank(t: str) -> int:
    if t in SOCIAL_ORDER:
        return SOCIAL_ORDER.index(t)
    return len(SOCIAL_ORDER) + {"website": 0, "web": 2}.get(t, 1)


def sort_social_first(sources: list[ClientSource]) -> list[ClientSource]:
    """Social networks in SOCIAL_ORDER, then official website, then wikipedia/academic, then plain web."""
    return sorted(sources, key=lambda s: _platform_rank(s.type))


def _platforms(c: FeCandidate) -> list[str]:
    found = {platform_of_url(p.url) for p in c.socialProfiles}
    return [p for p in SOCIAL_ORDER if p in found]


def stable_candidate_id(c: FeCandidate) -> str:
    """The person's own link (Wikipedia, else a social profile) identifies them
    across searches better than LLM-worded descriptions."""
    socials = sorted(c.socialProfiles, key=lambda p: _platform_rank(platform_of_url(p.url)))
    anchor = person_wiki(c) or (socials[0].url if socials else None)
    if anchor:
        return slug_id(c.displayName, link_key(anchor))
    role = re.split(r",|;|/|&|\band\b|\bvà\b", c.occupation or "", maxsplit=1)[0]
    return slug_id(c.displayName, "|".join(fold(x) for x in (role, c.nationality or "") if x.strip()))


def _source_type(s: Source) -> str:
    if s.sourceType == "official_website":
        return "website"
    if s.sourceType == "wiki":
        return "wikipedia"
    if s.sourceType == "academic":
        return "academic"
    p = platform_of_url(s.url)
    return p if p in SOCIAL_ORDER else "web"


def client_sources(c: FeCandidate) -> list[ClientSource]:
    """The sources a client sees for `c` (social profiles, official links, other sources)."""
    by_url = {link_key(s.url): s for s in c.sources}
    out: dict[str, ClientSource] = {}

    def add(url: str, type_: str, src: Source | None) -> None:
        key = link_key(url)
        if key in out or not _plain_url(url):
            return
        src = src or by_url.get(key)
        out[key] = ClientSource(
            url=url,
            title=src.title if src else "",
            type=type_,
            snippet=(src.snippet or None) if src else None,
            # Links without a source of their own (an English article found through a
            # verified article's language links) come from Wikipedia itself.
            verified=src.provider != "llm_knowledge" if src else True,
        )

    for p in c.socialProfiles:
        platform = platform_of_url(p.url)
        src = by_url.get(link_key(p.url)) or next((s for s in c.sources if s.id in p.sourceIds), None)
        add(p.url, platform if platform in SOCIAL_ORDER else "web", src)
    for u in c.officialLinks or official_links(c):
        add(u, "wikipedia" if platform_of_url(u) == "wikipedia" else "website", None)
    for s in sorted(c.sources, key=lambda s: -s.rank):
        if s.sourceType != "social_profile":
            add(s.url, _source_type(s), s)
    return sort_social_first(list(out.values()))


_FACT_LABELS = {
    "fullName": "Full name",
    "born": "Born",
    "jobTitle": "Role",
    "company": "Company",
    "occupation": "Occupation",
    "knownFor": "Known for",
    "nationality": "Nationality",
    "education": "Education",
    "location": "Location",
    "field": "Field",
    "industry": "Industry",
    "specialization": "Specialization",
    "yearsActive": "Years active",
    "awards": "Awards",
    "notableWorks": "Notable works",
    "teams": "Teams",
    "academicInterests": "Academic interests",
    "publisher": "Publisher",
    "bookSeries": "Book series",
    "website": "Website",
    "age": "Age",
    "email": "Email",
    "phone": "Phone",
}
# Our key -> onesearch FactsTable key, where they differ.
_ONESEARCH_KEYS = {"jobTitle": "primaryRole"}
_PLATFORM_NAMES = {
    "facebook": "Facebook",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "x": "X",
    "tiktok": "TikTok",
    "threads": "Threads",
    "pinterest": "Pinterest",
    "reddit": "Reddit",
    "youtube": "YouTube",
}
_AUDIENCE_PLATFORMS = ("instagram", "tiktok", "x", "youtube", "facebook", "pinterest")
_AUDIENCE = re.compile(r"(\d[\d.,]*\s*[KMB]?)\s+(Followers|Subscribers|người theo dõi|người đăng ký)", re.I)


def _social_labels(sources: list[ClientSource]) -> list[tuple[str, str, str]]:
    """(label, url, platform name) for each social account, first occurrence wins."""
    out = []
    seen: set[str] = set()
    for s in sources:
        name = _PLATFORM_NAMES.get(s.type)
        if not name or s.url in seen:
            continue
        seen.add(s.url)
        handle = account_handle(s.url)
        out.append((f"{name} · {handle}" if handle else name, s.url, name))
    return out


def social_profiles_text(sources: list[ClientSource]) -> str | None:
    """"Instagram · @handle, YouTube · @handle" for the person's social accounts."""
    return ", ".join(label for label, _, _ in _social_labels(sources)) or None


def social_profiles_markdown(sources: list[ClientSource]) -> str | None:
    """"[Instagram · @handle](url)[Facebook · @name](url)" — links concatenated, as on the profile card."""
    parts = [f"[{label}]({url})" for label, url, _ in _social_labels(sources)]
    return "".join(parts) or None


def audience_fact(sources: list[ClientSource]) -> ClientKeyFact | None:
    """Follower/subscriber counts the networks show in their search snippets (read as written)."""
    parts = []
    for s in sources:
        if s.type not in _AUDIENCE_PLATFORMS:
            continue
        if m := _AUDIENCE.search(f"{s.title} {s.snippet or ''}"):
            kind = "subscribers" if re.search(r"subscri|đăng ký", m.group(2), re.I) else "followers"
            count = re.sub(r"\s+", "", m.group(1))
            parts.append(f"{_PLATFORM_NAMES[s.type]}: {count} {kind}")
    return ClientKeyFact(key="audience", label="Audience", value=" · ".join(dict.fromkeys(parts)), verified=True) if parts else None


def onesearch_facts(key_facts: list[ClientKeyFact], sources: list[ClientSource]) -> dict[str, str]:
    """keyFacts as the onesearch `facts` record (FactsTable keys), verified values first."""
    out: dict[str, str] = {}
    for f in sorted(key_facts, key=lambda f: not f.verified):
        out.setdefault(_ONESEARCH_KEYS.get(f.key, f.key), f.value)
    if social := social_profiles_text(sources):
        out.setdefault("socialProfiles", social)
    return out


def _field_text(f: FieldValue) -> str | None:
    text = f.value or (", ".join(f.values) if f.values else None)
    return _strip_citations(text) if text else None


def _key_facts(c: FeCandidate) -> list[ClientKeyFact]:
    facts: list[ClientKeyFact] = []
    for key, label in _FACT_LABELS.items():
        f = c.fields.get(key)
        if f and f.status == "found" and (text := _field_text(f)):
            facts.append(ClientKeyFact(key=key, label=label, value=text, verified=True))
    have = {f.key for f in facts}
    for key, value in (("occupation", c.occupation), ("nationality", c.nationality), ("knownFor", c.knownFor)):
        if key not in have and value:
            facts.append(ClientKeyFact(key=key, label=_FACT_LABELS[key], value=value, verified=False))
    return facts


def to_client_candidate(c: FeCandidate) -> ClientCandidate:
    headline = c.headline if c.headline and c.headline.strip().lower() != c.displayName.strip().lower() else None
    loc = c.fields.get("location")
    profile = [p for p in (_strip_citations(x) for x in c.narrative) if p]
    sources = client_sources(c)
    key_facts = _key_facts(c)
    if audience := audience_fact(sources):
        key_facts.append(audience)
    return ClientCandidate(
        id=stable_candidate_id(c),
        displayName=c.displayName,
        aliases=c.aliases,
        headline=headline,
        confidence=round(max(0.0, min(1.0, c.confidenceScore)), 2),
        disambiguationNote=c.disambiguationNote or None,
        nationality=c.nationality,
        occupation=c.occupation,
        knownFor=c.knownFor,
        location=_field_text(loc) if loc and loc.status == "found" else None,
        platforms=_platforms(c),
        sources=sources,
        keyFacts=key_facts,
        facts=onesearch_facts(key_facts, sources),
        photos=[
            ClientPhoto(url=p.url, pageUrl=_plain_url(p.pageUrl), title=p.title, fromSource=bool(p.sourceIds), faceMatch=p.faceMatch, fallbackUrl=p.fallbackUrl)
            for p in c.photos
            if _plain_url(p.url)
        ],
        avatarUrl=_plain_url(c.avatarHint),
        imageSearchQuery=c.imageSearchQuery or c.displayName,
        profile=profile,
        profileSource=c.narrativeSource if profile else None,
        career=[
            ClientCareer(period=x.period, title=x.title, organization=x.organization) for x in c.careerArc if x.title or x.organization
        ],
        tags=c.tags,
        queryFaceMatch=c.queryFaceMatch,
    )


def to_client_response(result: SearchResult) -> PeopleSearchResponse:
    cache_hit = bool(result.stats.get("cacheHit"))
    stats = result.stats or {}
    return PeopleSearchResponse(
        mode=result.mode,
        searchId=result.searchId,
        status=result.status,
        clarificationQuestion=result.clarificationQuestion,
        candidates=ensure_unique_ids([cc for cc in (to_client_candidate(c) for c in result.candidates) if cc.sources]),
        meta=ClientMeta(
            searchCalls=0 if cache_hit else stats.get("searchCalls", 0),
            llmCalls=0 if cache_hit else stats.get("llmCalls", 0),
            promptTokens=0 if cache_hit else stats.get("promptTokens", 0),
            completionTokens=0 if cache_hit else stats.get("completionTokens", 0),
            costUsd=0 if cache_hit else stats.get("costUsd", 0),
            costVnd=0 if cache_hit else stats.get("costVnd", 0),
            costLines=[] if cache_hit else stats.get("costLines") or [],
            cacheHit=cache_hit,
            totalSources=len(result.sources),
        ),
    )
