"""Fact validation, entity resolution and explainable candidate scoring."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.infra.fetcher import FetchedPage, named_images
from app.core.llm import EXTRACT_FIELDS, AnalysisOutput, AnalyzedCandidate, RecalledCandidate
from app.core.models import (
    CareerItem,
    Fact,
    FeCandidate,
    FieldValue,
    Photo,
    ScoreBreakdown,
    SEED_FIELDS,
    SocialProfile,
    Source,
    utcnow,
)
from app.search.ranking import token_overlap
from app.infra.search_adapter import SearchHit
from app.core.urls import SOURCE_QUALITY, classify_source, fold, host_of, is_news, looks_like_fan_account, name_match, normalize_url, profile_root, social_platform, spacing_variant

CORE_FIELDS = list(EXTRACT_FIELDS)
IDENTIFYING_FIELDS = ["jobTitle", "company", "occupation", "knownFor", "education", "location", "field"]
# A person can hold several of these at once (singer and actress), so differing
# values are combined instead of being treated as a conflict.
MULTI_VALUED_FIELDS = {
    "occupation",
    "knownFor",
    "field",
    "industry",
    "specialization",
    "education",
    "awards",
    "notableWorks",
    "teams",
    "academicInterests",
    "publisher",
    "bookSeries",
}
# A birth date is public only where the person or an encyclopedia publishes it.
BORN_SOURCE_TYPES = ("wiki", "official_website")
# Older cached analyses used these names.
_LEGACY_FIELDS = {"award": "awards", "publication": "notableWorks", "skill": "specialization"}
FORBIDDEN_FIELD_PATTERN = re.compile(r"birth|address|id_?card|religion|health|politic", re.I)
WEAK_NAME_CAP = 0.49
_URLISH = re.compile(r"https?://\S+|\[|\]|\(|\)|[*_`#]")


@dataclass
class Mention:
    id: str
    source: Source
    label: str
    name_as_written: str
    page_type: str
    facts: list[Fact] = field(default_factory=list)
    career: list[CareerItem] = field(default_factory=list)
    profile_links: list[str] = field(default_factory=list)


def _norm_ws(s: str) -> str:
    return " ".join(fold(s).split())


def source_text(src: Source, page: FetchedPage | None) -> str:
    parts = [src.title, src.snippet]
    if page and page.status == "fetched":
        parts += [page.title, page.description, page.text]
    return "\n".join(p for p in parts if p)


def _validated_fact(fname: str, value: str, evidence: str, src: Source, text_folded: str, base: float, key: str) -> Fact | None:
    fname, value = fname.strip(), value.strip()
    fname = _LEGACY_FIELDS.get(fname, fname)
    if fname not in EXTRACT_FIELDS or not value or FORBIDDEN_FIELD_PATTERN.search(fname):
        return None
    if fname == "born" and not (src.knowledgeGraph or src.sourceType in BORN_SOURCE_TYPES):
        return None
    ev_ok = bool(evidence) and _norm_ws(evidence) in text_folded
    val_ok = _norm_ws(value) in text_folded or token_overlap(value, text_folded) >= 0.8
    if not ev_ok and not val_ok:
        return None
    return Fact(
        field=fname,
        value=value[:300],
        sourceIds=[src.id],
        evidence=[evidence[:300]] if evidence else [],
        confidence=round(base * (1.0 if ev_ok else 0.6), 3),
        observedAt=utcnow(),
        personKey=key,
    )


@dataclass
class AnalyzedGroup:
    mentions: list[Mention]
    profile: AnalyzedCandidate


def groups_from_analysis(
    analysis: AnalysisOutput,
    sources: dict[str, Source],
    pages: dict[str, FetchedPage],
    query_name: str,
) -> list[AnalyzedGroup]:
    """Turn LLM-proposed people into verified mention groups. Facts whose
    evidence is not in the cited source are dropped."""
    groups: list[AnalyzedGroup] = []
    folded: dict[str, str] = {}
    for ac in analysis.candidates:
        if name_match(query_name, ac.nameAsWritten) < 0.8:
            continue
        own = set(ac.ownSiteSourceIds)
        cited = ac.sourceIds + [f.sourceId for f in ac.facts] + [c.sourceId for c in ac.career]
        key = f"p{len(groups) + 1}"
        mentions: dict[str, Mention] = {}
        for sid in dict.fromkeys(cited):
            src = sources.get(sid)
            if not src:
                continue
            page = pages.get(src.url)
            if sid in own and src.sourceType in ("other", "directory"):
                src.sourceType = "official_website"
            mentions[sid] = Mention(
                id=f"{key}:{sid}",
                source=src,
                label=(ac.headline or ac.nameAsWritten)[:120],
                name_as_written=ac.nameAsWritten.strip(),
                page_type="personal_website" if sid in own else src.sourceType,
                profile_links=list(page.profile_links) if page and sid in own else [],
            )
            folded[sid] = folded.get(sid) or _norm_ws(source_text(src, page))
        for f in ac.facts:
            m = mentions.get(f.sourceId)
            if m:
                base = SOURCE_QUALITY[m.source.sourceType] * (1.0 if m.source.contentStatus == "fetched" else 0.8)
                if fact := _validated_fact(f.field, f.value, f.evidence, m.source, folded[f.sourceId], base, key):
                    m.facts.append(fact)
        for c in ac.career:
            m = mentions.get(c.sourceId)
            t = folded.get(c.sourceId, "")
            if m and c.evidence and (_norm_ws(c.evidence) in t or (c.organization and _norm_ws(c.organization) in t)):
                m.career.append(CareerItem(title=c.title, organization=c.organization, period=c.period, sourceIds=[c.sourceId]))
        if mentions:
            groups.append(AnalyzedGroup(list(mentions.values()), ac))
    return _merge_cross_linked(groups)


def _merge_cross_linked(groups: list[AnalyzedGroup]) -> list[AnalyzedGroup]:
    """A person's own site linking to a profile page held by another group is
    strong evidence they are the same person."""
    idx = [str(i) for i in range(len(groups))]
    uf = _UF(idx)
    owner: dict[str, list[str]] = defaultdict(list)
    for i, g in enumerate(groups):
        for m in g.mentions:
            owner[m.source.url].append(str(i))
    for i, g in enumerate(groups):
        for m in g.mentions:
            for link in m.profile_links:
                targets = [t for t in owner.get(link, []) if t != str(i)]
                if len(targets) == 1:
                    uf.union(str(i), targets[0])
    def richness(p: AnalyzedCandidate) -> tuple[int, int]:
        return (len(p.facts) + len(p.career), sum(len(x) for x in p.profile))

    merged: dict[str, AnalyzedGroup] = {}
    for i, g in enumerate(groups):
        root = uf.find(str(i))
        if root not in merged:
            merged[root] = AnalyzedGroup(list(g.mentions), g.profile)
            continue
        target = merged[root]
        seen = {m.source.id for m in target.mentions}
        target.mentions += [m for m in g.mentions if m.source.id not in seen]
        if richness(g.profile) > richness(target.profile):
            target.profile = g.profile
    return list(merged.values())


class _UF:
    def __init__(self, items: list[str]):
        self.p = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ---------- verification & scoring ----------


def _same_value(a: str, b: str) -> bool:
    fa, fb = _norm_ws(a), _norm_ws(b)
    return fa == fb or fa in fb or fb in fa


def _cluster_values(facts: list[Fact]) -> list[list[Fact]]:
    clusters: list[list[Fact]] = []
    for f in facts:
        for c in clusters:
            if _same_value(c[0].value, f.value):
                c.append(f)
                break
        else:
            clusters.append([f])
    return clusters


def _support(cluster: list[Fact]) -> int:
    return len({sid for f in cluster for sid in f.sourceIds})


def _field_value(cluster: list[Fact]) -> tuple[str, list[str], list[str], float]:
    value = Counter(f.value for f in cluster).most_common(1)[0][0]
    sids = list(dict.fromkeys(s for f in cluster for s in f.sourceIds))
    evidence = list(dict.fromkeys(e for f in cluster for e in f.evidence))[:3]
    n = len(sids)
    conf = min(0.98, max(f.confidence for f in cluster) + 0.08 * (n - 1))
    return value, sids, evidence, round(conf, 3)


def verify_fields(facts: list[Fact], sources: list[Source]) -> tuple[dict[str, FieldValue], int]:
    by_field: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        by_field[f.field].append(f)
    any_fetched = any(s.contentStatus == "fetched" for s in sources)
    any_blocked = any(s.contentStatus == "blocked" for s in sources)
    out: dict[str, FieldValue] = {}
    conflicts = 0
    for name in CORE_FIELDS:
        fs = by_field.get(name, [])
        if not fs:
            status = "not_found" if any_fetched else ("restricted" if any_blocked else "not_checked")
            out[name] = FieldValue(status=status)  # type: ignore[arg-type]
            continue
        clusters = sorted(_cluster_values(fs), key=_support, reverse=True)
        top = clusters[0]
        if name in MULTI_VALUED_FIELDS and len(clusters) > 1:
            parts = [_field_value(c) for c in clusters[:3]]
            out[name] = FieldValue(
                value=", ".join(p[0] for p in parts),
                values=[p[0] for p in parts],
                sourceIds=list(dict.fromkeys(s for p in parts for s in p[1])),
                evidence=[e for p in parts for e in p[2][:1]],
                confidence=max(p[3] for p in parts),
                status="found",
            )
        elif len(clusters) == 1 or _support(top) >= 2 * _support(clusters[1]):
            value, sids, ev, conf = _field_value(top)
            out[name] = FieldValue(value=value, sourceIds=sids, evidence=ev, confidence=conf, status="found")
        else:
            conflicts += 1
            vals, sids, evs = [], [], []
            for c in clusters[:4]:
                v, s, e, _ = _field_value(c)
                vals.append(v)
                sids += s
                evs += e[:1]
            out[name] = FieldValue(values=vals, sourceIds=list(dict.fromkeys(sids)), evidence=evs, confidence=0.0, status="conflicting")
    return out, conflicts


@dataclass
class CandidateDraft:
    candidate: FeCandidate
    facts: list[Fact]
    careers: list[CareerItem]


def candidate_seeds(cand: FeCandidate) -> dict[str, str]:
    """Discovered attributes usable as anchors for follow-up queries."""
    return {k: v.value for k in SEED_FIELDS if (v := cand.fields.get(k)) and v.status == "found" and v.value}


def build_candidate(
    idx: int,
    name: str,
    group: list[Mention],
    pages: dict[str, FetchedPage],
    image_hits: list[tuple[str, list[SearchHit]]],
    filters: dict[str, str] | None = None,
) -> CandidateDraft:
    sources = list({m.source.id: m.source for m in group}.values())
    facts = [f for m in group for f in m.facts]
    careers = [c for m in group for c in m.career]
    fields, conflicts = verify_fields(facts, sources)

    nm = max((name_match(name, m.name_as_written) for m in group), default=0.0)
    identifying = [k for k in IDENTIFYING_FIELDS if fields[k].status == "found"]
    attr = min(1.0, len(identifying) / 3)
    qualities = sorted((SOURCE_QUALITY[s.sourceType] * (1.0 if s.contentStatus == "fetched" else 0.8) for s in sources), reverse=True)
    quality = sum(qualities[:3]) / min(3, len(qualities)) if qualities else 0.0
    hosts = {host_of(s.url) for s in sources}
    multi_supported = sum(1 for fv in fields.values() if fv.status == "found" and len(fv.sourceIds) >= 2)
    agreement = min(1.0, (len(hosts) - 1) / 3 + 0.15 * multi_supported)
    penalty = min(0.6, 0.1 * conflicts)
    news_hosts = {host_of(s.url) for s in sources if s.sourceType == "news"}
    if any(s.knowledgeGraph for s in sources):
        prominence = 1.0
    elif any(s.sourceType == "wiki" for s in sources):
        prominence = 0.75
    else:
        prominence = min(0.5, 0.15 * len(news_hosts))
    score = 0.25 * nm + 0.25 * attr + 0.15 * quality + 0.15 * agreement + 0.2 * prominence - penalty
    filter_match = 0.0
    filters = filters or {}
    if filters:
        filter_match = _filter_match(fields, filters, careers)
        score += 0.1 * filter_match - (0.2 if filter_match == 0 else 0.0)
    display = _display_name(name, group)
    score = max(0.0, min(1.0, score))
    if not _strong_name_match(name, display):
        score = min(score, WEAK_NAME_CAP)

    socials: dict[str, SocialProfile] = {}
    for s in sources:
        p = social_platform(s.url)
        root = profile_root(s.url) if p else None
        if root and looks_like_fan_account(root, s.title):
            continue
        if root and root not in socials:
            socials[root] = SocialProfile(platform=p, url=root, sourceIds=[s.id], confidence=round(min(0.95, score + 0.05), 3))
        elif root:
            socials[root].sourceIds.append(s.id)
    for m in group:
        for link in m.profile_links:
            if link not in socials and not looks_like_fan_account(link):
                socials[link] = SocialProfile(platform=social_platform(link) or "other", url=link, sourceIds=[m.source.id], confidence=round(min(0.9, score), 3))

    photos: list[Photo] = []
    for m in group:
        page = pages.get(m.source.url)
        if not page:
            continue
        for url in named_images(page, name, display):
            photos.append(Photo(url=url, pageUrl=m.source.url, title=page.title, sourceIds=[m.source.id]))
        if page.image and m.page_type in ("personal_website", "social_profile", "wiki"):
            photos.append(Photo(url=page.image, pageUrl=m.source.url, title=page.title, sourceIds=[m.source.id]))
    for s in sources:
        if s.knowledgeGraph and s.image:
            photos.insert(0, Photo(url=s.image, pageUrl=s.url, title=s.title, sourceIds=[s.id], note="Image from the search engine knowledge graph."))
    src_by_url = {s.url: s for s in sources}
    anchors = [fields[k].value for k in SEED_FIELDS if fields[k].status == "found" and fields[k].value]
    for anchor, hits in image_hits:
        anchored = any(_same_value(anchor, a) for a in anchors)
        for h in hits:
            if not h.image_url:
                continue
            page_url = normalize_url(h.url)
            if page_url in src_by_url:
                photos.append(Photo(url=h.image_url, pageUrl=page_url, title=h.title, sourceIds=[src_by_url[page_url].id], fallbackUrl=h.image_fallback))
            elif anchored and name_match(name, h.title) >= 0.8:
                photos.append(
                    Photo(url=h.image_url, pageUrl=page_url, title=h.title, fallbackUrl=h.image_fallback, note="Suggested image from image search by name and found attributes; identity not confirmed.")
                )
    photos = list({p.url: p for p in photos}.values())[:6]

    note_parts = []
    if identifying:
        note_parts.append("Identified by: " + " · ".join(dict.fromkeys(fields[k].value or "" for k in identifying)))
    else:
        note_parts.append("Name match only, no distinguishing information")
    note_parts.append(f"{len(sources)} sources from {len(hosts)} domains")
    if filters:
        note_parts.append(f"matches {round(filter_match * len(filters))}/{len(filters)} filters")
    if conflicts:
        note_parts.append(f"{conflicts} fields with conflicting sources")
    headline_bits = [fields["jobTitle"].value, fields["company"].value]
    headline = " at ".join(b for b in headline_bits if b) or (group[0].label if group else None)

    cand = FeCandidate(
        candidateId=f"cand_{idx}",
        displayName=display,
        headline=headline,
        confidenceScore=round(score, 3),
        scoreBreakdown=ScoreBreakdown(
            nameMatch=round(nm, 3),
            distinguishingAttributes=round(attr, 3),
            sourceQuality=round(quality, 3),
            crossSourceAgreement=round(agreement, 3),
            prominence=round(prominence, 3),
            filterMatch=round(filter_match, 3),
            conflictPenalty=round(penalty, 3),
        ),
        disambiguationNote=_fallback_note(fields),
        matchNote="; ".join(note_parts) + ".",
        nationality=fields["nationality"].value,
        occupation=fields["occupation"].value or fields["jobTitle"].value,
        knownFor=fields["knownFor"].value,
        avatarHint=_avatar(photos),
        fields=fields,
        socialProfiles=sorted(socials.values(), key=lambda s: s.confidence, reverse=True),
        sources=sources,
        photos=photos,
    )
    cand.imageSearchQuery = clean_image_query(None, cand)
    return CandidateDraft(cand, facts, careers)


RECALL_CONFIDENCE_CAP = 0.7
KNOWN_CONFIDENCE_CAP = 0.95
RECALL_NOTE = "From model knowledge; search found no matching sources, so it is unverified."
KNOWN_NOTE = "From model knowledge; the model is confident enough that no search was needed."
UNCONFIRMED_NOTE = "From model knowledge; search results did not confirm this person."


def candidate_from_recall(
    idx: int,
    query_name: str,
    rc: RecalledCandidate,
    start_index: int,
    cap: float = RECALL_CONFIDENCE_CAP,
    note: str = RECALL_NOTE,
) -> tuple[CandidateDraft, list[Source]]:
    """Candidate from the model's own knowledge (no search evidence). Confidence
    is capped and fields stay `not_checked`. Recalled URLs are unverified
    (`provider="llm_knowledge"`) until `verify_known_links` matches them."""
    now = utcnow()
    sources: list[Source] = []
    socials: dict[str, SocialProfile] = {}
    for raw in rc.sourceUrls:
        if not isinstance(raw, str) or not re.match(r"^https?://[^\s\[\]<>\"']+$", raw) or "](" in raw:
            continue
        url = normalize_url(raw)
        if is_news(url) or any(s.url == url for s in sources):
            continue
        src = Source(
            id=f"src_{start_index + len(sources)}",
            title=rc.displayName,
            url=url,
            sourceType=classify_source(url),  # type: ignore[arg-type]
            provider="llm_knowledge",
            query="",
            retrievedAt=now,
        )
        sources.append(src)
        root = profile_root(url) if social_platform(url) else None
        if root and root not in socials and not looks_like_fan_account(root):
            socials[root] = SocialProfile(platform=social_platform(root) or "other", url=root, sourceIds=[src.id], confidence=0.5)

    display, aliases = split_aliases(rc.displayName, rc.aliases)
    rc = rc.model_copy(update={"displayName": display})
    confidence = max(0.0, min(cap, rc.confidence))
    if not _strong_name_match(query_name, rc.displayName):
        confidence = min(confidence, WEAK_NAME_CAP)
    fields = {k: FieldValue(status="not_checked") for k in CORE_FIELDS}
    cand = FeCandidate(
        candidateId=f"cand_{idx}",
        displayName=rc.displayName,
        headline=_plain(rc.headline, 120),
        confidenceScore=round(confidence, 3),
        scoreBreakdown=ScoreBreakdown(
            nameMatch=1.0 if _strong_name_match(query_name, rc.displayName) else 0.5,
            distinguishingAttributes=0.0,
            sourceQuality=0.0,
            crossSourceAgreement=0.0,
            prominence=0.0,
            filterMatch=0.0,
            conflictPenalty=0.0,
        ),
        disambiguationNote=_plain(rc.disambiguationNote, 300) or "",
        matchNote=note,
        nationality=_plain(rc.nationality, 60),
        occupation=_plain(rc.occupation, 80),
        knownFor=_plain(rc.knownFor, 120),
        fields=fields,
        socialProfiles=list(socials.values()),
        sources=sources,
        aliases=aliases,
    )
    cand.imageSearchQuery = clean_image_query(rc.imageSearchQuery, cand)
    if paragraphs := _paragraphs(rc.profile):
        cand.narrative, cand.narrativeSource = paragraphs, "llm"
    return CandidateDraft(cand, [], []), sources


def _fallback_note(fields: dict[str, FieldValue]) -> str:
    """Language-neutral identifier line used until/unless the LLM writes one."""
    keys = ("nationality", "occupation", "jobTitle", "knownFor", "company", "location")
    vals = [fields[k].value for k in keys if fields[k].status == "found" and fields[k].value]
    uniq = list({fold(v): v for v in reversed(vals)}.values())[::-1]
    return " · ".join(uniq)


def _display_name(name: str, group: list[Mention]) -> str:
    names = Counter(m.name_as_written for m in group if name_match(name, m.name_as_written) >= 0.99)
    if not names:
        return name
    # Prefer the fullest frequently-used form ("Cristiano Ronaldo" over "Ronaldo").
    top = names.most_common(3)
    return max(top, key=lambda kv: (kv[1], len(kv[0])))[0]


def _strong_name_match(query: str, display: str) -> bool:
    """Every query token must appear in the person's name (diacritic-insensitive)."""
    q, d = fold(query).split(), set(fold(display).split())
    return bool(q) and (all(t in d for t in q) or spacing_variant(query, display))


# Fields a filter is compared with: "Field or industry" matches the person's
# industry or specialization too, a job title their occupation.
_FILTER_FIELDS = {
    "jobTitle": ["jobTitle", "occupation"],
    "occupation": ["occupation", "jobTitle"],
    "field": ["field", "industry", "specialization", "occupation", "academicInterests"],
    "education": ["education"],
    "company": ["company"],
}


def _filter_match(fields: dict[str, FieldValue], filters: dict[str, str], careers: list[CareerItem] | None = None) -> float:
    """Share of filters the candidate's facts satisfy; past roles count for job title and company."""
    hit = 0
    for k, want in filters.items():
        keys = _FILTER_FIELDS.get(k, [k])
        vals = [v for kk in keys if (fv := fields.get(kk)) for v in ([fv.value] if fv.value else []) + (fv.values or [])]
        if k in ("company", "jobTitle", "occupation"):
            vals += [x for c in careers or [] for x in ((c.organization,) if k == "company" else (c.title,)) if x]
        if any(_same_value(want, v) or token_overlap(want, v) >= 0.6 for v in vals):
            hit += 1
    return hit / len(filters)


def _avatar(photos: list[Photo]) -> str | None:
    verified = [p for p in photos if p.sourceIds] + [p for p in photos if not p.sourceIds]
    for p in verified:
        if p.url.startswith(("http://", "https://")):
            return p.url
    return None


def clean_image_query(raw: str | None, cand: FeCandidate) -> str:
    q = " ".join(_URLISH.sub(" ", raw or "").split())
    if not q:
        extra = [cand.occupation, cand.nationality, cand.fields["company"].value if "company" in cand.fields else None]
        q = " ".join([cand.displayName] + [e for e in extra if e][:2])
    if fold(cand.displayName) not in fold(q):
        q = f"{cand.displayName} {q}"
    if len(q) > 80:
        q = q[:80].rsplit(" ", 1)[0]
    return q


def _plain(text: str | None, limit: int) -> str | None:
    """Short plain string: no URLs/markdown, trimmed."""
    if not text:
        return None
    t = " ".join(re.sub(r"https?://\S+|\[([^\]]*)\]\([^)]*\)|[*_`#]", lambda m: m.group(1) or "", text).split())
    return t[:limit] or None


def split_aliases(display: str, aliases: list[str]) -> tuple[str, list[str]]:
    """"Trần Minh Hiếu (HIEUTHUHAI)" -> ("Trần Minh Hiếu", ["HIEUTHUHAI"]), plus
    the given aliases, deduplicated and without the display name."""
    m = re.match(r"^\s*(.+?)\s*\(([^()]+)\)\s*$", display or "")
    if m:
        display, aliases = m.group(1), [m.group(2), *aliases]
    out: dict[str, str] = {}
    for a in aliases:
        a = _plain(a, 60) if isinstance(a, str) else None
        if a and fold(a) != fold(display):
            out.setdefault(fold(a), a)
    return display, list(out.values())[:3]


def apply_profile(draft: CandidateDraft, profile: AnalyzedCandidate) -> None:
    """Apply LLM wording, constrained to what verified facts support."""
    cand = draft.candidate
    if (h := _plain(profile.headline, 120)) and fold(h) != fold(cand.displayName):
        cand.headline = h
    if note := _plain(profile.disambiguationNote, 300):
        cand.disambiguationNote = note
    f = cand.fields
    if f["occupation"].status == "found" or f["jobTitle"].status == "found":
        cand.occupation = _plain(profile.occupation, 80) or cand.occupation
    if any(f[k].status == "found" for k in ("knownFor", "company")) or any(x.field in ("awards", "notableWorks", "teams") for x in draft.facts):
        cand.knownFor = _plain(profile.knownFor, 120) or cand.knownFor
    cand.imageSearchQuery = clean_image_query(profile.imageSearchQuery, cand)
    cand.aliases = split_aliases(cand.displayName, profile.aliases + cand.aliases)[1]
    cand.careerArc = draft.careers
    cand.tags = list(dict.fromkeys(t.strip().lower() for t in profile.tags if t.strip()))[:6]
    if paragraphs := _paragraphs(profile.profile):
        cand.narrative, cand.narrativeSource = paragraphs, "llm"


def _paragraphs(paragraphs: list[str]) -> list[str]:
    return [p for p in (_plain(x, 800) for x in paragraphs[:3]) if p]
