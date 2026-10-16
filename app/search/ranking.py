"""Turns raw search hits into deduplicated, ranked Source records."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models import Source, utcnow
from app.infra.search_adapter import SearchHit
from app.core.urls import SOURCE_QUALITY, classify_source, fold, is_news, is_profile_url, name_match, normalize_url, social_platform


@dataclass
class _Agg:
    url: str
    hits: list[SearchHit] = field(default_factory=list)


def token_overlap(needle: str, text: str) -> float:
    """Share of needle tokens present in text (diacritic-insensitive)."""
    h, t = fold(needle), fold(text)
    if not h or not t:
        return 0.0
    if h in t:
        return 1.0
    tokens = [x for x in h.split() if len(x) > 1]
    if not tokens:
        return 0.0
    words = set(t.split())
    return sum(1 for x in tokens if x in words) / len(tokens)


def build_sources(
    hits: list[SearchHit],
    name: str,
    seeds: list[str] | None = None,
    exclude_urls: set[str] | None = None,
    start_index: int = 1,
) -> list[Source]:
    """Normalize, dedupe and rank hits. `seeds` are discovered attributes
    (enrichment round) that boost results mentioning them."""
    exclude_urls = exclude_urls or set()
    seeds = [s for s in (seeds or []) if s]
    by_url: dict[str, _Agg] = {}
    for h in hits:
        if not h.url.startswith("http"):
            continue
        key = normalize_url(h.url)
        if key in exclude_urls:
            continue
        by_url.setdefault(key, _Agg(key)).hits.append(h)

    now = utcnow()
    kept: list[Source] = []
    for url, agg in by_url.items():
        best = min(agg.hits, key=lambda h: h.position)
        title = best.title
        snippet = max((h.snippet for h in agg.hits), key=len)
        text = f"{title} {snippet}"
        nm = name_match(name, text)
        profile = bool(social_platform(url) and is_profile_url(url))
        if profile:
            # Profile titles are usually "Name - Headline | Platform".
            nm = max(nm, name_match(name, title))
        if nm < 0.4:
            continue
        stype = classify_source(url)
        if is_news(url):
            continue
        seed_score = max((token_overlap(s, text) for s in seeds), default=0.0)
        position_bonus = max(0.0, (11 - best.position) / 10) * 0.1
        multi_query_bonus = min(0.1, 0.05 * (len({h.query for h in agg.hits}) - 1))
        rank = 0.5 * nm + 0.2 * seed_score + 0.15 * SOURCE_QUALITY[stype] + (0.1 if profile else 0) + position_bonus + multi_query_bonus
        kept.append(
            Source(
                id="",
                title=title,
                url=url,
                snippet=snippet,
                sourceType=stype,  # type: ignore[arg-type]
                provider=best.provider,
                query=best.query,
                retrievedAt=now,
                contentStatus="snippet_only",
                rank=round(rank, 4),
            )
        )

    kept.sort(key=lambda s: s.rank, reverse=True)
    for i, s in enumerate(kept, start_index):
        s.id = f"src_{i}"
    return kept
