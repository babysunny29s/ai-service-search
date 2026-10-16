"""Deterministic search planner.

Discovery: only the name is known, so queries cast a wide net across
profile pages, bios and professional roles.
Enrichment: attributes discovered for a candidate (company, title, school,
location) become seeds for targeted follow-up queries."""

from __future__ import annotations

from dataclasses import dataclass

SOCIAL_SITES_CLAUSE = (
    "(site:facebook.com OR site:instagram.com OR site:youtube.com OR site:x.com"
    " OR site:tiktok.com OR site:linkedin.com/in OR site:github.com OR site:pinterest.com OR site:reddit.com/user)"
)


@dataclass(frozen=True)
class PlannedQuery:
    query: str
    kind: str  # "web" | "image"
    purpose: str


def _q(s: str) -> str:
    return f'"{s}"'


def _name_q(s: str) -> str:
    """Single-word names (stage names, nicknames) stay unquoted so the engine
    also matches spaced spellings ("Chipu" -> "Chi Pu")."""
    return _q(s) if len(s.split()) > 1 else s


def _dedupe(plan: list[PlannedQuery], done: set[str]) -> list[PlannedQuery]:
    out: list[PlannedQuery] = []
    for p in plan:
        key = f"{p.kind}:{' '.join(p.query.split())}"
        if key not in done:
            done.add(key)
            out.append(PlannedQuery(" ".join(p.query.split()), p.kind, p.purpose))
    return out


def discovery_queries(name: str, budget: int, done: set[str], filters: dict[str, str] | None = None) -> list[PlannedQuery]:
    """Ordered by value. The orchestrator runs the first one and only runs
    the rest when the first did not surface a knowledge panel."""
    n = _name_q(name)
    first = f"{n} {' '.join(filters.values())}" if filters else n
    plan = [
        PlannedQuery(first, "web", "general"),
        PlannedQuery(
            f"{n} (site:en.wikipedia.org OR site:wikipedia.org OR site:linkedin.com/in OR site:facebook.com OR site:instagram.com"
            " OR site:youtube.com OR site:x.com OR site:tiktok.com OR site:github.com)",
            "web",
            "profiles",
        ),
        PlannedQuery(f'{n} (CEO OR founder OR "giám đốc" OR "giảng viên" OR "phỏng vấn" OR interview)', "web", "professional"),
    ]
    return _dedupe(plan, done)[:budget]


def social_query(display_name: str, public_figure: bool = False) -> PlannedQuery:
    """Profiles of one resolved candidate. The name is not quoted because
    profiles often use a short form ("Leo Messi" for "Lionel Messi")."""
    name = display_name.replace('"', " ")
    official = " official" if public_figure else ""
    return PlannedQuery(" ".join(f"{name}{official} {SOCIAL_SITES_CLAUSE}".split()), "web", "socials")


def _anchor(seeds: dict[str, str]) -> str | None:
    for key in ("company", "education", "jobTitle", "occupation", "field", "location"):
        if seeds.get(key):
            return seeds[key]
    return None


def enrichment_queries(
    name: str,
    seeds: dict[str, str],
    budget: int,
    done: set[str],
    have_platforms: set[str] = frozenset(),  # type: ignore[assignment]
    include_image: bool = False,
) -> list[PlannedQuery]:
    """Targeted queries for one candidate."""
    anchor = _anchor(seeds)
    if not anchor:
        return []
    n = _q(name)
    plan: list[PlannedQuery] = [PlannedQuery(f"{n} {_q(anchor)}", "web", "anchor")]
    second = next((seeds[k] for k in ("jobTitle", "education", "location") if seeds.get(k) and seeds[k] != anchor), None)
    if second:
        plan.append(PlannedQuery(f"{n} {_q(anchor)} {second}", "web", "anchor+"))
    if "linkedin" not in have_platforms:
        plan.append(PlannedQuery(f"site:linkedin.com/in {n} {anchor}", "web", "platform:linkedin"))
    web = _dedupe(plan, done)[:budget]
    if include_image:
        web += _dedupe([PlannedQuery(f"{n} {anchor}", "image", "image")], done)
    return web
