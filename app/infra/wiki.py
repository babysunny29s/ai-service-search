"""Public Wikipedia + Wikidata reads. No search credits.

A Wikipedia article about the person is treated as a profile: the article
text, the lead image, infobox facts from Wikidata, and external profile
links (Instagram, YouTube, the official site, …).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

from app.infra.fetcher import extlinks_from_payload
from app.core.urls import link_key, profile_of

_EXTRACT_CHARS = 8000
_MONTH = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
# Living-person public facts. Street address is not requested. Email and phone
# are kept when Wikidata publishes them (P968, P1329).
_FACT_PROPS = {
    "P569": "born",
    "P19": "birthPlace",
    "P27": "nationality",
    "P551": "location",
    "P106": "occupation",
    "P108": "company",
    "P69": "education",
    "P136": "field",
    "P800": "knownFor",
    "P166": "awards",
    "P463": "knownFor",
    "P856": "website",
    "P968": "email",
    "P1329": "phone",
}
_SOCIAL_PROPS = {
    "P2002": "https://x.com/{v}",
    "P2003": "https://www.instagram.com/{v}/",
    "P2013": "https://www.facebook.com/{v}",
    "P7085": "https://www.tiktok.com/@{v}",
    "P2397": "https://www.youtube.com/channel/{v}",
    "P11260": "https://www.youtube.com/@{v}",
    "P4265": "https://www.linkedin.com/in/{v}/",
    "P2037": "https://github.com/{v}",
    "P4175": "https://www.pinterest.com/{v}/",
    "P3265": "https://www.reddit.com/user/{v}/",
}
_FACT_LABELS = {
    "born": "Born",
    "nationality": "Nationality",
    "location": "Location",
    "occupation": "Occupation",
    "company": "Company",
    "education": "Education",
    "field": "Field",
    "knownFor": "Known for",
    "awards": "Awards",
    "website": "Website",
    "email": "Email",
    "phone": "Phone",
}


@dataclass
class WikiPage:
    url: str
    title: str = ""
    extract: str = ""
    image: str | None = None
    links: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    qid: str | None = None
    english_url: str | None = None


def is_wikipedia(url: str, type_: str | None = None) -> bool:
    return type_ == "wikipedia" or (wiki_target(url) is not None)


def is_english_wiki(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in ("en.wikipedia.org", "en.m.wikipedia.org")


def wiki_host_rank(url: str) -> int:
    """English Wikipedia first, then any other edition."""
    return 0 if is_english_wiki(url) else 1


def one_wikipedia(sources: list) -> list:
    """Keep one Wikipedia article: the English edition when we have it."""
    wikis = [s for s in sources if is_wikipedia(getattr(s, "url", "") or "", getattr(s, "type", None))]
    keep = min(wikis, key=lambda s: wiki_host_rank(s.url)) if wikis else None
    keep_key = link_key(keep.url) if keep is not None else None
    out, taken = [], False
    for s in sources:
        url = getattr(s, "url", "") or ""
        if is_wikipedia(url, getattr(s, "type", None)):
            if taken or (keep_key and link_key(url) != keep_key):
                continue
            taken = True
        out.append(s)
    return out


def wiki_target(url: str) -> tuple[str, str] | None:
    """(host, title) for a /wiki/ article URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if not host.endswith(".wikipedia.org") or not parts.path.startswith("/wiki/"):
        return None
    title = unquote(parts.path[len("/wiki/") :])
    if not title or title.lower().startswith(("file:", "category:", "special:", "template:", "help:", "wikipedia:")):
        return None
    return host, title


def paragraphs(extract: str, limit: int = 4) -> list[str]:
    parts = [p for p in (" ".join(x.split()) for x in re.split(r"\n+", extract or "")) if len(p) >= 24]
    return parts[:limit]


_SKIP_FILE = re.compile(
    r"\b(logo|icon|flag|map|coat[ _]of[ _]arms|signature|symbol|banner|wordmark|pictogram|padlock|star|question)\b",
    re.I,
)


def images_from_generator(data: dict, limit: int = 10) -> list[str]:
    """Portrait-sized JPEG/PNG URLs from a MediaWiki `generator=images` response."""
    pages = (data.get("query") or {}).get("pages") or {}
    out: list[str] = []
    seen: set[str] = set()
    for page in pages.values() if isinstance(pages, dict) else []:
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "")
        if _SKIP_FILE.search(title.replace("File:", "").replace("_", " ")):
            continue
        info = (page.get("imageinfo") or [{}])[0]
        if not isinstance(info, dict):
            continue
        mime = str(info.get("mime") or "")
        if mime and mime not in ("image/jpeg", "image/png", "image/webp"):
            continue
        url = info.get("thumburl") or info.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        key = url.split("?")[0].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(url)
        if len(out) >= limit:
            break
    return out


def query_params(title: str) -> dict[str, str]:
    return {
        "action": "query",
        "prop": "extracts|pageimages|extlinks|pageprops|langlinks",
        "exlimit": "1",
        "explaintext": "1",
        "exsectionformat": "plain",
        "piprop": "original|thumbnail",
        "pithumbsize": "800",
        "ppprop": "wikibase_item",
        "ellimit": "500",
        "elprotocol": "https",
        "lllang": "en",
        "llprop": "url",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }


def parse_query(data: dict, url: str) -> WikiPage:
    pages = (data.get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), {}) if pages else {}
    if not isinstance(page, dict) or page.get("missing") is not None:
        return WikiPage(url=url)
    extract = page.get("extract") if isinstance(page.get("extract"), str) else ""
    extract = extract.strip()[:_EXTRACT_CHARS]
    original = (page.get("original") or {}).get("source")
    thumb = (page.get("thumbnail") or {}).get("source")
    image = original if isinstance(original, str) and original.startswith("https://") else None
    if image is None and isinstance(thumb, str) and thumb.startswith("https://"):
        image = thumb
    lang = page.get("langlinks") or []
    english = next((x.get("url") for x in lang if str(x.get("url") or "").startswith("https://en.wikipedia.org/wiki/")), "") or None
    qid = (page.get("pageprops") or {}).get("wikibase_item")
    qid = qid if isinstance(qid, str) and qid.startswith("Q") else None
    return WikiPage(
        url=url,
        title=str(page.get("title") or ""),
        extract=extract,
        image=image,
        links=extlinks_from_payload(data),
        qid=qid,
        english_url=english,
    )


def english_wiki_url(page: WikiPage) -> str | None:
    for candidate in (page.english_url, page.url):
        if candidate and is_english_wiki(candidate) and wiki_target(candidate):
            return candidate
    return None


def merge_pages(primary: WikiPage, extra: WikiPage | None) -> WikiPage:
    if extra is None:
        return primary
    en, local = (None, None)
    if is_english_wiki(extra.url) and extra.extract:
        en, local = extra, primary
    elif is_english_wiki(primary.url):
        en, local = primary, extra
    if en is not None:
        links = list(dict.fromkeys([*en.links, *local.links]))
        extract = en.extract
        if local.extract and local.extract not in extract:
            extract = (extract + "\n\n" + local.extract).strip()[:_EXTRACT_CHARS]
        return WikiPage(
            url=en.url,
            title=en.title or local.title,
            extract=extract[:_EXTRACT_CHARS],
            image=en.image or local.image,
            links=links,
            facts=en.facts or local.facts,
            qid=en.qid or local.qid,
            english_url=en.url,
        )
    links = list(dict.fromkeys([*primary.links, *extra.links]))
    extract = primary.extract
    if len(extra.extract) > len(extract) + 200:
        extract = extra.extract if not extract else extract + "\n\n" + extra.extract
    elif extra.extract and extra.extract not in extract:
        extract = (extract + "\n\n" + extra.extract).strip()[:_EXTRACT_CHARS]
    return WikiPage(
        url=primary.url,
        title=primary.title or extra.title,
        extract=extract[:_EXTRACT_CHARS],
        image=primary.image or extra.image,
        links=links,
        facts=primary.facts or extra.facts,
        qid=primary.qid or extra.qid,
        english_url=primary.english_url or extra.english_url or extra.url,
    )


def _time(value: dict) -> str:
    raw = str(value.get("time") or "")
    m = re.match(r"[+-]?(\d+)-(\d+)-(\d+)", raw)
    if not m:
        return ""
    year, month, day = m.group(1).lstrip("0") or "0", int(m.group(2)), int(m.group(3))
    precision = int(value.get("precision") or 11)
    if precision <= 9:
        return year
    if not (1 <= month <= 12):
        return year
    if precision == 10:
        return f"{_MONTH[month - 1]} {year}"
    return f"{day} {_MONTH[month - 1]} {year}"


def _snaks(entity: dict, prop: str) -> list[dict]:
    claims = (entity.get("claims") or {}).get(prop) or []
    out: list[dict] = []
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("rank") == "deprecated":
            continue
        snak = claim.get("mainsnak") or {}
        if snak.get("snaktype") != "value":
            continue
        dv = snak.get("datavalue") or {}
        val = dv.get("value")
        if val is not None:
            out.append({"type": dv.get("type"), "value": val, "rank": claim.get("rank")})
    out.sort(key=lambda s: s["rank"] != "preferred")
    return out


def _need_labels(entity: dict) -> list[str]:
    ids: list[str] = []
    for prop in _FACT_PROPS:
        for snak in _snaks(entity, prop):
            if snak["type"] == "wikibase-entityid" and isinstance(snak["value"], dict):
                qid = snak["value"].get("id")
                if isinstance(qid, str) and qid.startswith("Q"):
                    ids.append(qid)
    return list(dict.fromkeys(ids))


def _label(labels: dict[str, str], qid: str) -> str:
    return labels.get(qid) or ""


def facts_from_entity(entity: dict, labels: dict[str, str]) -> dict[str, str]:
    """Public infobox facts. Several occupations/schools are joined."""
    out: dict[str, list[str]] = {}
    for prop, key in _FACT_PROPS.items():
        values: list[str] = []
        for snak in _snaks(entity, prop)[:4]:
            raw = snak["value"]
            if snak["type"] == "time" and isinstance(raw, dict):
                text = _time(raw)
            elif snak["type"] == "wikibase-entityid" and isinstance(raw, dict):
                text = _label(labels, str(raw.get("id") or ""))
            elif snak["type"] == "string" and isinstance(raw, str):
                text = raw.strip()
            elif snak["type"] == "monolingualtext" and isinstance(raw, dict):
                text = str(raw.get("text") or "").strip()
            else:
                text = ""
            if text and text not in values:
                values.append(text)
        if values:
            out.setdefault(key, []).extend(values)
    facts: dict[str, str] = {}
    born = (out.get("born") or [""])[0]
    place = (out.get("birthPlace") or [""])[0]
    if born and place:
        facts["born"] = f"{born}, {place}"
    elif born:
        facts["born"] = born
    elif place:
        facts["born"] = place
    for key in ("nationality", "location", "occupation", "company", "education", "field", "knownFor", "awards", "website"):
        vals = [v for v in out.get(key) or [] if v]
        if vals:
            facts[key] = ", ".join(vals[:3])[:240]
    return facts


def social_from_entity(entity: dict) -> list[str]:
    links: list[str] = []
    for prop, template in _SOCIAL_PROPS.items():
        for snak in _snaks(entity, prop)[:2]:
            if snak["type"] != "string" or not isinstance(snak["value"], str):
                continue
            handle = snak["value"].strip().lstrip("@")
            if not handle:
                continue
            url = template.format(v=handle)
            if profile_of(url) or prop == "P856":
                links.append(url)
    for snak in _snaks(entity, "P856")[:2]:
        if snak["type"] == "string" and isinstance(snak["value"], str) and snak["value"].startswith("http"):
            links.append(snak["value"].strip())
    return list(dict.fromkeys(links))


def labels_from_entities(data: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for qid, ent in (data.get("entities") or {}).items():
        if not isinstance(ent, dict):
            continue
        labs = ent.get("labels") or {}
        text = (labs.get("en") or labs.get("vi") or next(iter(labs.values()), {}) or {}).get("value")
        if isinstance(text, str) and text.strip():
            out[qid] = text.strip()
    return out


def fact_items(facts: dict[str, str]):
    for key, label in _FACT_LABELS.items():
        if facts.get(key):
            yield key, label, facts[key]
