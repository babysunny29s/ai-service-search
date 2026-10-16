"""Scrape Creators profile reads for networks that do not serve a page to a bot.

At most two GET /v1/{platform}/profile calls per person (1 credit each),
the most used networks first. Every public field on those profiles is kept for
the model, including age, email and phone. Photo-feed routes are not called.
https://docs.scrapecreators.com
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core import tracing
from app.infra.about_profiles import public_text
from app.core.cost import extra_vnd, scrape_cost_details
from app.core.cache import TTLCache
from app.core.models import ClientPhoto, ClientSource
from app.core.urls import account_handle, image_key, link_key, looks_like_image_url, profile_of

log = logging.getLogger(__name__)

_PLATFORM = {
    "facebook": "facebook",
    "instagram": "instagram",
    "tiktok": "tiktok",
    "x": "x",
    "linkedin": "linkedin",
    "threads": "threads",
}
_CRAWL_ORDER = ("facebook", "instagram", "threads", "tiktok", "x", "linkedin")
_META = {"success", "credits_remaining", "credits_charged", "cached", "cached_at"}
_MAX_CALLS = 2
_TEXT_CAP = 4000
_LINK = re.compile(r"https?://[^\s<>\"')\]]+")
_IMAGE_FILE = re.compile(r"\.(?:jpe?g|png|webp)(?:\?|$)", re.I)
_SKIP_KEYS = {
    "avatar_url", "id", "pk", "fbid", "user_id", "entity_id", "fetched_at", "source",
    "platform", "raw", "profile_pic_url", "hd_profile_pic_url_info", "eimu_id",
    "strong_id__", "interop_messaging_user_fbid", "gender", "password", "token", "cursor",
    "success", "credits_remaining", "credits_charged", "cached", "cached_at",
}
_SKIP_PART = (
    "profile_pic", "profile_picture", "password", "token", "cursor", "tracking",
    "eimu", "password", "nonce",
)
_PRIVATE = re.compile(r"(gender|password|token)$", re.I)


def _http_error(e: BaseException) -> str:
    resp = getattr(e, "response", None)
    bits = [type(e).__name__]
    text = str(e).strip()
    if text:
        bits.append(text)
    if resp is not None:
        bits.append(str(getattr(resp, "status_code", "") or ""))
        body = (getattr(resp, "text", None) or "")[:160].strip()
        if body:
            bits.append(body)
    return " ".join(b for b in bits if b) or "error"
_LABEL = {
    "display_name": "Name",
    "handle": "Handle",
    "bio": "Bio",
    "category": "Category",
    "verified": "Verified",
    "followers": "Followers",
    "following": "Following",
    "posts_count": "Posts",
    "country": "Country",
    "lang": "Language",
    "website": "Website",
    "location": "Location",
    "hometown": "Hometown",
    "work": "Work",
    "education": "Education",
    "about": "About",
    "occupation": "Occupation",
    "company": "Company",
    "school": "School",
    "city": "City",
    "current_city": "City",
    "description": "Bio",
    "biography": "Bio",
    "pageIntro": "Intro",
    "address": "Location",
    "followerCount": "Followers",
    "follower_count": "Followers",
    "full_name": "Name",
    "username": "Handle",
    "name": "Name",
    "intro": "Intro",
    "email": "Email",
    "phone": "Phone",
    "age": "Age",
    "birthday": "Birthday",
    "birth_date": "Birthday",
    "birthdate": "Birthday",
}


@dataclass
class CrawlBatch:
    text: dict[str, str] = field(default_factory=dict)
    photos: list[ClientPhoto] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    calls: int = 0
    credits: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: CrawlBatch) -> CrawlBatch:
        self.calls += other.calls
        self.credits += other.credits
        self.errors.extend(other.errors)
        for key, text in other.text.items():
            self.text.setdefault(key, text)
        have = {image_key(p.url) for p in self.photos}
        self.photos.extend(p for p in other.photos if image_key(p.url) not in have)
        seen = {link_key(u) for u in self.links}
        self.links.extend(u for u in other.links if link_key(u) not in seen)
        return self


def crawl_target(url: str) -> tuple[str, str] | None:
    """(platform, handle) for a profile URL, or None when we can read the page ourselves."""
    profile = profile_of(url)
    if not profile or profile.platform not in _PLATFORM:
        return None
    handle = account_handle(profile.url)
    if handle:
        return _PLATFORM[profile.platform], handle.lstrip("@")
    return _PLATFORM[profile.platform], profile.url


def scrape_request(platform: str, handle: str, page_url: str = "") -> tuple[str, dict[str, str]]:
    """Path and query for a Scrape Creators profile read."""
    url = page_url if page_url.startswith("http") else ""
    if platform == "facebook":
        return "/v1/facebook/profile", {"url": url or f"https://www.facebook.com/{handle}"}
    if platform == "linkedin":
        return "/v1/linkedin/profile", {"url": url or f"https://www.linkedin.com/in/{handle.strip('/')}/"}
    if platform == "instagram":
        return "/v1/instagram/profile", {"handle": handle, "trim": "true"}
    if platform == "tiktok":
        return "/v1/tiktok/profile", {"handle": handle}
    if platform == "x":
        return "/v1/twitter/profile", {"handle": handle}
    if platform == "threads":
        return "/v1/threads/profile", {"handle": handle}
    raise ValueError(platform)


def unwrap_profile(body: dict) -> dict:
    """Flatten a Scrape Creators response into the fields profile_text / photos expect."""
    data = body.get("data")
    if isinstance(data, dict) and isinstance(data.get("user"), dict):
        core = dict(data["user"])
    elif isinstance(data, dict):
        core = dict(data)
    else:
        core = {k: v for k, v in body.items() if k not in _META}
    legacy = core.get("legacy") if isinstance(core.get("legacy"), dict) else {}
    if not isinstance(core.get("avatar_url"), str):
        for key in ("profilePicLarge", "profilePicMedium", "profile_pic_url", "image"):
            val = core.get(key)
            if isinstance(val, str) and val.startswith("http"):
                core["avatar_url"] = val
                break
        pic = legacy.get("profile_image_url_https")
        if isinstance(pic, str) and pic.startswith("http"):
            core.setdefault("avatar_url", pic.replace("_normal.", "."))
    if not isinstance(core.get("bio"), str):
        for key in ("biography", "about", "pageIntro", "description"):
            val = core.get(key)
            if isinstance(val, str) and val.strip():
                core["bio"] = val
                break
        if isinstance(legacy.get("description"), str):
            core.setdefault("bio", legacy["description"])
    if not isinstance(core.get("display_name"), str):
        for key in ("full_name", "name"):
            val = core.get(key)
            if isinstance(val, str) and val.strip():
                core["display_name"] = val
                break
        if isinstance(legacy.get("name"), str):
            core.setdefault("display_name", legacy["name"])
    if "followers" not in core:
        for key in ("followerCount", "follower_count", "followers"):
            val = core.get(key)
            if isinstance(val, int):
                core["followers"] = val
                break
        edge = core.get("edge_followed_by")
        if isinstance(edge, dict) and isinstance(edge.get("count"), int):
            core["followers"] = edge["count"]
        elif isinstance(legacy.get("followers_count"), int):
            core["followers"] = legacy["followers_count"]
    core["raw"] = body
    return core


def _skip_key(key: str) -> bool:
    k = (key or "").lower()
    return k in _SKIP_KEYS or bool(_PRIVATE.search(k)) or any(p in k for p in _SKIP_PART)


def _fmt(value: object) -> str | None:
    if value is True:
        return "yes"
    if value is False or value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value) if value else None
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.startswith("data:"):
            return None
        if text.startswith("http") and _IMAGE_FILE.search(text):
            return None
        return text
    return None


def _walk(obj: object, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str) or _skip_key(key):
                continue
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                rows.extend(_walk(value, path))
                continue
            text = _fmt(value)
            if text:
                rows.append((path, text))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                rows.extend(_walk(item, prefix))
            else:
                text = _fmt(item)
                if text:
                    rows.append((prefix or "value", text))
    return rows


def profile_fields(data: dict) -> list[tuple[str, str]]:
    """Every public scalar on the profile, including include_raw, labeled for the model."""
    body = dict(data)
    raw = body.pop("raw", None)
    rows = _walk(body)
    if isinstance(raw, (dict, list)):
        rows.extend(_walk(raw, "raw"))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for key, value in rows:
        item = (key.lower(), value.lower() if not value.startswith("http") else value)
        if item in seen:
            continue
        seen.add(item)
        out.append((key, value))
    return out


def profile_text(data: dict) -> str:
    """The public profile fields, as one block for the model."""
    lines: list[str] = []
    seen_values: set[str] = set()
    for key, value in profile_fields(data):
        leaf = key.rsplit(".", 1)[-1]
        label = _LABEL.get(leaf) or _LABEL.get(key)
        line = value if label is None and leaf in {"bio", "description", "about"} else f"{label or leaf.replace('_', ' ')}: {value}"
        if leaf in {"display_name", "full_name", "name"} and label == "Name":
            line = value
        folded = line.lower()
        if folded in seen_values:
            continue
        seen_values.add(folded)
        lines.append(line)
    for link in data.get("links") or []:
        if isinstance(link, str) and link.startswith("http") and link.lower() not in seen_values:
            seen_values.add(link.lower())
            lines.append(link)
    cleaned = []
    for line in lines:
        text = public_text(line)
        if text:
            cleaned.append(text)
    return "\n".join(cleaned)[:_TEXT_CAP]


def _href(obj: object) -> str | None:
    if isinstance(obj, str) and obj.startswith("http"):
        return obj.rstrip(".,;)")
    if isinstance(obj, dict):
        for key in ("url", "href", "link", "website", "external_url"):
            value = obj.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value.rstrip(".,;)")
    return None


def hrefs_from_profile(data: dict, text: str = "") -> list[str]:
    """Every http(s) link on the profile: `links`, bio_links, website, raw, and the text dump."""
    raw: list[object] = []
    for key in ("links", "bio_links", "website", "external_url", "facebook_url", "threads_url"):
        value = data.get(key)
        if isinstance(value, list):
            raw.extend(value)
        elif value:
            raw.append(value)
    nested = data.get("raw")
    if isinstance(nested, dict):
        for key in ("links", "bio_links", "website", "external_url", "fb_url", "facebook_url"):
            value = nested.get(key)
            if isinstance(value, list):
                raw.extend(value)
            elif value:
                raw.append(value)
    for _, value in profile_fields(data):
        if value.startswith("http"):
            raw.append(value)
    raw.extend(_LINK.findall(text or ""))
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        url = _href(item)
        if not url:
            continue
        account = profile_of(url)
        url = account.url if account else url
        key = link_key(url)
        if key in seen:
            continue
        seen.add(key)
        out.append(url)
    return out


_SKIP_PHOTO = {"similarProfiles", "recommendations", "contributors", "articles", "activity", "recentPosts"}


def _photo_urls(obj: object, seen: set[str], depth: int = 0) -> list[str]:
    """https image URLs in the profile, including LinkedIn media.licdn.com and Instagram CDN in raw."""
    out: list[str] = []
    if depth > 8:
        return out
    if isinstance(obj, str) and obj.startswith("https://"):
        if looks_like_image_url(obj):
            ident = image_key(obj)
            if ident not in seen:
                seen.add(ident)
                out.append(obj)
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _SKIP_PHOTO or (isinstance(key, str) and key in _META):
                continue
            out.extend(_photo_urls(value, seen, depth + 1))
    elif isinstance(obj, list):
        for value in obj[:40]:
            out.extend(_photo_urls(value, seen, depth + 1))
    return out


def photos_from_profile(data: dict, page_url: str, title: str | None = None, text: str = "") -> list[ClientPhoto]:
    """Avatar, cover, post images in raw, and image files in the bio. No extra Crawlfeed call."""
    seen: set[str] = set()
    urls: list[str] = []
    for key in ("avatar_url", "cover_url", "banner_url", "photo_url", "profile_pic_url"):
        url = data.get(key)
        if isinstance(url, str) and url.startswith("https://"):
            ident = image_key(url)
            if ident not in seen:
                seen.add(ident)
                urls.append(url)
    urls.extend(_photo_urls(data, seen))
    for url in hrefs_from_profile(data, text):
        if looks_like_image_url(url):
            ident = image_key(url)
            if ident not in seen:
                seen.add(ident)
                urls.append(url)
    return [ClientPhoto(url=url, pageUrl=page_url, title=title, fromSource=True) for url in urls[:10]]


def linked_from_profile(data: dict, text: str) -> list[str]:
    """Other profile URLs written on this page (bio links, website, raw). Image URLs are photos, not crawls."""
    out: list[str] = []
    seen: set[str] = set()
    for url in hrefs_from_profile(data, text):
        if looks_like_image_url(url) or crawl_target(url) is None or link_key(url) in seen:
            continue
        seen.add(link_key(url))
        out.append(url)
    return out


def _bill_scrape(obs: Any, credits: int, **kwargs: Any) -> None:
    usage_details, cost_details = scrape_cost_details(credits)
    vnd = extra_vnd(scrape_credits=credits)
    obs.update(
        usage_details=usage_details,
        cost_details=cost_details,
        metadata={"vnd": round(vnd, 2)},
        **kwargs,
    )
    tracing.mark_vnd(obs, vnd)


class SocialCrawl:
    def __init__(self, api_key: str, base_url: str, timeout_s: float, cache_ttl_s: float):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.cache: TTLCache[tuple[dict | list, int]] = TTLCache(cache_ttl_s, max_items=500)

    async def read(self, sources: list[ClientSource], max_calls: int = _MAX_CALLS) -> CrawlBatch:
        if not self.api_key or max_calls <= 0:
            return CrawlBatch()
        queue: list[ClientSource] = []
        seen: set[tuple[str, str]] = set()

        def offer(source: ClientSource) -> None:
            target = crawl_target(source.url)
            if not target or target in seen:
                return
            seen.add(target)
            queue.append(source)

        ranked = sorted(sources, key=lambda s: _CRAWL_ORDER.index(p) if (p := (crawl_target(s.url) or ("",))[0]) in _CRAWL_ORDER else 99)
        for source in ranked:
            offer(source)
        batch = CrawlBatch()

        async def call(source: ClientSource):
            target = crawl_target(source.url)
            return await self._one(source, *target, errors=batch.errors) if target else None

        def absorb(source: ClientSource, row) -> None:
            if row is None:
                return
            key, text, photos, credits, data = row
            batch.credits += credits
            if text:
                batch.text[key] = text
            have = {image_key(p.url) for p in batch.photos}
            batch.photos.extend(p for p in photos if image_key(p.url) not in have)
            seen_links = {link_key(u) for u in batch.links}
            for url in hrefs_from_profile(data, text):
                if looks_like_image_url(url) or link_key(url) in seen_links:
                    continue
                seen_links.add(link_key(url))
                batch.links.append(url)
            if batch.calls < max_calls:
                for url in linked_from_profile(data, text):
                    offer(ClientSource(url=url, title=source.title, type=crawl_target(url)[0], verified=True))

        first: list[ClientSource] = []
        while queue and len(first) < max_calls:
            first.append(queue.pop(0))
        if first:
            rows = await asyncio.gather(*(call(s) for s in first))
            batch.calls += len(first)
            for source, row in zip(first, rows):
                absorb(source, row)
        while queue and batch.calls < max_calls:
            source = queue.pop(0)
            batch.calls += 1
            absorb(source, await call(source))
        return batch

    async def _one(
        self, source: ClientSource, platform: str, handle: str, errors: list[str] | None = None
    ) -> tuple[str, str, list[ClientPhoto], int, dict] | None:
        try:
            ident = source.url if platform in ("facebook", "linkedin") else handle
            data, credits = await self._profile(platform, ident)
        except Exception as e:
            detail = _http_error(e)
            log.warning("scrapecreators %s %s failed: %s", platform, handle, detail)
            if errors is not None:
                errors.append(f"{platform} @{handle}: {detail}")
            return None
        if not data:
            if errors is not None:
                errors.append(f"{platform} @{handle}: empty")
            return None
        text = profile_text(data)
        photos = photos_from_profile(data, source.url, source.title or None, text)
        return link_key(source.url), text, photos, credits, data

    async def _profile(self, platform: str, handle: str) -> tuple[dict, int]:
        data, credits = await self._get(platform, handle)
        return (data if isinstance(data, dict) else {}), credits

    async def _get(self, platform: str, handle: str) -> tuple[dict, int]:
        key = (platform, handle.lower())
        cached = self.cache.get(key)
        if cached is not None:
            tracing.record_scrape(cached=True)
            return cached
        path, params = scrape_request(platform, handle, handle if handle.startswith("http") else "")
        who = f"{platform} @{handle}"
        with tracing.span(f"scrapecreators:{platform}", as_type="generation", model="scrapecreators", input={"handle": handle}) as obs:
            try:
                log.info("scrapecreators GET %s %s", path, handle)
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    resp = await client.get(
                        f"{self.base_url}{path}",
                        params=params,
                        headers={"x-api-key": self.api_key},
                    )
                if resp.status_code == 404:
                    tracing.record_scrape(calls=1, credits=0, detail=f"{who} · 404")
                    _bill_scrape(obs, 0, output={"status": 404, "credits": 0}, model="scrapecreators")
                    self.cache.set(key, ({}, 0))
                    return {}, 0
                resp.raise_for_status()
                body = resp.json()
                if not isinstance(body, dict):
                    tracing.record_scrape(calls=1, credits=0, detail=f"{who} · empty")
                    _bill_scrape(obs, 0, output={"status": resp.status_code, "credits": 0}, model="scrapecreators")
                    self.cache.set(key, ({}, 0))
                    return {}, 0
                credits = int(body.get("credits_charged") or 0)
                their_cache = bool(body.get("cached"))
                if their_cache:
                    credits = 0
                tracing.record_scrape(calls=1, credits=credits, detail=who)
                _bill_scrape(
                    obs,
                    credits,
                    output={"credits": credits, "cached": their_cache, "status": resp.status_code},
                    model="scrapecreators",
                )
                value = unwrap_profile(body)
                self.cache.set(key, (value, credits))
                return value, credits
            except Exception as e:
                tracing.record_scrape(calls=1, credits=0, detail=f"{who} · {type(e).__name__}")
                _bill_scrape(
                    obs,
                    0,
                    output={"error": _http_error(e), "credits": 0},
                    model="scrapecreators",
                    level="ERROR",
                )
                raise

    def known_missing(self, url: str) -> bool:
        """True when a profile read already came back 404. Does not make a new call."""
        target = crawl_target(url)
        if not target:
            return False
        hit = self.cache.get((target[0], target[1].lower()))
        return hit is not None and not hit[0]

    def known_live(self, url: str) -> bool:
        """True when a profile read already returned a profile. Does not make a new call."""
        target = crawl_target(url)
        if not target:
            return False
        hit = self.cache.get((target[0], target[1].lower()))
        return hit is not None and bool(hit[0])
