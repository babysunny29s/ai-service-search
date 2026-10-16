"""Reads public pages with timeout, size limit and short-lived cache.
Pages behind login walls are skipped and marked as restricted."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import unquote, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.core.cache import TTLCache
from app.core.urls import can_fetch, is_profile_url, link_key, name_match, normalize_url, social_platform

DEFAULT_USER_AGENT = f"ProfileSearchBot/0.1 (+https://example.com/bot) httpx/{httpx.__version__}"
MAX_TEXT_CHARS = 12000


@dataclass
class FetchedPage:
    url: str
    status: str  # fetched | blocked | failed
    title: str = ""
    text: str = ""
    description: str = ""
    image: str | None = None
    profile_links: list[str] = field(default_factory=list)
    error: str | None = None
    # Content images as (url, alt/title text); og:image is often the site logo.
    images: list[tuple[str, str]] = field(default_factory=list)


def _is_public_host(url: str) -> bool:
    host = urlsplit(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


@dataclass
class ParsedPage:
    title: str
    description: str
    text: str
    image: str | None
    profile_links: list[str]
    images: list[tuple[str, str]] = field(default_factory=list)


def link_is_dead(status: int | None) -> bool:
    """404 and 410 are gone. 0 means the host does not resolve. A block or a timeout is not dead."""
    return status in (0, 404, 410)


MAX_PAGE_IMAGES = 40
_NOT_PHOTO = re.compile(r"logo|icon|sprite|banner|placeholder|blank|spacer|loading|avatar-default|flag|cron|pixel|captcha", re.I)
_LAZY_SRC = ("data-src", "data-original", "data-lazy-src", "data-full", "src")
# Academic and professional titles before a name ("PGS.TS. Nguyễn Văn A", "Dr. Jane Doe"): not in file names.
_HONORIFIC = re.compile(r"^(?:(?:gs|pgs|ts|tskh|ths|ks|bs|cn|ls|nts|dr|prof|mr|mrs|ms)\b\.?\s*)+", re.I)


def _small(tag) -> bool:
    for attr in ("width", "height"):
        v = str(tag.get(attr) or "").strip().removesuffix("px")
        if v.isdigit() and int(v) < 60:
            return True
    return False


def page_images(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Content images (lazy-load attributes first: they hold the full-size file), without
    logos, icons, tracking pixels and tiny images."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tag in soup.find_all("img"):
        src = next((tag.get(a) for a in _LAZY_SRC if isinstance(tag.get(a), str) and tag.get(a).strip()), None)
        if not src or src.startswith("data:") or _small(tag):
            continue
        url = urljoin(base_url, src.strip())
        path = urlsplit(url).path.lower()
        if not url.startswith(("http://", "https://")) or path.endswith((".svg", ".gif", ".php")) or _NOT_PHOTO.search(path):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append((url, " ".join(str(tag.get(a) or "") for a in ("alt", "title")).strip()))
        if len(out) >= MAX_PAGE_IMAGES:
            break
    return out


def named_images(page: FetchedPage, *names: str, limit: int = 2) -> list[str]:
    """Page images whose file name or alt text carries the full name: staff and profile
    pages show the portrait in the body ("nguyen-truong-giang.jpg") while og:image is
    the site logo."""
    bare = [_HONORIFIC.sub("", n).strip() for n in names if n]
    out = []
    for url, alt in page.images:
        label = f"{unquote(urlsplit(url).path.rsplit('/', 1)[-1])} {alt}"
        if any(len(n) >= 4 and name_match(n, label) >= 0.8 for n in bare):
            out.append(url)
            if len(out) >= limit:
                break
    return out


def parse_html(html: str, base_url: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    description = (desc_tag.get("content") or "").strip() if desc_tag else ""
    og = soup.find("meta", attrs={"property": "og:image"})
    image = urljoin(base_url, og["content"]) if og and og.get("content") else None
    images = page_images(soup, base_url)
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.startswith("http") and social_platform(href) and is_profile_url(href):
            n = normalize_url(href)
            if n not in links:
                links.append(n)
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "iframe", "header", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return ParsedPage(title, description, text[:MAX_TEXT_CHARS], image, links[:20], images)


def _fold_1to1(text: str) -> str:
    """Lowercase + strip diacritics keeping string length, so indexes map back."""
    out = []
    for c in text:
        d = "d" if c in "đĐ" else unicodedata.normalize("NFKD", c)[0]
        out.append(d.lower() if len(d.lower()) == 1 else d)
    return "".join(out)


def extlinks_from_payload(data: dict) -> list[str]:
    """URLs from a MediaWiki `prop=extlinks` response. The link field is `*` or `url`."""
    pages = (data.get("query") or {}).get("pages") or {}
    out: list[str] = []
    seen: set[str] = set()
    for page in pages.values():
        if not isinstance(page, dict):
            continue
        for item in page.get("extlinks") or []:
            raw = item.get("*") or item.get("url") if isinstance(item, dict) else None
            if isinstance(raw, str) and raw.startswith("http") and raw not in seen:
                seen.add(raw)
                out.append(raw)
    return out


def excerpt(text: str, name: str, limit: int, window: int = 350) -> str:
    """Passages around mentions of the name, capped at `limit` chars."""
    text = unicodedata.normalize("NFC", text)
    hay = _fold_1to1(text)
    tokens = _fold_1to1(unicodedata.normalize("NFC", name)).split()
    needles = [" ".join(tokens)] + ([" ".join(tokens[-2:])] if len(tokens) > 2 else []) + ([tokens[-1]] if len(tokens) > 1 else [])
    spans: list[list[int]] = []
    for needle in needles:
        for m in re.finditer(re.escape(needle), hay):
            spans.append([max(0, m.start() - window), min(len(text), m.end() + window)])
        if spans:
            break
    compact = re.sub(r"\W", "", "".join(tokens))
    if not spans and len(compact) >= 4:
        for m in re.finditer(r"\W?".join(map(re.escape, compact)), hay):
            spans.append([max(0, m.start() - window), min(len(text), m.end() + window)])
    if not spans:
        return text[:limit]
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, total = [], 0
    for a, b in merged:
        chunk = text[a:b].strip()
        if total + len(chunk) > limit:
            chunk = chunk[: max(0, limit - total)]
        if chunk:
            out.append(chunk)
            total += len(chunk)
        if total >= limit:
            break
    return " … ".join(out)


class PageFetcher:
    def __init__(self, timeout_s: float, max_bytes: int, cache_ttl_s: float, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.cache: TTLCache[FetchedPage] = TTLCache(cache_ttl_s, max_items=500)
        self._wiki_pages: TTLCache[object] = TTLCache(cache_ttl_s, max_items=200)

    async def fetch_many(self, urls: list[str], concurrency: int = 4) -> dict[str, FetchedPage]:
        sem = asyncio.Semaphore(concurrency)

        async def run(u: str) -> FetchedPage:
            async with sem:
                return await self.fetch(u)

        pages = await asyncio.gather(*(run(u) for u in urls))
        return {p.url: p for p in pages}

    async def dead_links(self, urls: list[str]) -> set[str]:
        """Pages that are gone: 404, 410, or the host does not resolve.
        A timeout or a login wall is not a dead link."""
        unique: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if not url.startswith("http"):
                continue
            key = url.split("#", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
            if len(unique) >= 20:
                break
        sem = asyncio.Semaphore(6)

        async def one(url: str) -> str | None:
            async with sem:
                status = await self.link_status(url)
                return url if link_is_dead(status) else None

        rows = await asyncio.gather(*(one(u) for u in unique))
        return {link_key(u) for u in rows if u}

    async def link_status(self, url: str) -> int | None:
        """Final HTTP status. 0 when the host does not resolve. None when the check did not finish."""
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        if not await asyncio.to_thread(_is_public_host, url):
            return 0
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, follow_redirects=True, headers={"User-Agent": self.user_agent}
            ) as client:
                resp = await client.head(url)
                if resp.status_code in (405, 501):
                    async with client.stream("GET", url) as got:
                        return got.status_code
                return resp.status_code
        except httpx.ConnectError:
            return 0
        except httpx.HTTPError:
            return None

    async def fetch(self, url: str) -> FetchedPage:
        cached = self.cache.get(url)
        if cached:
            return cached
        page = await self._fetch(url)
        if page.status != "failed":
            self.cache.set(url, page)
        return page

    async def wikipedia_page(self, url: str):
        """Article text, lead image, Wikidata facts and external profile links."""
        from app.infra.wiki import WikiPage, merge_pages, parse_query, query_params, wiki_target

        target = wiki_target(url)
        if not target:
            return WikiPage(url=url)
        host, title = target
        key = f"wikipage:{host}:{title}"
        if (cached := self._wiki_pages.get(key)) is not None:
            return cached
        page = await self._wiki_query(host, title, url)
        if page.english_url and wiki_target(page.english_url):
            en_host, en_title = wiki_target(page.english_url)
            if en_host != host or en_title != title:
                extra = await self._wiki_query(en_host, en_title, page.english_url)
                page = merge_pages(page, extra)
        if page.qid:
            facts, social = await self._wikidata(page.qid)
            page.facts = facts
            page.links = list(dict.fromkeys([*page.links, *social]))
        self._wiki_pages.set(key, page)
        if page.url:
            en_target = wiki_target(page.url)
            if en_target and f"wikipage:{en_target[0]}:{en_target[1]}" != key:
                self._wiki_pages.set(f"wikipage:{en_target[0]}:{en_target[1]}", page)
        return page

    async def _wiki_query(self, host: str, title: str, url: str):
        from app.infra.wiki import WikiPage, parse_query, query_params

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, headers={"User-Agent": self.user_agent}) as client:
                resp = await client.get(f"https://{host}/w/api.php", params=query_params(title))
                if resp.status_code != 200:
                    return WikiPage(url=url)
                data = resp.json()
                page = parse_query(data, url)
                nxt = data.get("continue") or {}
                for _ in range(3):
                    token = nxt.get("elcontinue")
                    if not token:
                        break
                    more = await client.get(
                        f"https://{host}/w/api.php",
                        params={"action": "query", "prop": "extlinks", "ellimit": "500", "elprotocol": "https", "elcontinue": token, "redirects": "1", "format": "json", "titles": title},
                    )
                    if more.status_code != 200:
                        break
                    extra = more.json()
                    page.links = list(dict.fromkeys([*page.links, *extlinks_from_payload(extra)]))
                    nxt = extra.get("continue") or {}
                return page
        except (httpx.HTTPError, ValueError):
            pass
        return WikiPage(url=url)

    async def _wikidata(self, qid: str) -> tuple[dict[str, str], list[str]]:
        from app.infra.wiki import facts_from_entity, labels_from_entities, social_from_entity, _need_labels

        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, headers=headers) as client:
                resp = await client.get(
                    "https://www.wikidata.org/w/api.php",
                    params={"action": "wbgetentities", "ids": qid, "props": "labels|claims", "languages": "en|vi", "format": "json"},
                )
            if resp.status_code != 200:
                return {}, []
            payload = resp.json()
            entity = (payload.get("entities") or {}).get(qid) or {}
            if not isinstance(entity, dict):
                return {}, []
            labels = labels_from_entities(payload)
            missing = [i for i in _need_labels(entity) if i not in labels]
            if missing:
                async with httpx.AsyncClient(timeout=self.timeout_s, headers=headers) as client:
                    lab = await client.get(
                        "https://www.wikidata.org/w/api.php",
                        params={"action": "wbgetentities", "ids": "|".join(missing[:50]), "props": "labels", "languages": "en|vi", "format": "json"},
                    )
                if lab.status_code == 200:
                    labels.update(labels_from_entities(lab.json()))
            return facts_from_entity(entity, labels), social_from_entity(entity)
        except (httpx.HTTPError, ValueError):
            return {}, []

    async def wikipedia_summary(self, url: str) -> tuple[str | None, str]:
        """Lead image and lead-section summary of a Wikipedia article via the
        public REST summary API (free, no search credits)."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host.endswith(".wikipedia.org") or not parts.path.startswith("/wiki/"):
            return None, ""
        title = parts.path[len("/wiki/") :]
        key = f"wikisummary:{host}:{title}"
        if (cached := self.cache.get(key)) is not None:
            return cached.image, cached.text
        image, extract = None, ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, headers={"User-Agent": self.user_agent}) as client:
                resp = await client.get(f"https://{host}/api/rest_v1/page/summary/{title}")
            if resp.status_code == 200:
                data = resp.json()
                src = (data.get("thumbnail") or data.get("originalimage") or {}).get("source")
                image = src if isinstance(src, str) and src.startswith("https://") else None
                # Disambiguation pages list namesakes; they say nothing about the person.
                if data.get("type") != "disambiguation" and isinstance(data.get("extract"), str):
                    extract = data["extract"].strip()
        except (httpx.HTTPError, ValueError):
            return None, ""
        self.cache.set(key, FetchedPage(key, "fetched", image=image, text=extract))
        return image, extract

    async def wikipedia_thumbnail(self, url: str) -> str | None:
        return (await self.wikipedia_summary(url))[0]

    async def wikipedia_images(self, url: str, limit: int = 10) -> list[str]:
        """Several photos already on the article (lead + infobox/gallery), no search credit."""
        from app.infra.wiki import images_from_generator, wiki_target

        target = wiki_target(url)
        if not target:
            return []
        host, title = target
        key = f"wikiimg:{host}:{title}:{limit}"
        if (cached := self._wiki_pages.get(key)) is not None:
            return list(cached.links)
        lead = await self.wikipedia_thumbnail(url)
        urls: list[str] = [lead] if lead else []
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, headers={"User-Agent": self.user_agent}) as client:
                resp = await client.get(
                    f"https://{host}/w/api.php",
                    params={
                        "action": "query",
                        "generator": "images",
                        "titles": title,
                        "gimlimit": str(min(max(limit, 8), 20)),
                        "prop": "imageinfo",
                        "iiprop": "url|mime|size",
                        "iiurlwidth": "800",
                        "redirects": "1",
                        "format": "json",
                    },
                )
            if resp.status_code == 200:
                for image in images_from_generator(resp.json(), limit):
                    if image not in urls:
                        urls.append(image)
        except (httpx.HTTPError, ValueError):
            pass
        urls = urls[:limit]
        from app.infra.wiki import WikiPage

        self._wiki_pages.set(key, WikiPage(url=url, links=urls))
        return urls

    async def wikipedia_extlinks(self, url: str) -> list[str]:
        """External links on a Wikipedia article (infobox Facebook, Instagram, …).
        The public MediaWiki API, no search credits."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host.endswith(".wikipedia.org") or not parts.path.startswith("/wiki/"):
            return []
        title = unquote(parts.path[len("/wiki/") :])
        key = f"wikiext:{host}:{title}"
        if (cached := self.cache.get(key)) is not None:
            return list(cached.profile_links)
        links: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, headers={"User-Agent": self.user_agent}) as client:
                resp = await client.get(
                    f"https://{host}/w/api.php",
                    params={
                        "action": "query",
                        "prop": "extlinks",
                        "ellimit": "100",
                        "elprotocol": "https",
                        "redirects": "1",
                        "format": "json",
                        "titles": title,
                    },
                )
            if resp.status_code == 200:
                links = extlinks_from_payload(resp.json())
        except (httpx.HTTPError, ValueError):
            return []
        self.cache.set(key, FetchedPage(key, "fetched", profile_links=links))
        return links

    async def english_wikipedia(self, url: str) -> str | None:
        """The English article for a Wikipedia article in another language, via its
        interlanguage links; None when there is none."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host.endswith(".wikipedia.org") or not parts.path.startswith("/wiki/"):
            return None
        if host.split(".")[0] == "en":
            return url
        title = unquote(parts.path[len("/wiki/") :])
        key = f"wikien:{host}:{title}"
        if (cached := self.cache.get(key)) is not None:
            return cached.text or None
        en = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, headers={"User-Agent": self.user_agent}) as client:
                resp = await client.get(
                    f"https://{host}/w/api.php",
                    params={"action": "query", "prop": "langlinks", "lllang": "en", "llprop": "url", "redirects": "1", "format": "json", "titles": title},
                )
            if resp.status_code == 200:
                pages = (resp.json().get("query") or {}).get("pages") or {}
                links = next(iter(pages.values()), {}).get("langlinks") or []
                en = next((x.get("url") for x in links if str(x.get("url", "")).startswith("https://en.wikipedia.org/wiki/")), "") or ""
        except (httpx.HTTPError, ValueError):
            return None
        self.cache.set(key, FetchedPage(key, "fetched", text=en))
        return en or None

    async def _fetch(self, url: str) -> FetchedPage:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, follow_redirects=False, headers={"User-Agent": self.user_agent, "Accept-Language": "en"}
            ) as client:
                target = url
                for _ in range(4):
                    if not can_fetch(target):
                        return FetchedPage(url, "blocked", error="login-walled platform")
                    if urlsplit(target).scheme not in ("http", "https") or not await asyncio.to_thread(_is_public_host, target):
                        return FetchedPage(url, "blocked", error="non-public host")
                    req = client.build_request("GET", target)
                    resp = await client.send(req, stream=True)
                    if resp.is_redirect and resp.headers.get("location"):
                        await resp.aclose()
                        target = str(resp.url.join(resp.headers["location"]))
                        continue
                    break
                else:
                    return FetchedPage(url, "failed", error="too many redirects")
                try:
                    if resp.status_code in (401, 403, 451):
                        return FetchedPage(url, "blocked", error=f"http {resp.status_code}")
                    if resp.status_code >= 400:
                        return FetchedPage(url, "failed", error=f"http {resp.status_code}")
                    ctype = resp.headers.get("content-type", "")
                    if "html" not in ctype and "text" not in ctype:
                        return FetchedPage(url, "failed", error=f"unsupported content-type {ctype[:50]}")
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > self.max_bytes:
                            break
                    html = buf.decode(resp.encoding or "utf-8", errors="replace")
                finally:
                    await resp.aclose()
        except httpx.HTTPError as e:
            return FetchedPage(url, "failed", error=type(e).__name__)
        p = await asyncio.to_thread(parse_html, html, target)
        if len(p.text) < 80:
            return FetchedPage(url, "failed", title=p.title, description=p.description, error="empty page (likely JS-rendered)")
        return FetchedPage(
            url, "fetched", title=p.title, text=p.text, description=p.description, image=p.image, profile_links=p.profile_links, images=p.images
        )
