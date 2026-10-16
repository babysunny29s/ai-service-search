"""Photos taken from a person's own source pages, for people that have sources
but no photo. Costs no search call and no LLM call, only page fetches:

- Wikipedia: the article's lead image (public REST API)
- GitHub: the profile picture
- other fetchable pages (official site, YouTube, staff pages, ...): body images named
  after the person (file name or alt text; they count as the person's own), then og:image

Login-walled networks (Facebook, Instagram, TikTok, X, LinkedIn) are never
fetched. A page image is often a logo or a banner, so with the face model only
images showing the person's face are kept: the face of a trusted page (the
person's own) or the face two images agree on."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from app.config import Settings
from app.media.faces import FaceVerifier, distinct_pictures, distinct_urls, select_photos
from app.infra.fetcher import PageFetcher, named_images
from app.core.models import ClientPhoto, ClientSource
from app.core.urls import can_fetch, image_key, name_match, profile_of

# Fetch order: pages most likely to carry a portrait first.
_RANK = {"wikipedia": 0, "github": 1, "website": 2, "youtube": 3, "academic": 5, "web": 6}


@dataclass
class _Found:
    url: str
    page: ClientSource
    trusted: bool


def _is_wikipedia(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.hostname or "").endswith(".wikipedia.org") and parts.path.startswith("/wiki/")


def _trusted(s: ClientSource, name: str) -> bool:
    """The page is the person's own (its image shows them), not just a page mentioning them."""
    if _is_wikipedia(s.url):
        # A Wikipedia link from the model's memory counts when the article is named after the person.
        title = unquote(urlsplit(s.url).path[len("/wiki/") :]).replace("_", " ")
        return s.verified or name_match(name, title) >= 0.8
    return s.verified and s.type not in ("web", "academic")


class SourcePhotoFinder:
    def __init__(self, settings: Settings, fetcher: PageFetcher, faces: FaceVerifier | None):
        self.s = settings
        self.fetcher = fetcher
        self.faces = faces

    def _pages(self, sources: list[ClientSource], max_pages: int) -> list[ClientSource]:
        seen: set[str] = set()
        pages: list[ClientSource] = []
        for s in sources:
            if not s.url.startswith(("http://", "https://")) or not can_fetch(s.url) or s.url in seen:
                continue
            seen.add(s.url)
            pages.append(s)
        kind = lambda s: "wikipedia" if _is_wikipedia(s.url) else s.type  # noqa: E731
        pages.sort(key=lambda s: (_RANK.get(kind(s), 4), not s.verified))
        return pages[:max_pages]

    async def _images_of(self, s: ClientSource, name: str) -> list[tuple[str, bool]]:
        """(image url, named after the person) for one page, best first."""
        if _is_wikipedia(s.url):
            getter = getattr(self.fetcher, "wikipedia_images", None)
            urls = await getter(s.url, limit=10) if getter else [await self.fetcher.wikipedia_thumbnail(s.url)]
            return [(u, False) for u in urls if u]
        if (p := profile_of(s.url)) and p.platform == "github":
            return [(f"{p.url.rstrip('/')}.png", False)]
        page = await self.fetcher.fetch(s.url)
        if page.status != "fetched":
            return []
        named = [(u, True) for u in named_images(page, name)]
        og = [(page.image, False)] if page.image and page.image.startswith(("http://", "https://")) else []
        return named + og

    async def find(
        self,
        name: str,
        sources: list[ClientSource],
        limit: int | None = None,
        max_pages: int | None = None,
        require_face: bool = True,
    ) -> tuple[list[ClientPhoto], bool]:
        """Photos (best first) and whether their faces were checked.

        Search-profile passes require_face=False: keep the image that sits on the
        source. Deep search face-checks afterwards and still uses a no-face avatar.
        """
        limit = limit or self.s.source_photo_limit
        pages = self._pages(sources, max_pages or self.s.source_photo_pages)
        if not pages:
            return [], False
        per_page = await asyncio.gather(*(self._images_of(s, name) for s in pages), return_exceptions=True)
        found: list[_Found] = []
        seen: set[str] = set()
        for s, imgs in zip(pages, per_page):
            if isinstance(imgs, BaseException):
                continue
            for img, named in imgs:
                if (key := image_key(img)) not in seen:
                    seen.add(key)
                    found.append(_Found(img, s, named or _trusted(s, name)))
        # Trusted pages first: select_photos takes its reference face from the first trusted one.
        found.sort(key=lambda f: not f.trusted)
        if not found:
            return [], False

        def photo(f: _Found, score: float | None = None) -> ClientPhoto:
            return ClientPhoto(url=f.url, pageUrl=f.page.url, title=f.page.title or None, fromSource=True, faceMatch=score)

        if not require_face or not (self.faces and await self.faces.ready()):
            # Unchecked: only the person's own pages, whose image is most likely them.
            return [photo(f) for f in found if f.trusted][:limit], False
        faces = await self.faces.faces_in([f.url for f in found])
        kept = sorted(select_photos(faces, [f.trusted for f in found], self.s.face_gallery_threshold), key=lambda t: -t[1])
        if kept:
            uniq = distinct_pictures([faces[i] for i, _ in kept])
            return [photo(found[kept[j][0]], kept[j][1]) for j in uniq][:limit], True
        # No reference face: a single face on the person's own page, unverified.
        single = [i for i, (f, fs) in enumerate(zip(found, faces)) if f.trusted and fs and len(fs) == 1]
        return [photo(found[single[j]]) for j in distinct_pictures([faces[i] for i in single])][:limit], False

    async def distinct(self, photos: list[ClientPhoto]) -> list[ClientPhoto]:
        """Drops photos that are the same picture as an earlier one under another URL."""
        return [photos[i] for i in await distinct_urls(self.faces, [p.url for p in photos])]
