"""More photos of one candidate, on demand.

A search result carries one photo per person. When the user wants more, one
image search runs pinned to the candidate's own profile pages, and only images
whose face matches the reference photo are returned."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.media.faces import FaceVerifier, cosine, distinct_urls, select_photos
from app.core.models import ClientPhoto, MorePhotosMeta, MorePhotosRequest, MorePhotosResponse, ToolCallLog
from app.infra.search_locale import locale_for
from app.core.urls import image_key, image_on_sources, name_match, source_image_query


class PhotoError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class PhotoService:
    def __init__(self, settings: Settings, search: Any, faces: FaceVerifier | None):
        self.s = settings
        self.search = search
        self.faces = faces

    async def more(self, req: MorePhotosRequest) -> MorePhotosResponse:
        limit = min(req.limit, self.s.more_photos_max)
        check = bool(self.faces and await self.faces.ready())
        ref = None
        if check and req.referenceUrl:
            found = (await self.faces.faces_in([req.referenceUrl]))[0]
            if not found:
                raise PhotoError(422, "no face found in the reference photo")
            ref = max(found, key=lambda f: f.box[2] * f.box[3])

        logs: list[ToolCallLog] = []
        locale = locale_for(req.filters, (self.s.search_locale_gl, self.s.search_locale_hl), req.name or "")
        query = source_image_query(req.name or "", req.profileUrls) or req.query
        try:
            hits, _ = await self.search.search(query, "image", logs, locale)
        except RuntimeError as e:
            raise PhotoError(503, f"image search unavailable: {e}")

        seen = {image_key(u) for u in [req.referenceUrl, *req.exclude] if u}
        names = [n for n in [req.name, *req.aliases] if n and n.strip()]
        pool: list[ClientPhoto] = []
        for h in sorted(hits, key=lambda h: h.position):
            if not h.image_url or not h.image_url.startswith(("http://", "https://")):
                continue
            own = bool(req.profileUrls) and image_on_sources(h.url, h.image_url, req.profileUrls, h.title or "")
            if req.profileUrls:
                if not own:
                    continue
            elif names and max(name_match(n, f"{h.title} {h.url}") for n in names) < 0.6:
                continue
            key = image_key(h.image_url)
            if key in seen:
                continue
            seen.add(key)
            pool.append(ClientPhoto(url=h.image_url, pageUrl=h.url, title=h.title or None, fromSource=own, fallbackUrl=h.image_fallback))
            if len(pool) >= (limit * 2 if check else limit):
                break
        # Images from the person's own pages first: they give the reference face.
        pool.sort(key=lambda p: not p.fromSource)

        photos = pool
        if check and pool:
            found, used = await self.faces.faces_or_fallback([p.url for p in pool], [p.fallbackUrl for p in pool])
            pool = [p if u == p.url else p.model_copy(update={"url": u, "fallbackUrl": None}) for p, u in zip(pool, used)]
            photos = []
            if ref is None:
                # No known photo of the person: the first single face from their own pages,
                # else the face most results agree on.
                for i, score in select_photos(found, [p.fromSource for p in pool], self.s.face_gallery_threshold):
                    photos.append(pool[i].model_copy(update={"faceMatch": score}))
                if not photos:
                    # No two results agree: offer the name-matched single-face images, unverified.
                    photos = [p for p, fs in zip(pool, found) if fs and len(fs) == 1]
                    check = False
            else:
                for p, fs in zip(pool, found):
                    score = max((cosine(f.embedding, ref.embedding) for f in fs or []), default=-1.0)
                    if score >= self.s.face_gallery_threshold:
                        photos.append(p.model_copy(update={"faceMatch": round(score, 3)}))
            photos.sort(key=lambda p: -(p.faceMatch or 0))
            # Not the same picture as one the client already shows, nor twice.
            shown = [u for u in [req.referenceUrl, *req.exclude] if u]
            keep = set(await distinct_urls(self.faces, shown + [p.url for p in photos]))
            photos = [p for i, p in enumerate(photos, len(shown)) if i in keep]
        return MorePhotosResponse(
            photos=photos[:limit],
            meta=MorePhotosMeta(searchCalls=sum(1 for c in logs if not c.cached and not c.error), faceChecked=check),
        )
