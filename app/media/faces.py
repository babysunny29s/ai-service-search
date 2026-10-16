"""Face checks for candidate photos (OpenCV YuNet detector + SFace embeddings, CPU).

A photo is kept only when it shows a face and that face matches the person:
the reference is a trusted photo (Wikipedia / knowledge panel / the person's
own source) with exactly one face; without one, the face that several photos
agree on. Without any reference no photo can be confirmed, so none is kept.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.cache import TTLCache
from app.infra.fetcher import _is_public_host

log = logging.getLogger(__name__)

MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
MAX_IMAGE_BYTES = 5_000_000
MAX_SIDE = 1024
MIN_FACE_PX = 32
# A face smaller than this share of the image's shorter side is a bystander.
MIN_FACE_SHARE = 0.10
DETECT_SCORE = 0.8
_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF")


# Same picture (another URL, size, crop or encoding): the aligned face crops look
# alike (dHash distance), or the faces are near-identical. Different photos of one
# person score about 0.4-0.85 — those stay in the gallery.
SAME_PICTURE_BITS = 12
SAME_PICTURE_COSINE = 0.95


@dataclass
class Face:
    box: tuple[int, int, int, int]
    score: float
    embedding: "object"  # unit-length numpy vector
    crop_hash: int | None = None  # dHash of the aligned 112x112 face crop


def cosine(a, b) -> float:
    return float((a * b).sum())


def same_picture(a: Face, b: Face) -> bool:
    bits = (a.crop_hash ^ b.crop_hash).bit_count() if a.crop_hash is not None and b.crop_hash is not None else None
    if bits is not None and bits <= SAME_PICTURE_BITS:
        return True
    return cosine(a.embedding, b.embedding) >= SAME_PICTURE_COSINE


def distinct_pictures(faces: list[list[Face] | None]) -> list[int]:
    """Indexes of the photos to keep, in order: a photo whose main face is the same
    picture as an earlier photo's is dropped. Photos without a face are kept."""
    kept: list[int] = []
    seen: list[Face] = []
    for i, fs in enumerate(faces):
        if fs:
            main = max(fs, key=lambda f: f.box[2] * f.box[3])
            if any(same_picture(main, s) for s in seen):
                continue
            seen.append(main)
        kept.append(i)
    return kept


async def distinct_urls(faces: "FaceVerifier | None", urls: list[str]) -> list[int]:
    """distinct_pictures over image URLs (faces are cached); all kept without the face model."""
    if not urls or faces is None or not await faces.ready():
        return list(range(len(urls)))
    return distinct_pictures(await faces.faces_in(urls))


def _dhash(crop) -> int:
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def select_photos(faces: list[list[Face] | None], trusted: list[bool], threshold: float) -> list[tuple[int, float]]:
    """Indexes of photos showing the person, with their best similarity.
    `faces[i]` is None when photo i could not be read; `trusted[i]` marks photos
    from the person's own sources, best first."""
    single = [i for i, f in enumerate(faces) if f and len(f) == 1]
    ref: int | None = next((i for i in single if trusted[i]), None)
    if ref is None:
        # The face most other photos agree on, if at least two photos agree.
        best, best_n = None, 0
        for i in single:
            n = sum(1 for j in single if j != i and cosine(faces[i][0].embedding, faces[j][0].embedding) >= threshold)
            if n > best_n:
                best, best_n = i, n
        ref = best
    if ref is None:
        return []
    ref_emb = faces[ref][0].embedding
    out: list[tuple[int, float]] = []
    for i, fs in enumerate(faces):
        if not fs:
            continue
        score = 1.0 if i == ref else max(cosine(f.embedding, ref_emb) for f in fs)
        if score >= threshold:
            out.append((i, round(score, 3)))
    return out


def same_face_groups(refs: list[Face | None], threshold: float) -> list[list[int]]:
    """Groups (2+ members, ascending indexes) of candidates whose reference faces are the same person."""
    parent = list(range(len(refs)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(refs):
        for j in range(i + 1, len(refs)):
            b = refs[j]
            if a is not None and b is not None and cosine(a.embedding, b.embedding) >= threshold:
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(refs)):
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


class FaceVerifier:
    def __init__(self, model_dir: str, user_agent: str, timeout_s: float = 10, cache_ttl_s: float = 21600):
        self.model_dir = Path(model_dir)
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        # Wrapped in a tuple so an unreadable image (None) is cached too.
        self.cache: TTLCache[tuple[list[Face] | None]] = TTLCache(cache_ttl_s, max_items=3000)
        self._lock = threading.Lock()
        self._models = None
        self._failed = False

    def _paths(self) -> dict[str, Path]:
        return {name: self.model_dir / name for name in MODEL_URLS}

    async def ready(self) -> bool:
        if self._models is not None:
            return True
        if self._failed:
            return False
        try:
            missing = [(n, p) for n, p in self._paths().items() if not p.exists()]
            if missing:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                    for name, path in missing:
                        resp = await client.get(MODEL_URLS[name])
                        resp.raise_for_status()
                        path.write_bytes(resp.content)
            self._models = await asyncio.to_thread(self._load)
            return True
        except Exception as e:  # noqa: BLE001 - any failure disables the check, never the search
            log.warning("face verification disabled: %s", e)
            self._failed = True
            return False

    def _load(self):
        import cv2

        paths = self._paths()
        detector = cv2.FaceDetectorYN.create(str(paths["face_detection_yunet_2023mar.onnx"]), "", (320, 320), DETECT_SCORE, 0.3, 50)
        recognizer = cv2.FaceRecognizerSF.create(str(paths["face_recognition_sface_2021dec.onnx"]), "")
        return detector, recognizer

    async def faces_in(self, urls: list[str], concurrency: int = 8) -> list[list[Face] | None]:
        sem = asyncio.Semaphore(concurrency)

        async def one(url: str) -> list[Face] | None:
            if (hit := self.cache.get(url)) is not None:
                return hit[0]
            async with sem:
                data = await self._download(url)
            faces = await asyncio.to_thread(self._detect, data) if data else None
            self.cache.set(url, (faces,))
            return faces

        return list(await asyncio.gather(*(one(u) for u in urls)))

    async def faces_in_bytes(self, data: bytes) -> list[Face] | None:
        """Faces in an uploaded image (call `ready()` first)."""
        if len(data) > MAX_IMAGE_BYTES or not data.startswith(_MAGIC):
            return None
        return await asyncio.to_thread(self._detect, data)

    async def main_face_crop(self, data: bytes, pad: float = 1.75) -> bytes | None:
        """JPEG of the largest face, its box widened `pad` times around the centre
        (call `ready()` first). None when the image shows no face."""
        if len(data) > MAX_IMAGE_BYTES or not data.startswith(_MAGIC):
            return None
        return await asyncio.to_thread(self._crop_main_face, data, pad)

    def _crop_main_face(self, data: bytes, pad: float) -> bytes | None:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = self._detect(data)
        if not faces:
            return None
        h, w = img.shape[:2]
        scale = max(h, w) / MAX_SIDE if max(h, w) > MAX_SIDE else 1.0
        x, y, fw, fh = (v * scale for v in max(faces, key=lambda f: f.box[2] * f.box[3]).box)
        cx, cy, half_w, half_h = x + fw / 2, y + fh / 2, fw * pad / 2, fh * pad / 2
        x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
        x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
        if x1 - x0 < MIN_FACE_PX or y1 - y0 < MIN_FACE_PX:
            return None
        ok, buf = cv2.imencode(".jpg", img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes() if ok else None

    async def faces_or_fallback(self, urls: list[str], fallbacks: list[str | None]) -> tuple[list[list[Face] | None], list[str]]:
        """Faces per image, retrying with the fallback URL (Google's thumbnail) when the
        original does not serve an image. Also returns the URL that worked for each."""
        found = await self.faces_in(urls)
        used = list(urls)
        retry = [i for i, f in enumerate(found) if f is None and fallbacks[i] and fallbacks[i] != urls[i]]
        if retry:
            alt = await self.faces_in([fallbacks[i] for i in retry])  # type: ignore[misc]
            for i, faces in zip(retry, alt):
                if faces is not None:
                    found[i], used[i] = faces, fallbacks[i]  # type: ignore[assignment]
        return found, used

    async def _download(self, url: str) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=False, headers={"User-Agent": self.user_agent}) as client:
                target = url
                for _ in range(4):
                    if urlsplit(target).scheme not in ("http", "https") or not await asyncio.to_thread(_is_public_host, target):
                        return None
                    async with client.stream("GET", target) as resp:
                        if resp.is_redirect and resp.headers.get("location"):
                            target = urljoin(target, resp.headers["location"])
                            continue
                        if resp.status_code != 200:
                            return None
                        buf = bytearray()
                        async for chunk in resp.aiter_bytes():
                            buf.extend(chunk)
                            if len(buf) > MAX_IMAGE_BYTES:
                                return None
                        data = bytes(buf)
                        return data if data.startswith(_MAGIC) else None
        except httpx.HTTPError:
            return None
        return None

    def _detect(self, data: bytes) -> list[Face] | None:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
        min_side = max(MIN_FACE_PX, MIN_FACE_SHARE * min(h, w))
        with self._lock:
            detector, recognizer = self._models
            detector.setInputSize((w, h))
            _, found = detector.detect(img)
            faces: list[Face] = []
            for row in found if found is not None else []:
                x, y, fw, fh = (int(v) for v in row[:4])
                if min(fw, fh) < min_side:
                    continue
                crop = recognizer.alignCrop(img, row)
                emb = recognizer.feature(crop).flatten()
                emb = emb / (np.linalg.norm(emb) or 1.0)
                faces.append(Face(box=(x, y, fw, fh), score=float(row[-1]), embedding=emb, crop_hash=_dhash(crop)))
        return faces
