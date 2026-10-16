"""Who is this? The model's answer from its own knowledge, by name or by photo,
plus the photo search: identify the people in a photo, run the regular deep
search on each name, then rank candidates by how well their photos match the
uploaded face."""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import secrets

from app.core import tracing
from app.core.client_format import _source_type, stable_candidate_id
from app.config import Settings
from app.media.faces import FaceVerifier, cosine
from app.core.llm import LLMProvider, RecalledCandidate
from app.core.models import (
    ClientSource,
    FeCandidate,
    IdentifiedPerson,
    IdentifyRequest,
    IdentifyResponse,
    ImageSearchRequest,
    Job,
    JobStatus,
    Photo,
    PhotoWarning,
    QueryLog,
    ScoreBreakdown,
    SearchInput,
    SearchResult,
    SocialProfile,
    SocialSourcesRequest,
    Source,
    StepMeta,
    utcnow,
)
from app.search.orchestrator import ProfileSearchOrchestrator, _decide_status
from app.media.source_photos import SourcePhotoFinder
from app.search.sources import SourceLookup
from app.core.urls import classify_source, image_key, link_key, profile_root, social_platform
from app.infra.fetcher import PageFetcher
from app.infra.search_adapter import SearchAdapter

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5_000_000
_PLAIN_URL = re.compile(r"^https?://[^\s\[\]<>\"']+$")

# Client-reported face pre-check codes: they only lower confidence, never raise it.
PHOTO_WARNING_PENALTY: dict[str, float] = {"no_face": 0.6, "multiple_faces": 0.8, "face_too_small": 0.85}
_PHOTO_WARNING_HINTS = {
    "no_face": "No face was detected in the photo; rely on other visual context and lower confidence accordingly.",
    "multiple_faces": "The photo shows several people; identify the most prominent face.",
    "face_too_small": "The face is small and low detail; only return candidates you can support.",
}


class IdentifyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def parse_photo_warnings(raw: str | None) -> list[PhotoWarning] | None:
    """"a,b" -> deduped list in canonical order; an unknown code -> None (bad request)."""
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if any(p not in PHOTO_WARNING_PENALTY for p in parts):
        return None
    return [w for w in PHOTO_WARNING_PENALTY if w in parts]  # type: ignore[misc]


def warning_factor(warnings: list[str]) -> float:
    f = 1.0
    for w in warnings:
        f *= PHOTO_WARNING_PENALTY.get(w, 1.0)
    return f


def to_identified(rc: RecalledCandidate, factor: float = 1.0) -> IdentifiedPerson:
    q = " ".join((rc.imageSearchQuery or "").split())
    if not q or "http" in q or "[" in q or len(q) > 80:
        q = rc.displayName
    return IdentifiedPerson(
        displayName=" ".join(rc.displayName.split()),
        headline=rc.headline,
        confidence=round(max(0.0, min(1.0, rc.confidence * factor)), 2),
        disambiguationNote=rc.disambiguationNote,
        nationality=rc.nationality,
        occupation=rc.occupation,
        knownFor=rc.knownFor,
        aliases=[a for a in rc.aliases if isinstance(a, str) and a.strip()],
        sourceUrls=[u for u in rc.sourceUrls if isinstance(u, str) and _PLAIN_URL.match(u) and "](" not in u],
        imageSearchQuery=q,
        profile=rc.profile,
    )


def _cut(v: str | None, n: int = 200) -> str | None:
    return v[:n] if v else None


def _step_meta(*, search_calls: int = 0, llm_calls: int = 0) -> StepMeta:
    return StepMeta(searchCalls=search_calls, llmCalls=llm_calls, **tracing.client_cost(tracing.current_stats()))


def _result_stats(**counts: int) -> dict:
    return {**counts, **tracing.client_cost(tracing.current_stats())}


class IdentifyService:
    def __init__(
        self,
        llm: LLMProvider | None,
        sources: SourceLookup | None = None,
        settings: Settings | None = None,
        source_photos: SourcePhotoFinder | None = None,
    ):
        self.llm = llm
        self.sources = sources
        self.s = settings or Settings()
        self.source_photos = source_photos

    def _ready(self) -> LLMProvider:
        if self.llm is None:
            raise IdentifyError(503, "LLM provider is not configured")
        return self.llm

    async def by_name(self, req: IdentifyRequest) -> IdentifyResponse:
        llm = self._ready()
        with tracing.span("identify", input={"query": req.query, "filters": req.filters}, tags=["identify"]):
            try:
                out = await llm.recall(req.query, req.filters)
            except Exception as e:
                raise IdentifyError(502, f"identification failed: {str(e)[:200]}") from e
            people = sorted((to_identified(c) for c in out.candidates if c.displayName.strip()), key=lambda p: -p.confidence)
            res = IdentifyResponse(candidates=people, meta=StepMeta(llmCalls=1))
            if req.withSources:
                await self.attach_sources(res, req.filters)
            res.meta = _step_meta(search_calls=res.meta.searchCalls, llm_calls=res.meta.llmCalls)
            return res

    async def by_image(
        self,
        data: bytes,
        mime: str,
        filters: dict[str, str],
        caption: str | None = None,
        warnings: list[PhotoWarning] | None = None,
        with_sources: bool = False,
    ) -> IdentifyResponse:
        llm = self._ready()
        warnings = warnings or []
        uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        with tracing.span(
            "identify-image",
            input={"caption": caption, "filters": filters, "warnings": warnings},
            tags=["identify", "image"],
        ):
            try:
                out = await llm.recall_image(uri, filters, caption, [_PHOTO_WARNING_HINTS[w] for w in warnings])
            except Exception as e:
                raise IdentifyError(502, f"identification failed: {str(e)[:200]}") from e
            factor = warning_factor(warnings)
            people = sorted((to_identified(c, factor) for c in out.candidates if c.displayName.strip()), key=lambda p: -p.confidence)
            res = IdentifyResponse(candidates=people, photoWarnings=warnings, meta=StepMeta(llmCalls=1))
            if with_sources:
                await self.attach_sources(res, filters)
            res.meta = _step_meta(search_calls=res.meta.searchCalls, llm_calls=res.meta.llmCalls)
            return res

    async def attach_sources(self, res: IdentifyResponse, filters: dict[str, str]) -> None:
        """Social profiles + photos for the likely people. A failed lookup leaves that
        person without them and adds a warning; it never fails the identification."""
        if self.sources is None:
            res.warnings.append("search is not configured: no social profiles or photos")
            return
        people = [p for p in res.candidates if p.confidence >= self.s.image_min_confidence][: self.s.identify_sources_people]
        src = self.sources

        limit = max(1, self.s.initial_photos)

        async def one(p: IdentifiedPerson) -> int:
            calls = 0
            try:
                social = await src.social(
                    SocialSourcesRequest(
                        name=p.displayName[:120],
                        aliases=p.aliases[:10],
                        occupation=_cut(p.occupation),
                        headline=_cut(p.headline),
                        knownFor=_cut(p.knownFor),
                        filters=filters,
                    )
                )
                p.socialSources = social.sources
                calls += social.meta.searchCalls
            except Exception as e:
                res.warnings.append(f"{p.displayName}: social lookup failed ({str(e)[:120]})")
            # Photos from the person's own pages first: free, and the first is the profile photo.
            if self.source_photos:
                # Unverified links from the model's memory too: Wikipedia articles mostly.
                pages = p.socialSources + [ClientSource(url=u, type="web", verified=False) for u in p.sourceUrls]
                try:
                    p.photos, _ = await self.source_photos.find(p.displayName, pages, limit=limit)
                except Exception as e:
                    log.warning("source photos for %s failed: %s", p.displayName, e)
            if len(p.photos) < limit:
                try:
                    # Extra results: some are the source photo again under another URL.
                    found = await src.images(ImageSearchRequest(query=p.imageSearchQuery, name=p.displayName, filters=filters, limit=limit * 2))
                    calls += found.meta.searchCalls
                    have = {image_key(ph.url) for ph in p.photos}
                    pool = p.photos + [ph for ph in found.photos if image_key(ph.url) not in have]
                    if self.source_photos:
                        pool = await self.source_photos.distinct(pool)
                    p.photos = pool[:limit]
                except Exception as e:
                    res.warnings.append(f"{p.displayName}: image search failed ({str(e)[:120]})")
            return calls

        res.meta.searchCalls += sum(await asyncio.gather(*(one(p) for p in people)))


_UPLOAD_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
LENS_NOTE = "Named by pages that Google Lens found for this photo."


class ImagePeopleSearch:
    def __init__(
        self,
        settings: Settings,
        identify: IdentifyService,
        orch: ProfileSearchOrchestrator,
        faces: FaceVerifier | None,
        search: SearchAdapter | None = None,
        fetcher: PageFetcher | None = None,
        llm: LLMProvider | None = None,
        minio=None,
    ):
        self.s = settings
        self.identify = identify
        self.orch = orch
        self.faces = faces
        self.search = search
        self.fetcher = fetcher
        self.llm = llm
        self._minio = minio

    def minio(self):
        if self._minio is None:
            from minio_handler import get_minio

            self._minio = get_minio()
        return self._minio

    async def run(self, job: Job, data: bytes, mime: str, caption: str | None, warnings: list[PhotoWarning]) -> None:
        job.touch("identifying", JobStatus.running)
        try:
            job.result = await asyncio.wait_for(self._pipeline(job, data, mime, caption, warnings), timeout=self.s.job_timeout_s + self.s.llm_timeout_s)
            job.touch("complete", JobStatus.complete)
        except asyncio.TimeoutError:
            job.error = "photo search timed out"
            job.touch("failed", JobStatus.failed)
        except Exception as e:
            log.exception("photo search %s failed", job.searchId)
            job.error = f"{type(e).__name__}: {str(e)[:300]}"
            job.touch("failed", JobStatus.failed)

    async def _pipeline(self, job: Job, data: bytes, mime: str, caption: str | None, warnings: list[PhotoWarning]) -> SearchResult:
        with tracing.span(
            "image-search",
            input={"filters": job.input.filters, "caption": caption, "warnings": warnings},
            tags=["image-search"],
            session_id=job.searchId,
        ):
            ident = await self.identify.by_image(data, mime, job.input.filters, caption, warnings)
            job.extra["identified"] = ident.candidates
            job.extra["photoWarnings"] = warnings
            people = [p for p in ident.candidates if p.confidence >= self.s.image_min_confidence][: self.s.image_search_people]
            if not people:
                return await self._lens_search(job, data, mime, warnings)

            job.touch("searching")
            subs = [Job(searchId=f"{job.searchId}-{i}", input=SearchInput(name=p.displayName, filters=job.input.filters)) for i, p in enumerate(people, 1)]
            await asyncio.gather(*(self.orch.run(j) for j in subs))
            job.toolCalls.extend(tc for j in subs for tc in j.toolCalls)
            # Cached results are shared with name searches: copy before re-ranking.
            done = [j.result.model_copy(deep=True) for j in subs if j.status == JobStatus.complete and j.result]
            if not done:
                raise RuntimeError(subs[0].error or "search failed")

            job.touch("verifying")
            candidates: list[FeCandidate] = []
            seen: set[str] = set()
            for r in done:
                for c in r.candidates:
                    if (cid := stable_candidate_id(c)) not in seen:
                        seen.add(cid)
                        candidates.append(c)
            factor = warning_factor(warnings)
            for c in candidates:
                c.confidenceScore = round(c.confidenceScore * factor, 3)
            await self._match_faces(data, candidates)
            thr = self.s.face_match_threshold
            candidates.sort(key=lambda c: (not (c.queryFaceMatch is not None and c.queryFaceMatch >= thr), -c.confidenceScore))

            status, question = _decide_status(people[0].displayName, candidates)
            if candidates and candidates[0].queryFaceMatch is not None and candidates[0].queryFaceMatch >= thr:
                if not any(c.queryFaceMatch is not None and c.queryFaceMatch >= thr for c in candidates[1:]):
                    status, question = "resolved", None
            fresh = [r for r in done if not r.stats.get("cacheHit")]
            return SearchResult(
                searchId=job.searchId,
                mode="image",
                status=status,  # type: ignore[arg-type]
                input=job.input,
                candidates=candidates,
                sources=[s for r in done for s in r.sources],
                queries=[q for r in done for q in r.queries],
                clarificationQuestion=question,
                warnings=[w for r in done for w in r.warnings],
                completedAt=utcnow(),
                stats=_result_stats(
                    searchCalls=sum(r.stats.get("searchCalls", 0) for r in fresh),
                    llmCalls=1 + sum(r.stats.get("llmCalls", 0) for r in fresh),
                ),
            )

    async def _lens_search(self, job: Job, data: bytes, mime: str, warnings: list[PhotoWarning]) -> SearchResult:
        """Nobody recognised: find the photo on the web (Google Lens through Serper) and
        let the model read who those pages say it shows. The face (largest one, with
        some margin) is searched rather than the whole photo, so pages matching only
        the clothes or the background drop out."""
        notes = ["No public person recognised in the photo."]
        llm_calls, search_calls = 1, 0

        def result(candidates: list[FeCandidate], sources: list[Source], status: str = "no_match") -> SearchResult:
            return SearchResult(
                searchId=job.searchId,
                mode="image",
                status=status,  # type: ignore[arg-type]
                input=job.input,
                candidates=candidates,
                sources=sources,
                queries=[QueryLog(query="google lens", kind="image", round="discovery", purpose="reverse image search")] if search_calls else [],
                warnings=notes,
                completedAt=utcnow(),
                stats=_result_stats(searchCalls=search_calls, llmCalls=llm_calls),
            )

        if not (self.search and self.llm and self.search.has_lens()):
            notes.append("Reverse image search is not configured.")
            return result([], [])
        minio = self.minio()
        if not minio.is_available():
            notes.append("Reverse image search needs MinIO to host the photo; MinIO is not configured.")
            return result([], [])

        job.touch("lens")
        crop = await self.faces.main_face_crop(data, self.s.lens_face_pad) if self.faces and await self.faces.ready() else None
        image, ext, ctype = (crop, ".jpg", "image/jpeg") if crop else (data, _UPLOAD_EXT.get(mime, ".jpg"), mime)
        job.extra["lensInput"] = "face" if crop else "photo"
        url, _, ok = await asyncio.to_thread(minio.upload_bytes, image, f"lens-{job.searchId}-{secrets.token_hex(4)}", ext, ctype)
        if not ok:
            notes.append("Could not upload the photo for reverse image search.")
            return result([], [])
        job.extra["lensImageUrl"] = url
        hits = await self.search.lens(url, job.toolCalls)
        search_calls = 1

        now = utcnow()
        sources: list[Source] = []
        seen: set[str] = set()
        for h in hits:
            if (key := link_key(h.url)) in seen:
                continue
            seen.add(key)
            sources.append(
                Source(
                    id=f"lens_{len(sources) + 1}",
                    title=h.title,
                    url=h.url,
                    snippet=h.snippet,
                    sourceType=classify_source(h.url),  # type: ignore[arg-type]
                    provider=h.provider,
                    query="google lens",
                    retrievedAt=now,
                    rank=round(1 - len(sources) / 100, 2),
                    image=h.image_url,
                )
            )
        job.extra["lensResults"] = [
            ClientSource(url=s.url, title=s.title, type=_source_type(s), snippet=s.snippet or None, verified=True) for s in sources
        ]
        if not sources:
            notes.append("Google Lens found no page showing this photo.")
            return result([], [])

        job.touch("reading")
        to_read = sources[: self.s.lens_pages_to_read]
        fetched = await asyncio.gather(*(self.fetcher.fetch(s.url) for s in to_read)) if self.fetcher else []
        pages_by_id = {s.id: p for s, p in zip(to_read, fetched)}
        payload = []
        for s in sources:
            page = pages_by_id.get(s.id)
            text = ""
            if page and page.status == "fetched":
                s.contentStatus = "fetched"
                text = (page.description + "\n" + page.text).strip()[:1500]
            payload.append({"url": s.url, "title": (page.title if page and page.title else s.title), "source": s.snippet, "text": text})
        try:
            out = await self.llm.lens_people(payload)
            llm_calls += 1
        except Exception as e:
            log.warning("lens pages for %s: model failed: %s", job.searchId, e)
            notes.append("Could not read the pages Google Lens found.")
            return result([], sources)

        by_key = {link_key(s.url): s for s in sources}
        factor = warning_factor(warnings)
        candidates: list[FeCandidate] = []
        for lp in out.candidates if out.identified else []:
            if not lp.displayName.strip() or lp.confidence < self.s.image_min_confidence:
                continue
            own = list({s.id: s for u in lp.profileUrls if (s := by_key.get(link_key(u)))}.values())
            if not own:
                continue
            socials: dict[str, SocialProfile] = {}
            for s in own:
                root = profile_root(s.url) if social_platform(s.url) else None
                if root and root not in socials:
                    socials[root] = SocialProfile(platform=social_platform(root) or "other", url=root, sourceIds=[s.id], confidence=0.6)
            photos = [Photo(url=s.image, pageUrl=s.url, title=s.title, sourceIds=[s.id]) for s in own if s.image]
            cand = FeCandidate(
                candidateId=f"cand_{len(candidates) + 1}",
                displayName=lp.displayName.strip(),
                headline=lp.headline,
                confidenceScore=round(max(0.0, min(0.9, lp.confidence)) * factor, 3),
                scoreBreakdown=ScoreBreakdown(
                    nameMatch=0.0,
                    distinguishingAttributes=0.0,
                    sourceQuality=0.0,
                    crossSourceAgreement=min(1.0, len(own) / 3),
                    prominence=0.0,
                    filterMatch=0.0,
                    conflictPenalty=0.0,
                ),
                matchNote=LENS_NOTE,
                occupation=lp.occupation,
                knownFor=lp.knownFor,
                avatarHint=photos[0].url if photos else None,
                fields={},
                socialProfiles=list(socials.values()),
                sources=own,
                photos=photos,
                narrative=[p for p in lp.profile if p.strip()],
                narrativeSource="llm" if lp.profile else None,
            )
            cand.imageSearchQuery = cand.displayName
            candidates.append(cand)

        await self._match_faces(data, candidates)
        thr = self.s.face_match_threshold
        candidates.sort(key=lambda c: (not (c.queryFaceMatch is not None and c.queryFaceMatch >= thr), -c.confidenceScore))
        if candidates:
            job.touch("profiles")
            search_calls += await self._add_social_profiles(candidates, job.input.filters, notes)
        if not candidates:
            notes.append("The pages Google Lens found do not say who is in the photo.")
            return result([], sources)
        return result(candidates, sources, "resolved" if len(candidates) == 1 else "ambiguous")

    async def _add_social_profiles(self, candidates: list[FeCandidate], filters: dict[str, str], notes: list[str]) -> int:
        """Lens pages are mostly articles; the person's own accounts, found by the name
        those pages gave, are what deep search starts from. Returns the search calls."""
        src = self.identify.sources
        if src is None:
            return 0
        now = utcnow()

        async def one(c: FeCandidate) -> int:
            try:
                found = await src.social(
                    SocialSourcesRequest(
                        name=c.displayName[:120],
                        occupation=_cut(c.occupation),
                        headline=_cut(c.headline),
                        knownFor=_cut(c.knownFor),
                        filters=filters,
                    )
                )
            except Exception as e:
                notes.append(f"{c.displayName}: social lookup failed ({str(e)[:120]})")
                return 0
            have = {link_key(s.url) for s in c.sources}
            for s in found.sources:
                if link_key(s.url) in have:
                    continue
                source = Source(
                    id=f"social_{c.candidateId}_{len(c.sources) + 1}",
                    title=s.title,
                    url=s.url,
                    snippet=s.snippet or "",
                    sourceType="social_profile",
                    provider="serper",
                    query=c.displayName,
                    retrievedAt=now,
                    rank=1.0,
                )
                c.sources.append(source)
                c.socialProfiles.append(SocialProfile(platform=s.type, url=s.url, sourceIds=[source.id], confidence=0.6))
            return found.meta.searchCalls

        return sum(await asyncio.gather(*(one(c) for c in candidates[: self.s.identify_sources_people])))

    async def _match_faces(self, data: bytes, candidates: list[FeCandidate]) -> None:
        """queryFaceMatch = best similarity between the uploaded face (largest one) and the candidate's photos."""
        if not self.faces or not candidates or not await self.faces.ready():
            return
        found = await self.faces.faces_in_bytes(data)
        if not found:
            return
        ref = max(found, key=lambda f: f.box[2] * f.box[3])

        async def one(c: FeCandidate) -> None:
            urls = [p.url for p in c.photos] or ([c.avatarHint] if c.avatarHint else [])
            if not urls:
                return
            scores = [cosine(f.embedding, ref.embedding) for fs in await self.faces.faces_in(urls) for f in fs or []]
            if scores:
                c.queryFaceMatch = round(max(scores), 3)

        await asyncio.gather(*(one(c) for c in candidates))
