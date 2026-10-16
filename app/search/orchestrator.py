"""Profile Search Orchestrator (cost-optimized).

Typical cost per search: 1-3 search calls and 1-2 LLM calls.
1. Discovery: `"name"` first; more discovery queries only if the search
   engine returned no knowledge panel for that name.
2. Read a few top pages, send compact excerpts to ONE LLM `analyze` call
   (extraction + identity grouping + profile wording).
3. Enrichment only for candidates that are still thin and not public
   figures: one targeted query each, then one more `analyze` call.
4. The application verifies evidence and computes scores.
Results are cached per (name, filters).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.core import tracing
from app.core.cache import TTLCache
from app.core.client_format import client_sources, official_links, person_wiki
from app.config import Settings
from app.media.faces import Face, FaceVerifier, cosine, distinct_urls, same_face_groups, select_photos
from app.infra.fetcher import FetchedPage, PageFetcher, excerpt
from app.core.llm import KnowledgePlan, LLMProvider, RecalledCandidate
from app.core.models import FeCandidate, Job, JobStatus, Photo, QueryLog, SearchResult, SocialProfile, Source, utcnow
from app.search.planner import PlannedQuery, discovery_queries, enrichment_queries, social_query
from app.search.ranking import build_sources
from app.search.resolution import (
    IDENTIFYING_FIELDS,
    KNOWN_CONFIDENCE_CAP,
    KNOWN_NOTE,
    RECALL_CONFIDENCE_CAP,
    UNCONFIRMED_NOTE,
    CandidateDraft,
    apply_profile,
    build_candidate,
    candidate_from_recall,
    split_aliases,
    candidate_seeds,
    groups_from_analysis,
)
from app.infra.search_adapter import SearchAdapter, SearchHit
from app.infra.search_locale import locale_for
from app.media.source_photos import SourcePhotoFinder
from app.core.urls import can_fetch, image_key, fold, handle_matches_name, identity_words, link_key, looks_like_fan_account, mentions_identity, name_match, normalize_url, profile_root, social_platform, spacing_variant

log = logging.getLogger(__name__)

SNIPPET_CHARS = 300
# Knowledge-panel attributes about relatives or private contact are never passed on.
_PRIVATE_KG_ATTR = re.compile(r"spouse|partner|children|child|parent|sibling|relative|address|net worth|\b(vợ|chồng|con|bố|mẹ)\b", re.I)


@dataclass
class _State:
    name: str
    sources: list[Source] = field(default_factory=list)
    pages: dict[str, FetchedPage] = field(default_factory=dict)
    queries: list[QueryLog] = field(default_factory=list)
    done: set[str] = field(default_factory=set)
    image_hits: list[tuple[str, list[SearchHit]]] = field(default_factory=list)
    kg_images: dict[str, str | None] = field(default_factory=dict)
    filters: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    locale: tuple[str, str] | None = None
    hit_keys: set[str] = field(default_factory=set)


class ProfileSearchOrchestrator:
    def __init__(self, settings: Settings, search: SearchAdapter, fetcher: PageFetcher, llm: LLMProvider | None, faces: FaceVerifier | None = None):
        self.s = settings
        self.search = search
        self.fetcher = fetcher
        self.llm = llm
        self.faces = faces
        self.source_photos = SourcePhotoFinder(settings, fetcher, faces)
        self.results: TTLCache[SearchResult] = TTLCache(settings.result_cache_ttl_s, max_items=1000)

    @staticmethod
    def _cache_key(job: Job) -> tuple:
        return (fold(job.input.name), tuple(sorted((k, fold(v)) for k, v in job.input.filters.items())))

    async def run(self, job: Job) -> None:
        job.touch("planning", JobStatus.running)
        t0 = time.perf_counter()
        key = self._cache_key(job)
        cached = self.results.get(key)
        if cached:
            job.result = cached.model_copy(update={"searchId": job.searchId, "stats": {**cached.stats, "cacheHit": 1}})
            job.touch("complete", JobStatus.complete)
            log.info("job=%s served from result cache", job.searchId)
            return
        with tracing.span(
            "profile-search",
            input={"name": job.input.name, "filters": job.input.filters},
            metadata={"searchId": job.searchId, "strategy": self.s.search_strategy},
            session_id=job.searchId,
        ) as trace:
            try:
                job.result = await asyncio.wait_for(self._pipeline(job), timeout=self.s.job_timeout_s)
                job.touch("complete", JobStatus.complete)
                if not any(c.error for c in job.toolCalls):
                    self.results.set(key, job.result)
            except asyncio.TimeoutError:
                job.error = f"job timeout after {self.s.job_timeout_s}s"
                job.touch("failed", JobStatus.failed)
            except Exception as e:
                log.exception("job %s failed", job.searchId)
                job.error = f"{type(e).__name__}: {str(e)[:300]}"
                job.touch("failed", JobStatus.failed)
            trace.update(output=_trace_output(job), level="ERROR" if job.error else "DEFAULT", status_message=job.error)
        log.info(
            "job=%s status=%s ms=%d search_calls=%d (cached=%d) llm_calls=%d",
            job.searchId,
            job.status.value,
            int((time.perf_counter() - t0) * 1000),
            sum(1 for c in job.toolCalls if not c.cached),
            sum(1 for c in job.toolCalls if c.cached),
            job.result.stats.get("llmCalls", 0) if job.result else 0,
        )

    async def _pipeline(self, job: Job) -> SearchResult:
        if self.llm is None:
            raise RuntimeError("LLM provider is not configured (LLM_API_KEY)")
        st = _State(name=job.input.name, filters=job.input.filters)
        st.locale = locale_for(st.filters, (self.s.search_locale_gl, self.s.search_locale_hl), st.name)

        kp: KnowledgePlan | None = None
        llm_queries: list[PlannedQuery] = []
        if self.s.search_strategy == "llm_first" and self.s.llm_max_calls >= 2:
            job.touch("analyzing")
            kp = await self._ask_model(st)
            if kp is not None:
                # The model decides whether to search and what for; nothing else is searched.
                names = [st.name] + [n for c in kp.candidates for n in (c.displayName, *c.aliases)]
                llm_queries = _model_queries(kp.searchQueries, names, self.s.max_queries, st.done)
                if not llm_queries:
                    drafts = self._from_knowledge(st, kp.candidates, KNOWN_CONFIDENCE_CAP, KNOWN_NOTE)
                    if drafts:
                        st.warnings.append("Answered from model knowledge; no search needed." + (f" ({kp.reason})" if kp.reason else ""))
                    return await self._finish(job, st, drafts, lookup_socials=False)
        model_led = kp is not None
        known_raw = kp.candidates if kp else []

        # Round 1
        if llm_queries:
            await self._search(job, st, llm_queries, {}, "discovery", list(st.filters.values()), self.s.max_fetch_urls)
        else:
            plan = discovery_queries(st.name, self.s.max_queries, st.done, st.filters)
            await self._search(job, st, plan[:1], {}, "discovery", list(st.filters.values()), 0)
            if not _first_query_sufficient(st.sources):
                await self._search(job, st, plan[1:], {}, "discovery", list(st.filters.values()), 0)
            await self._read(st, st.sources, self.s.max_fetch_urls)
        if job.toolCalls and all(c.error for c in job.toolCalls):
            if known_raw:
                st.warnings.append("Search failed; results come from model knowledge only.")
                known = self._from_knowledge(st, known_raw, RECALL_CONFIDENCE_CAP, UNCONFIRMED_NOTE)
                return await self._finish(job, st, known, lookup_socials=False)
            raise RuntimeError("all search tool calls failed: " + (job.toolCalls[0].error or ""))

        job.touch("analyzing")
        try:
            drafts = await self._analyze(st)
        except Exception:
            if not known_raw:
                raise
            known = self._from_knowledge(st, known_raw, RECALL_CONFIDENCE_CAP, UNCONFIRMED_NOTE)
            return await self._finish(job, st, known, lookup_socials=False)

        # Round 2 (search-first only): thin, non-public-figure candidates.
        enrich_plan: list[PlannedQuery] = []
        anchors: dict[str, str] = {}
        seed_values: list[str] = []
        # Enrichment results need another analysis call to be read.
        if not model_led and st.llm_calls < self.s.llm_max_calls:
            for d in self._needs_enrichment(drafts):
                seeds = candidate_seeds(d.candidate)
                have = {sp.platform for sp in d.candidate.socialProfiles}
                qs = enrichment_queries(d.candidate.displayName, seeds, self.s.enrich_queries_per_candidate, st.done, have, include_image=self.s.enrich_images and not enrich_plan)
                for q in qs:
                    anchors[q.query] = next(iter(seeds.values()))
                enrich_plan += qs
                seed_values += list(seeds.values())
        if enrich_plan:
            job.touch("enriching")
            before = len(st.sources)
            await self._search(job, st, enrich_plan, anchors, "enrichment", seed_values, self.s.enrich_fetch_urls)
            if len(st.sources) > before:
                job.touch("analyzing")
                try:
                    drafts = await self._analyze(st)
                except Exception:
                    log.warning("enrichment analysis failed; keeping round-1 candidates")

        if known_raw:
            drafts = _merge_known(drafts, self._from_knowledge(st, known_raw, RECALL_CONFIDENCE_CAP, UNCONFIRMED_NOTE))
        elif not drafts and kp is None and st.llm_calls < self.s.llm_max_calls:
            drafts = await self._recall(st)
        return await self._finish(job, st, drafts, lookup_socials=not model_led)

    async def _finish(
        self, job: Job, st: _State, drafts: list[CandidateDraft], lookup_socials: bool
    ) -> SearchResult:
        job.touch("verifying")
        drafts.sort(key=lambda d: d.candidate.confidenceScore, reverse=True)
        drafts = [d for d in drafts if d.candidate.confidenceScore >= 0.2][: self.s.max_candidates]
        if lookup_socials:
            await self._lookup_socials(job, st, drafts)
        _assign_shared_sources(drafts)
        for i, d in enumerate(drafts, 1):
            d.candidate.candidateId = f"cand_{i}"
        candidates = [d.candidate for d in drafts]
        _make_image_queries_distinct(candidates)
        await self._free_avatars(candidates)
        await self._source_photos(candidates)
        await self._photo_search(job, st, candidates)
        refs = await self._verify_faces(candidates)
        await self._resolve_shared_faces(st, candidates, refs)
        shown = [
            c for c in candidates
            if (any(s.verified for s in client_sources(c)) if self.s.require_source else client_sources(c)) and (not self.s.require_photo or c.photos)
        ]
        if hidden := len(candidates) - len(shown):
            st.warnings.append(f"{hidden} candidate(s) hidden: no search-found source" + (" or no verified photo." if self.s.require_photo else "."))
        candidates = shown
        await asyncio.gather(*(self._order_photos(c) for c in candidates))
        for c in candidates:
            c.officialLinks = official_links(c)
        await self._wiki_profiles(candidates)

        status, question = _decide_status(st.name, candidates)
        return SearchResult(
            searchId=job.searchId,
            status=status,  # type: ignore[arg-type]
            input=job.input,
            candidates=candidates,
            sources=st.sources,
            queries=st.queries,
            clarificationQuestion=question,
            warnings=st.warnings,
            completedAt=utcnow(),
            stats={
                "searchCalls": sum(1 for c in job.toolCalls if not c.cached and not c.error),
                "cachedSearchCalls": sum(1 for c in job.toolCalls if c.cached),
                "llmCalls": st.llm_calls,
                "pagesFetched": sum(1 for p in st.pages.values() if p.status == "fetched"),
                **tracing.client_cost(tracing.current_stats()),
            },
        )

    def _needs_enrichment(self, drafts: list[CandidateDraft]) -> list[CandidateDraft]:
        out = []
        for d in drafts:
            c = d.candidate
            if len(out) >= self.s.enrich_candidates or c.confidenceScore < 0.25:
                break
            if any(s.knowledgeGraph or s.sourceType == "wiki" for s in c.sources):
                continue
            identifying = sum(1 for k in IDENTIFYING_FIELDS if c.fields[k].status == "found")
            if identifying < self.s.enrich_min_identifying and candidate_seeds(c):
                out.append(d)
        return out

    async def _search(
        self,
        job: Job,
        st: _State,
        plan: list[PlannedQuery],
        anchors: dict[str, str],
        round_name: str,
        seeds: list[str],
        fetch_budget: int,
    ) -> list[Source]:
        if not plan:
            return []
        job.touch("searching")
        outs = await asyncio.gather(
            *(self.search.search(p.query, "image" if p.kind == "image" else "web", job.toolCalls, st.locale) for p in plan)
        )
        web_hits: list[SearchHit] = []
        for p, (hits, kgs) in zip(plan, outs):
            st.queries.append(QueryLog(query=p.query, kind=p.kind, round=round_name, purpose=p.purpose))  # type: ignore[arg-type]
            if p.kind == "image":
                st.image_hits.append((anchors.get(p.query, ""), hits))
                continue
            web_hits.extend(hits)
            for kg in kgs:
                if hit := self._kg_hit(st, kg, p.query):
                    web_hits.append(hit)
        st.hit_keys.update(link_key(h.url) for h in web_hits)

        new_sources = build_sources(web_hits, st.name, seeds, {s.url for s in st.sources}, start_index=len(st.sources) + 1)
        st.sources.extend(new_sources)
        for s in st.sources:
            if s.url in st.kg_images:
                s.knowledgeGraph = True
                s.image = s.image or st.kg_images[s.url]
        if fetch_budget:
            await self._read(st, new_sources, fetch_budget)
        return new_sources

    async def _lookup_socials(self, job: Job, st: _State, drafts: list[CandidateDraft]) -> None:
        """One search per strong candidate still missing social profiles. The
        top-ranked profile per platform whose title/snippet carries the
        candidate's full name is attached; posts and fan groups are ignored."""
        profiles_searched = any(q.purpose == "profiles" for q in st.queries)
        targets = [
            d
            for d in drafts
            if d.candidate.confidenceScore >= 0.4
            and len({p.platform for p in d.candidate.socialProfiles}) < self.s.social_min_platforms
            and not (profiles_searched and fold(d.candidate.displayName) == fold(st.name) and not d.candidate.aliases)
        ][: self.s.social_lookup_candidates]
        plan = [
            (d, q)
            for d in targets
            # Accounts usually carry the stage name (HIEUTHUHAI), not the legal name.
            if (q := social_query(d.candidate.aliases[0] if d.candidate.aliases else _common_name(d.candidate.displayName, st.name), any(s.knowledgeGraph or s.sourceType == "wiki" for s in d.candidate.sources)))
            and f"web:{q.query}" not in st.done
        ]
        if not plan:
            return
        job.touch("searching")
        outs = await asyncio.gather(*(self.search.search(q.query, "web", job.toolCalls, st.locale) for _, q in plan))
        claimed = {p.url: d for d in drafts for p in d.candidate.socialProfiles}
        for (d, q), (hits, _) in zip(plan, outs):
            st.done.add(f"web:{q.query}")
            st.queries.append(QueryLog(query=q.query, kind="web", round="enrichment", purpose=q.purpose))
            c = d.candidate
            have = {p.platform for p in c.socialProfiles}
            id_words = identity_words(c.occupation, c.knownFor, c.headline)
            # With a namesake in the results the name alone cannot tell whose account it is.
            namesake = any(o is not d and name_match(o.candidate.displayName, c.displayName) >= 0.99 for o in drafts)
            for h in sorted(hits, key=lambda h: h.position):
                url = normalize_url(h.url)
                platform = social_platform(url)
                root = profile_root(url) if platform else None
                if not root or platform in have or looks_like_fan_account(root, h.title):
                    continue
                if claimed.get(root, d) is not d:
                    continue
                if not handle_matches_name(root, c.displayName, st.name, *c.aliases):
                    continue
                text = f"{h.title} {h.snippet}"
                # A namesake shares the name, so a partial name match must also describe the person.
                exact = name_match(c.displayName, text) >= 0.99 or name_match(st.name, h.title) >= 0.99
                described = name_match(c.displayName, text) >= 0.6 and mentions_identity(text, id_words)
                by_alias = any(name_match(a, text) >= 0.99 for a in c.aliases)
                if not (described or by_alias) and (namesake or not exact):
                    continue
                known = next((s for s in st.sources if s.url == url), None)
                if known is None:
                    matched = next((a for a in c.aliases if name_match(a, text) >= 0.99), c.displayName)
                    new = build_sources([h], matched, start_index=len(st.sources) + 1)
                    if not new:
                        continue
                    known = new[0]
                    st.sources.append(known)
                if known not in c.sources:
                    c.sources.append(known)
                c.socialProfiles.append(
                    SocialProfile(platform=platform, url=root, sourceIds=[known.id], confidence=round(min(0.9, c.confidenceScore), 3))
                )
                have.add(platform)
                claimed[root] = d

    async def _ask_model(self, st: _State) -> KnowledgePlan | None:
        """What the model knows about the name and which searches it still needs."""
        assert self.llm is not None
        st.llm_calls += 1
        try:
            return await self.llm.plan(st.name, st.filters, self.s.max_queries)
        except Exception as e:
            log.warning("knowledge plan failed, falling back to search-first: %s", e)
            return None

    def _from_knowledge(self, st: _State, recalled: list[RecalledCandidate], cap: float, note: str) -> list[CandidateDraft]:
        drafts = []
        for i, rc in enumerate(recalled[: self.s.max_candidates], 1):
            rc = _verified_links(rc, st.hit_keys)
            draft, sources = candidate_from_recall(i, st.name, rc, len(st.sources) + 1, cap=cap, note=note)
            st.sources.extend(sources)
            drafts.append(draft)
        return drafts

    async def _recall(self, st: _State) -> list[CandidateDraft]:
        """Search had nothing usable (spelling variant, sparse index): fall back to
        the model's knowledge of public people. No extra search call."""
        assert self.llm is not None
        st.llm_calls += 1
        try:
            out = await self.llm.recall(st.name, st.filters)
        except Exception as e:
            log.warning("recall failed: %s", e)
            return []
        drafts = []
        for i, rc in enumerate(out.candidates[: self.s.max_candidates], 1):
            rc = _verified_links(rc, st.hit_keys)
            draft, sources = candidate_from_recall(i, st.name, rc, start_index=len(st.sources) + 1)
            st.sources.extend(sources)
            drafts.append(draft)
        if drafts:
            st.warnings.append("No matching sources found; candidates come from model knowledge and are unverified.")
        drafts.sort(key=lambda d: d.candidate.confidenceScore, reverse=True)
        return drafts

    async def _wiki_profiles(self, candidates: list[FeCandidate]) -> None:
        """The profile is the lead section of the person's English Wikipedia
        article; without one, the model's profile stays."""

        async def one(c: FeCandidate) -> None:
            wiki = person_wiki(c)
            en = await self.fetcher.english_wikipedia(wiki) if wiki else None
            if en and en != wiki:
                c.officialLinks = [en if u == wiki else u for u in c.officialLinks]
            _, extract = await self.fetcher.wikipedia_summary(en) if en else (None, "")
            if paragraphs := [p for p in (" ".join(x.split()) for x in extract.split("\n")) if p]:
                c.narrative, c.narrativeSource = paragraphs, "wikipedia"

        await asyncio.gather(*(one(c) for c in candidates))

    async def _free_avatars(self, candidates: list[FeCandidate]) -> None:
        """Avatars that cost no search call: the Wikipedia lead image, else the
        GitHub profile picture. A knowledge-panel image is kept when present."""

        async def one(c: FeCandidate) -> None:
            if any(s.knowledgeGraph and s.image for s in c.sources):
                return
            # The main article has the shortest path ("Cristiano_Ronaldo" vs "Career_of_Cristiano_Ronaldo").
            wikis = sorted((s for s in c.sources if s.sourceType == "wiki"), key=lambda s: len(s.url))
            for s in wikis[:1]:
                if thumb := await self.fetcher.wikipedia_thumbnail(s.url):
                    c.avatarHint = thumb
                    return
            if not c.avatarHint:
                github = next((p.url for p in c.socialProfiles if p.platform == "github"), None)
                if github:
                    c.avatarHint = f"{github}.png"

        await asyncio.gather(*(one(c) for c in candidates))

    async def _order_photos(self, c: FeCandidate) -> None:
        """Photos from the person's own pages first (the knowledge-panel or Wikipedia avatar
        leading; the first one is the profile photo), the same picture only once."""
        photos = sorted(c.photos, key=lambda p: (not p.sourceIds, p.url != c.avatarHint, -(p.faceMatch or 0)))
        photos = [photos[i] for i in await distinct_urls(self.faces, [p.url for p in photos])]
        c.photos = photos[: max(1, self.s.initial_photos)]
        if c.photos:
            c.avatarHint = c.photos[0].url

    async def _source_photos(self, candidates: list[FeCandidate]) -> None:
        """Photos from each candidate's own pages (Wikipedia, GitHub, named portraits,
        og:image): no search, no LLM. Pages read during the search come from the cache."""
        limit = max(1, self.s.initial_photos)

        async def one(c: FeCandidate) -> None:
            if sum(1 for p in c.photos if p.sourceIds) >= limit or not (refs := client_sources(c)):
                return
            try:
                found, _ = await self.source_photos.find(c.displayName, refs, limit=limit)
            except Exception as e:
                log.warning("source photos for %s failed: %s", c.displayName, e)
                return
            by_page = {link_key(s.url): s.id for s in c.sources}
            have = {image_key(p.url) for p in c.photos}
            for f in found:
                if image_key(f.url) not in have and (sid := by_page.get(link_key(f.pageUrl or ""))):
                    c.photos.append(Photo(url=f.url, pageUrl=f.pageUrl, title=f.title, sourceIds=[sid], faceMatch=f.faceMatch))

        await asyncio.gather(*(one(c) for c in candidates))

    async def _photo_search(self, job: Job, st: _State, candidates: list[FeCandidate]) -> None:
        """Fill the photo gallery: the avatar first, then one image search on
        `imageSearchQuery` per strong candidate. Results whose title does not
        carry the person's name are dropped; images hosted on one of the
        candidate's own sources are marked as sourced."""
        for c in candidates:
            if c.avatarHint and c.avatarHint.startswith(("http://", "https://")) and all(p.url != c.avatarHint for p in c.photos):
                src = next((s for s in c.sources if s.sourceType == "wiki" or (s.knowledgeGraph and s.image == c.avatarHint)), None)
                c.photos.insert(0, Photo(url=c.avatarHint, pageUrl=src.url if src else None, title=c.displayName, sourceIds=[src.id] if src else []))
        # Candidates with enough photos from their own pages need no image search.
        need = max(1, self.s.initial_photos)
        targets = [
            c for c in candidates
            if c.confidenceScore >= 0.4 and len(c.photos) < self.s.max_photos and sum(1 for p in c.photos if p.sourceIds) < need
        ][: self.s.photo_search_candidates]
        if not targets:
            return
        queries = [c.imageSearchQuery or c.displayName for c in targets]
        try:
            outs = await asyncio.gather(*(self.search.search(q, "image", job.toolCalls, st.locale) for q in queries))
        except RuntimeError as e:
            log.warning("photo search skipped: %s", e)
            return
        for c, q, (hits, _) in zip(targets, queries, outs):
            st.queries.append(QueryLog(query=q, kind="image", round="enrichment", purpose="photos"))
            by_key = {link_key(s.url): s for s in c.sources}
            by_id = {s.id: s for s in c.sources}
            for sp in c.socialProfiles:
                if (src := next((by_id[i] for i in sp.sourceIds if i in by_id), None)) is not None:
                    by_key.setdefault(link_key(sp.url), src)
            names = [c.displayName, st.name, *c.aliases]
            for h in sorted(hits, key=lambda h: h.position):
                if not h.image_url or not h.image_url.startswith(("http://", "https://")):
                    continue
                # Images on the person's own pages count even when their caption lacks the name.
                src = by_key.get(link_key(h.url))
                if not src and max(name_match(n, f"{h.title} {h.url}") for n in names) < 0.6:
                    continue
                photo = Photo(url=h.image_url, pageUrl=h.url, title=h.title, fallbackUrl=h.image_fallback)
                if src:
                    photo.sourceIds = [src.id]
                else:
                    photo.note = "Suggested image from image search by name; identity not confirmed."
                c.photos.append(photo)
            unique: dict[str, Photo] = {}
            for p in c.photos:
                unique.setdefault(image_key(p.url), p)
            c.photos = list(unique.values())[: self.s.max_photos]
            if not c.avatarHint and c.photos:
                c.avatarHint = c.photos[0].url

    async def _verify_faces(self, candidates: list[FeCandidate]) -> list[Face | None]:
        """Drop photos without a face or whose face is someone else's. Photos
        from the person's own sources (the avatar first) give the reference face.
        Returns each candidate's reference face (None when there is none)."""
        refs: list[Face | None] = [None] * len(candidates)
        if not self.faces or not await self.faces.ready():
            return refs

        async def one(k: int, c: FeCandidate) -> None:
            if not c.photos:
                return
            found, used = await self.faces.faces_or_fallback([p.url for p in c.photos], [p.fallbackUrl for p in c.photos])
            for p, u in zip(c.photos, used):
                if u != p.url:
                    p.url, p.fallbackUrl = u, None
            kept = select_photos(found, [bool(p.sourceIds) for p in c.photos], self.s.face_match_threshold)
            photos = []
            for i, score in kept:
                p = c.photos[i]
                p.faceMatch = score
                if score == 1.0 and found[i] and len(found[i]) == 1:
                    refs[k] = found[i][0]
                p.note = (
                    "Reference face for this person."
                    if score == 1.0
                    else f"Face matches the person's reference face (similarity {score:.2f})."
                )
                photos.append(p)
            c.photos = photos
            if not c.avatarHint or c.avatarHint not in {p.url for p in photos}:
                c.avatarHint = photos[0].url if photos else None

        with tracing.span("verify-faces", input={c.displayName: len(c.photos) for c in candidates}) as obs:
            await asyncio.gather(*(one(k, c) for k, c in enumerate(candidates)))
            obs.update(output={c.displayName: len(c.photos) for c in candidates})
        return refs

    async def _resolve_shared_faces(self, st: _State, candidates: list[FeCandidate], refs: list[Face | None]) -> None:
        """Namesakes often get each other's photos (an image search for one returns
        the other). A face belongs to one person only: when several candidates have
        the same reference face, the one with the most photos of it keeps it; the
        others lose those photos and pick their reference again from the rest."""
        if not self.faces or self.s.face_shared_threshold <= 0:
            return
        thr = self.s.face_match_threshold
        for group in same_face_groups(refs, self.s.face_shared_threshold):
            owner = max(group, key=lambda i: (len(candidates[i].photos), candidates[i].confidenceScore))
            face = refs[owner]
            for i in group:
                if i == owner:
                    continue
                c = candidates[i]
                found = await self.faces.faces_in([p.url for p in c.photos])
                rest = [
                    (p, fs) for p, fs in zip(c.photos, found)
                    if fs and max(cosine(f.embedding, face.embedding) for f in fs) < thr
                ]
                kept = select_photos([fs for _, fs in rest], [bool(p.sourceIds) for p, _ in rest], thr)
                photos = []
                for j, score in kept:
                    p = rest[j][0]
                    p.faceMatch = score
                    p.note = "Reference face for this person." if score == 1.0 else f"Face matches the person's reference face (similarity {score:.2f})."
                    photos.append(p)
                st.warnings.append(
                    f"{len(c.photos) - len(photos)} photo(s) removed from {c.displayName!r} ({c.headline or ''}): "
                    f"they show {candidates[owner].headline or candidates[owner].displayName}."
                )
                c.photos = photos
                if c.avatarHint not in {p.url for p in photos}:
                    c.avatarHint = photos[0].url if photos else None

    async def _read(self, st: _State, sources: list[Source], budget: int) -> None:
        to_fetch = [s for s in sources if can_fetch(s.url) and s.url not in st.pages][:budget]
        pages = {}
        if to_fetch:
            with tracing.span("fetch-pages", input=[s.url for s in to_fetch]) as obs:
                pages = await self.fetcher.fetch_many([s.url for s in to_fetch])
                obs.update(output={u: p.status for u, p in pages.items()})
        st.pages.update(pages)
        for s in sources:
            page = st.pages.get(s.url)
            if page:
                s.contentStatus = page.status if page.status in ("fetched", "blocked") else "failed"  # type: ignore[assignment]
                if page.status == "fetched" and not s.title:
                    s.title = page.title
            elif not can_fetch(s.url):
                s.contentStatus = "blocked"

    @staticmethod
    def _kg_hit(st: _State, kg: dict[str, Any], query: str) -> SearchHit | None:
        """Search-engine knowledge panel -> a strong source for public figures."""
        title = str(kg.get("title") or "")
        url = kg.get("descriptionLink") or kg.get("website")
        if not title or not isinstance(url, str) or not url.startswith("http") or name_match(st.name, title) < 0.99:
            return None
        image = kg.get("imageUrl")
        st.kg_images[normalize_url(url)] = image if isinstance(image, str) and image.startswith(("http://", "https://")) else None
        attrs = kg.get("attributes") if isinstance(kg.get("attributes"), dict) else {}
        # Attributes go before the description, which is long and gets truncated.
        parts = [title, kg.get("type"), *(f"{k}: {v}" for k, v in attrs.items() if isinstance(v, str) and not _PRIVATE_KG_ATTR.search(str(k)))]
        parts.append(kg.get("description"))
        return SearchHit(provider="knowledge_graph", query=query, title=title, url=url, snippet=" — ".join(str(x) for x in parts if x), position=0)

    def _llm_payload(self, st: _State) -> list[dict[str, Any]]:
        ranked = sorted(st.sources, key=lambda s: (not s.knowledgeGraph, s.contentStatus != "fetched", -s.rank))[: self.s.max_llm_sources]
        items = []
        for s in sorted(ranked, key=lambda s: int(s.id.split("_")[1])):
            item: dict[str, Any] = {"sourceId": s.id, "url": s.url, "title": s.title, "snippet": s.snippet[:SNIPPET_CHARS]}
            page = st.pages.get(s.url)
            if page and page.status == "fetched":
                item["excerpt"] = excerpt(page.text, st.name, self.s.excerpt_chars)
            items.append(item)
        return items

    async def _analyze(self, st: _State) -> list[CandidateDraft]:
        assert self.llm is not None
        if not st.sources:
            return []
        st.llm_calls += 1
        try:
            analysis = await self.llm.analyze(st.name, st.filters, self._llm_payload(st))
        except Exception as e:
            log.warning("analyze failed: %s", e)
            st.warnings.append("LLM source analysis failed.")
            raise
        groups = groups_from_analysis(analysis, {s.id: s for s in st.sources}, st.pages, st.name)
        drafts = []
        for i, g in enumerate(groups, 1):
            d = build_candidate(i, st.name, g.mentions, st.pages, st.image_hits, st.filters)
            apply_profile(d, g.profile)
            _drop_unrelated_social(d, st.name)
            drafts.append(d)
        drafts.sort(key=lambda d: d.candidate.confidenceScore, reverse=True)
        return drafts


def _model_queries(raw: list[str], names: list[str], budget: int, done: set[str]) -> list[PlannedQuery]:
    """Searches requested by the model, kept on topic: each must contain the
    query name or a name/alias of a person the model returned."""
    out: list[PlannedQuery] = []
    for q in raw:
        q = " ".join(str(q).split())[:200]
        if not q:
            continue
        if not any(n and (fold(n) in fold(q) or spacing_variant(n, q)) for n in names):
            q = f"{names[0]} {q}"
        key = f"web:{q}"
        if key in done:
            continue
        done.add(key)
        out.append(PlannedQuery(q, "web", "model"))
        if len(out) >= budget:
            break
    return out


def _assign_shared_sources(drafts: list[CandidateDraft]) -> None:
    """A page belongs to one person: when namesakes share a source, only the
    candidate whose facts cite it most keeps it (ties go to the higher score)."""
    owners: dict[str, list[CandidateDraft]] = {}
    for d in drafts:
        for s in d.candidate.sources:
            owners.setdefault(s.url, []).append(d)

    def cites(c: FeCandidate, url: str) -> int:
        ids = {s.id for s in c.sources if s.url == url}
        n = sum(len(ids & set(f.sourceIds)) for f in c.fields.values())
        return n + sum(len(ids & set(ci.sourceIds)) for ci in c.careerArc)

    for url, ds in owners.items():
        if len(ds) < 2:
            continue
        keep = max(ds, key=lambda d: (cites(d.candidate, url), d.candidate.confidenceScore))
        for d in ds:
            if d is keep:
                continue
            c = d.candidate
            gone = {s.id for s in c.sources if s.url == url}
            c.sources = [s for s in c.sources if s.url != url]
            c.socialProfiles = [p for p in c.socialProfiles if p.url != url and not (p.sourceIds and set(p.sourceIds) <= gone)]


def _drop_unrelated_social(d: CandidateDraft, query_name: str) -> None:
    """Social pages found by search are only kept when they plausibly are the
    person's own: an account whose handle carries their name or an alias
    (fan accounts like instagram.com/joonie94_bts do not), and no posts or
    reels by other users unless a verified fact cites them. Profile links read
    from the person's own website are kept as they are."""
    c = d.candidate
    names = [c.displayName, query_name, *c.aliases]
    cited = {sid for f in c.fields.values() for sid in f.sourceIds} | {sid for x in d.careers for sid in x.sourceIds}
    unrelated: set[str] = set()
    for s in c.sources:
        if social_platform(s.url) and ((root := profile_root(s.url)) is None or not handle_matches_name(root, *names)):
            unrelated.add(s.id)
    if not unrelated:
        return
    c.socialProfiles = [p for p in c.socialProfiles if not (p.sourceIds and set(p.sourceIds) <= unrelated)]
    kept = [p for p in c.photos if not (p.sourceIds and set(p.sourceIds) <= unrelated)]
    if c.avatarHint in {p.url for p in c.photos} - {p.url for p in kept}:
        c.avatarHint = next((p.url for p in kept if p.sourceIds), None)
    c.photos = kept
    c.sources = [s for s in c.sources if s.id not in unrelated or s.id in cited]


def _verified_links(rc: RecalledCandidate, hit_keys: set[str]) -> RecalledCandidate:
    """Keeps only the model's remembered links that some search result points to
    (same page, or a tab of the same profile); the rest may be made up."""
    return rc.model_copy(update={"sourceUrls": [u for u in rc.sourceUrls if isinstance(u, str) and link_key(u) in hit_keys]})


def _merge_known(drafts: list[CandidateDraft], known: list[CandidateDraft]) -> list[CandidateDraft]:
    """Search-verified candidates win; people the model knows but the search did
    not confirm are kept with their capped, unverified confidence. When the
    search confirmed a known person, the model's profile links move over to
    that candidate instead of being lost."""
    out = list(drafts)

    def same_name(a: FeCandidate, b: FeCandidate) -> bool:
        # "Lalisa Manobal" (model) is the search candidate "Lisa" with alias "Lalisa Manobal".
        return any(name_match(x, y) >= 0.99 for x in (a.displayName, *a.aliases) for y in (b.displayName, *b.aliases))

    def shared_link(a: FeCandidate, b: FeCandidate) -> bool:
        keys = {link_key(s.url) for s in a.sources} | {link_key(p.url) for p in a.socialProfiles}
        return bool(keys & ({link_key(s.url) for s in b.sources} | {link_key(p.url) for p in b.socialProfiles}))

    for k in known:
        kc = k.candidate
        linked = [d for d in drafts if shared_link(kc, d.candidate)]
        matches = linked if len(linked) == 1 else [d for d in drafts if same_name(kc, d.candidate)]
        if not matches:
            out.append(k)
        elif target := _same_person(kc, matches):
            _adopt_links(target, kc, drafts)
    return out


def _same_person(known: FeCandidate, matches: list[CandidateDraft]) -> CandidateDraft | None:
    """The verified namesake that is the known person: a shared link first, then
    the most shared role/field words, else the only namesake."""
    urls = {s.url for s in known.sources}
    for d in matches:
        if urls & {s.url for s in d.candidate.sources} or urls & {p.url for p in d.candidate.socialProfiles}:
            return d
    words = identity_words(known.occupation, known.knownFor, known.headline)
    if words:
        def overlap(d: CandidateDraft) -> int:
            c = d.candidate
            text = " ".join(x for x in (c.headline, c.occupation, c.knownFor, c.disambiguationNote) if x)
            return sum(1 for w in words if mentions_identity(text, [w]))

        best = max(matches, key=overlap)
        if overlap(best) > 0 and sum(1 for d in matches if overlap(d) == overlap(best)) == 1:
            return best
    return matches[0] if len(matches) == 1 else None


def _adopt_links(target: CandidateDraft, known: FeCandidate, drafts: list[CandidateDraft]) -> None:
    c = target.candidate
    c.aliases = split_aliases(c.displayName, c.aliases + known.aliases)[1]
    taken = {p.url for d in drafts if d is not target for p in d.candidate.socialProfiles}
    have_urls = {s.url for s in c.sources}
    platforms = {p.platform for p in c.socialProfiles}
    by_id = {s.id: s for s in known.sources}
    for p in known.socialProfiles:
        if p.platform in platforms or p.url in taken:
            continue
        c.socialProfiles.append(p)
        platforms.add(p.platform)
        for sid in p.sourceIds:
            if (s := by_id.get(sid)) and s.url not in have_urls:
                c.sources.append(s)
                have_urls.add(s.url)
    for s in known.sources:
        if s.sourceType in ("wiki", "official_website") and s.url not in have_urls:
            c.sources.append(s)
            have_urls.add(s.url)


def _common_name(display: str, query: str) -> str:
    """Short public form of a long legal name, cut after the last query word:
    ("Cristiano Ronaldo dos Santos Aveiro", "Ronaldo") -> "Cristiano Ronaldo"."""
    words = display.split()
    folded = [fold(w) for w in words]
    hits = [i for i, w in enumerate(folded) if w in set(fold(query).split())]
    return " ".join(words[: hits[-1] + 1]) if hits and hits[-1] >= 1 and len(words) > 3 else display


def _first_query_sufficient(sources: list[Source]) -> bool:
    """A knowledge panel, or a wiki page plus profiles on two platforms, already
    covers what the extra profile query would add."""
    if any(s.knowledgeGraph for s in sources):
        return True
    platforms = {social_platform(s.url) for s in sources if profile_root(s.url)}
    return any(s.sourceType == "wiki" for s in sources) and len(platforms) >= 2


def _trace_output(job: Job) -> dict[str, Any]:
    r = job.result
    if r is None:
        return {"status": job.status.value, "error": job.error}
    return {
        "status": r.status,
        "clarificationQuestion": r.clarificationQuestion,
        "warnings": r.warnings,
        "stats": r.stats,
        "queries": [f"{q.kind}/{q.purpose}: {q.query}" for q in r.queries],
        "candidates": [
            {
                "displayName": c.displayName,
                "confidence": c.confidenceScore,
                "matchNote": c.matchNote,
                "links": c.officialLinks + [p.url for p in c.socialProfiles],
                "photos": len(c.photos),
            }
            for c in r.candidates
        ],
    }


def _make_image_queries_distinct(candidates: list[FeCandidate]) -> None:
    seen: set[str] = set()
    for c in candidates:
        q = c.imageSearchQuery
        if fold(q) in seen:
            for extra in (c.knownFor, c.occupation, c.nationality, c.fields["company"].value, c.fields["location"].value):
                if extra and fold(extra) not in fold(q):
                    candidate_q = f"{q} {extra}"
                    if len(candidate_q) <= 80 and fold(candidate_q) not in seen:
                        q = candidate_q
                        break
            c.imageSearchQuery = q
        seen.add(fold(q))


def _decide_status(name: str, candidates: list[FeCandidate]) -> tuple[str, str | None]:
    if not candidates:
        return "no_match", None
    top = candidates[0]
    plausible = [c for c in candidates if c.confidenceScore >= 0.4]
    if top.confidenceScore >= 0.6 and (len(plausible) < 2 or plausible[1].confidenceScore < top.confidenceScore - 0.2):
        return "resolved", None
    if len(plausible) >= 2:
        options = [c.headline or c.disambiguationNote for c in plausible[:3]]
        return "ambiguous", f"Found {len(plausible)} people named {name} in public results. Which one do you mean: " + " or ".join(f"“{o}”" for o in options) + "?"
    return "ambiguous", f"Not enough public information to confirm the identity of {name}; the results below need further checking."
