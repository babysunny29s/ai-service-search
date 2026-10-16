"""Profile Search HTTP API (plain REST/JSON, not MCP)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.core.client_format import to_client_response
from app.core.llm import OUTPUT_SCHEMAS, _parse_json
from app.core.models import (
    ChatRequest,
    ChatResponse,
    ClientCandidate,
    EnrichRequest,
    EnrichResponse,
    IdentifyRequest,
    IdentifyResponse,
    ImageSearchRequest,
    ImageSearchResponse,
    Job,
    JobStatus,
    MorePhotosRequest,
    MorePhotosResponse,
    PeopleSearchRequest,
    PeopleSearchResponse,
    PhotoWarning,
    DeepSearchRequest,
    DeepSearchResponse,
    SearchInput,
    SearchProfileRequest,
    ClientSource,
    SocialSourcesRequest,
    SourcePhotosRequest,
    SourcePhotosResponse,
    SourcesResponse,
    StepMeta,
    WebSourcesRequest,
    _clean_filters,
)
from app.infra.search_store import ChatSessionConflict, deep_metadata, persist_people, reuse_deep, reuse_search
from app.media.photos import PhotoError
from app.search.deep_search import DeepSearchError
from app.search.enrich import EnrichError
from app.search.identify import MAX_UPLOAD_BYTES, IdentifyError, detect_image_mime, parse_photo_warnings
from app.search.preview import SearchUnavailable
from app.search.search_profile import SearchProfileError
from app.search.sources import SourceError
from .runtime import start_runtime

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("profile-search")


@asynccontextmanager
async def lifespan(app: FastAPI):
    rt = await start_runtime()
    rt.bind(app.state)
    try:
        yield
    finally:
        await rt.close()


app = FastAPI(title="Profile Search API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _start(inp: SearchInput) -> tuple[Job, asyncio.Task]:
    job = app.state.jobs.create(inp)
    task = asyncio.create_task(app.state.orch.run(job))
    app.state.jobs.attach(job, task)
    return job, task


def _http_wait_s() -> float:
    return max(0.0, float(getattr(app.state.settings, "http_wait_s", 0) or 0))


def _json(payload: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status, headers={"Cache-Control": "no-store"})


def _client_payload(job: Job) -> dict | None:
    return job.extra.get("client") if isinstance(job.extra.get("client"), dict) else None


async def _run_client(job: Job, coro, stage: str) -> None:
    job.touch(stage, JobStatus.running)
    try:
        out = await coro
        if "searchId" in getattr(out, "model_fields", {}) and not getattr(out, "searchId", None):
            out = out.model_copy(update={"searchId": job.searchId})
        job.extra["client"] = json.loads(out.model_dump_json())
        job.touch("complete", JobStatus.complete)
    except (SearchProfileError, DeepSearchError) as e:
        job.error = str(e)
        job.extra["httpStatus"] = e.status
        job.touch("failed", JobStatus.failed)
    except Exception as e:
        job.error = f"{type(e).__name__}: {str(e)[:300]}"
        job.touch("failed", JobStatus.failed)


async def _await_client(job: Job, task: asyncio.Task):
    wait = _http_wait_s()
    if wait <= 0:
        return _json(_pending(job), 202)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=wait)
    except asyncio.TimeoutError:
        return _json(_pending(job), 202)
    if job.status == JobStatus.failed:
        raise HTTPException(int(job.extra.get("httpStatus") or 502), job.error or "search failed")
    payload = _client_payload(job)
    if not payload:
        raise HTTPException(502, {"searchId": job.searchId, "error": job.error or "empty result"})
    return _json(payload)


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _html(name: str) -> FileResponse:
    """Dev UI HTML — never cache so button/layout edits show up after reload."""
    return FileResponse(
        STATIC_DIR / name,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/", include_in_schema=False)
async def test_ui():
    return _html("index.html")


@app.get("/prompt-lab", include_in_schema=False)
async def prompt_lab_ui():
    return _html("prompt-lab.html")


class PromptRunRequest(BaseModel):
    system: str
    user: str
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    maxTokens: int = Field(default=4096, ge=1, le=32000)
    jsonMode: bool = True
    schemaName: str | None = None


def _lab_llm():
    llm = app.state.llm
    if llm is None or not hasattr(llm, "chat_raw"):
        raise HTTPException(503, "LLM provider is not configured")
    return llm


@app.get("/prompt-lab/templates")
async def prompt_lab_templates():
    llm = _lab_llm()
    s = app.state.settings
    return {
        "model": llm.model,
        "temperature": llm.temperature,
        "schemas": list(OUTPUT_SCHEMAS),
        "templates": llm.templates(s.max_queries),
    }


@app.get("/prompt-lab/history")
async def prompt_lab_history():
    return {"items": list(_lab_llm().history)}


@app.post("/prompt-lab/run")
async def prompt_lab_run(req: PromptRunRequest):
    llm = _lab_llm()
    try:
        out = await llm.chat_raw(
            req.system, req.user, model=req.model or None, temperature=req.temperature, max_tokens=req.maxTokens, json_mode=req.jsonMode
        )
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {str(e)[:500]}")
    parsed, parse_error, validation_error = None, None, None
    try:
        parsed = _parse_json(out["content"])
    except (json.JSONDecodeError, ValueError) as e:
        parse_error = str(e)
    if parsed is not None and req.schemaName in OUTPUT_SCHEMAS:
        try:
            OUTPUT_SCHEMAS[req.schemaName].model_validate(parsed)
        except ValidationError as e:
            validation_error = str(e)
    return {**out, "parsed": parsed, "parseError": parse_error, "validationError": validation_error}


@app.get("/health")
async def health():
    return {"status": "ok", "mcp": app.state.mcp.status(), "llm": getattr(app.state.llm, "name", None)}


async def _load_store_candidate(candidate_id: str | None, user_id: str | None) -> dict:
    if not (candidate_id or "").strip():
        raise HTTPException(422, "candidateId required")
    try:
        return await app.state.search_store.session(candidate_id.strip(), user_id)
    except KeyError:
        raise HTTPException(404, "candidate not found") from None
    except PermissionError:
        raise HTTPException(403, "candidate belongs to another user") from None


async def _persist_profile(job: Job, req: SearchProfileRequest):
    out = None
    if req.page == 1 and not (req.excludeUrls or req.excludeNames or req.refresh):
        out = await reuse_search(app.state.search_store, req.query, req.filters, "searchprofile", app.state.settings.reuse_search_max_age_h)
    if out is None:
        out = await app.state.profile_search.run(req.query, req.filters, req.excludeUrls, req.excludeNames, req.page)
    sid = (req.searchId or job.searchId).strip()
    saved = await persist_people(
        app.state.search_store,
        search_id=sid,
        query=req.query,
        filters=req.filters,
        mode="searchprofile",
        candidates=out.candidates,
        user_id=req.userId,
    )
    return out.model_copy(update={"searchId": sid, "candidates": saved})


async def _persist_deep(candidate: ClientCandidate, filters: dict[str, str], rec: dict, refresh: bool = False):
    out = None if refresh else await reuse_deep(app.state.search_store, candidate, app.state.settings.reuse_deep_max_age_h)
    if out is None:
        out = await app.state.deep_search.run(candidate, filters)
    try:
        await app.state.search_store.update_candidate(candidate.id, deep_metadata(out))
    except KeyError:
        log.warning("deep-search candidate %s not in store", candidate.id)
    sid = rec.get("searchId")
    if sid:
        out = out.model_copy(update={"searchId": sid})
    return out


@app.post("/v1/search-profile", response_model=PeopleSearchResponse)
async def search_profile(req: SearchProfileRequest):
    """Find people by name. Up to 2 Serper queries (name + filters + social/profile
    sites; English Wikipedia is preferred), then one LLM call that merges the same
    person and drops namesakes. Send excludeUrls / excludeNames and page=2, 3, …
    to search the next Serper pages of the same two queries. Every candidate has
    at least one profile source. Slow runs return 202 with searchId; poll
    GET /v1/people-search/{searchId}."""
    if not app.state.mcp.servers_for("web_search"):
        raise HTTPException(503, "no search MCP server connected")
    job = app.state.jobs.create(SearchInput(name=req.query, filters=req.filters))
    task = asyncio.create_task(_run_client(job, _persist_profile(job, req), "searching"))
    app.state.jobs.attach(job, task)
    return await _await_client(job, task)


@app.post("/v1/deep-search", response_model=DeepSearchResponse)
async def deep_search(req: DeepSearchRequest):
    """Dossier for one person chosen from /v1/search-profile. Their pages are read
    first, then articles are searched, then one LLM call writes biography and
    key facts. Social links are kept only when a read page states them.
    Slow runs return 202 with searchId; poll GET /v1/people-search/{searchId}."""
    if not app.state.mcp.servers_for("web_search"):
        raise HTTPException(503, "no search MCP server connected")
    rec: dict = {}
    candidate = req.candidate
    filters = req.filters
    if candidate is None:
        rec = await _load_store_candidate(req.candidateId, req.userId)
        candidate = ClientCandidate.model_validate(rec["candidate"])
        filters = filters or rec.get("filters") or {}
    job = app.state.jobs.create(SearchInput(name=candidate.displayName, filters=filters))
    task = asyncio.create_task(_run_client(job, _persist_deep(candidate, filters, rec, req.refresh), "enriching"))
    app.state.jobs.attach(job, task)
    return await _await_client(job, task)


@app.post("/v1/people-search", response_model=PeopleSearchResponse)
async def people_search(req: PeopleSearchRequest):
    """preview: search results only, no LLM, answered inline.
    deep: full search; when it outlasts the sync timeout the reply is 202 with a
    searchId to poll at GET /v1/people-search/{searchId}."""
    if not app.state.mcp.servers_for("web_search"):
        raise HTTPException(503, "no search MCP server connected")
    if req.mode == "preview":
        try:
            return await app.state.preview.run(req.query, req.filters, req.maxSearchQueries)
        except SearchUnavailable as e:
            raise HTTPException(503, {"error": "search unavailable", "detail": str(e)[:300]})
    if app.state.llm is None:
        raise HTTPException(503, "LLM provider is not configured")
    job, task = _start(SearchInput(name=req.query, filters=req.filters))
    return await _await_job(job, task)


async def _await_job(job: Job, task: asyncio.Task):
    wait = _http_wait_s()
    if wait <= 0:
        return _json(_pending(job), 202)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=wait)
    except asyncio.TimeoutError:
        return _json(_pending(job), 202)
    if job.status != JobStatus.complete or not job.result:
        raise HTTPException(502, {"searchId": job.searchId, "error": job.error})
    if payload := _client_payload(job):
        return _json(payload)
    return _response(job)


def _response(job: Job) -> PeopleSearchResponse:
    out = to_client_response(job.result)
    if job.result.mode == "image":
        out.identified = job.extra.get("identified", [])
        out.photoWarnings = job.extra.get("photoWarnings", [])
        out.lensInput = job.extra.get("lensInput")
        out.lensResults = job.extra.get("lensResults")
        out.lensImageUrl = job.extra.get("lensImageUrl")
        out.warnings = job.result.warnings or None
    return out


def _pending(job: Job) -> dict:
    out = {"searchId": job.searchId, "status": job.status.value, "stage": job.stage, "resultUrl": f"/v1/people-search/{job.searchId}"}
    progress = job.extra.get("progress")
    if isinstance(progress, dict):
        if progress.get("stage"):
            out["stage"] = progress["stage"]
        if progress.get("text"):
            out["text"] = progress["text"]
        if progress.get("queries"):
            out["queries"] = progress["queries"]
    return out


async def _run_chat_candidate(job: Job, rec: dict, question: str) -> None:
    await _consume_chat(job, app.state.chat.stream_with(rec, rec["sessionId"], question, app.state.search_store))


async def _consume_chat(job: Job, events) -> None:
    job.touch("thinking", JobStatus.running)
    job.extra["progress"] = {"stage": "thinking", "text": ""}
    try:
        async for ev in events:
            event = ev.get("event")
            if event == "status":
                job.stage = ev.get("stage") or job.stage
                job.extra["progress"] = {**(job.extra.get("progress") or {}), "stage": job.stage, "queries": ev.get("queries")}
            elif event == "reset":
                job.extra["progress"] = {**(job.extra.get("progress") or {}), "text": ""}
            elif event == "delta":
                progress = job.extra.get("progress") or {}
                progress["text"] = (progress.get("text") or "") + (ev.get("text") or "")
                progress["stage"] = progress.get("stage") or "answering"
                job.extra["progress"] = progress
                job.stage = progress["stage"]
            elif event == "done":
                job.extra["client"] = {k: v for k, v in ev.items() if k != "event"}
                job.touch("complete", JobStatus.complete)
                return
            elif event == "error":
                job.error = ev.get("message") or "chat failed"
                job.extra["httpStatus"] = ev.get("status") or 502
                job.touch("failed", JobStatus.failed)
                return
        if not _client_payload(job):
            job.error = "no answer"
            job.touch("failed", JobStatus.failed)
    except Exception as e:
        job.error = f"{type(e).__name__}: {str(e)[:300]}"
        job.touch("failed", JobStatus.failed)


async def _read_upload(image: UploadFile) -> tuple[bytes, str]:
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(422, "image required")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image must be 5 MB or smaller")
    mime = detect_image_mime(data)
    if not mime:
        raise HTTPException(415, "image must be jpeg, png or webp")
    return data, mime


def _form_filters(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        raise HTTPException(422, 'filters must be a JSON object, e.g. {"location": "Hanoi"}')
    return _clean_filters(value)


def _form_warnings(raw: str | None) -> list[PhotoWarning]:
    warnings = parse_photo_warnings(raw)
    if warnings is None:
        raise HTTPException(422, "photoWarnings: comma-separated no_face, multiple_faces, face_too_small")
    return warnings


async def _run_image_and_persist(job: Job, data: bytes, mime: str, caption: str | None, warnings: list, search_id: str | None, user_id: str | None, filters: dict[str, str]) -> None:
    await app.state.image_search.run(job, data, mime, caption, warnings)
    if job.status != JobStatus.complete or not job.result:
        return
    out = to_client_response(job.result)
    out.identified = job.extra.get("identified", [])
    out.photoWarnings = job.extra.get("photoWarnings", [])
    out.lensInput = job.extra.get("lensInput")
    out.lensResults = job.extra.get("lensResults")
    out.lensImageUrl = job.extra.get("lensImageUrl")
    out.warnings = job.result.warnings or None
    sid = (search_id or job.searchId).strip()
    try:
        saved = await persist_people(
            app.state.search_store,
            search_id=sid,
            query="photo search",
            filters=filters,
            mode="image",
            candidates=out.candidates,
            user_id=user_id,
        )
    except Exception:
        log.exception("saving photo search %s failed", sid)
        saved = out.candidates
    job.extra["client"] = json.loads(out.model_copy(update={"searchId": sid, "candidates": saved}).model_dump_json())


@app.post("/v1/people-search/image", response_model=PeopleSearchResponse)
async def people_search_image(
    image: UploadFile = File(..., description="jpeg, png or webp, <= 5 MB"),
    filters: str | None = Form(default=None, description='JSON object, e.g. {"location": "Hanoi"}'),
    caption: str | None = Form(default=None, max_length=200),
    photoWarnings: str | None = Form(default=None, description="Client face pre-check codes: no_face, multiple_faces, face_too_small"),
    searchId: str | None = Form(default=None, max_length=128, description="Client-declared search id stored with candidates"),
    userId: str | None = Form(default=None, max_length=128),
):
    """Photo search: the model names the public people in the photo, each name gets the
    regular deep search, and candidates whose photos match the uploaded face come first.
    Same 202 + polling contract as /v1/people-search. The photo is not stored."""
    if not app.state.mcp.servers_for("web_search"):
        raise HTTPException(503, "no search MCP server connected")
    if app.state.llm is None:
        raise HTTPException(503, "LLM provider is not configured")
    data, mime = await _read_upload(image)
    warnings = _form_warnings(photoWarnings)
    parsed_filters = _form_filters(filters)
    job = app.state.jobs.create(SearchInput(name="photo search", filters=parsed_filters))
    task = asyncio.create_task(_run_image_and_persist(job, data, mime, caption, warnings, searchId, userId, parsed_filters))
    app.state.jobs.attach(job, task)
    return await _await_job(job, task)


# ---------- building blocks: one step of the search each ----------


@app.post("/v1/identify", response_model=IdentifyResponse)
async def identify(req: IdentifyRequest):
    """Who the name may refer to: one LLM call. With withSources (default) the likely people
    also get verified social profiles and photos (2-3 search calls each); sourceUrls stay unverified."""
    try:
        return await app.state.identify.by_name(req)
    except IdentifyError as e:
        raise HTTPException(e.status, str(e))


@app.post("/v1/identify/image", response_model=IdentifyResponse)
async def identify_image(
    image: UploadFile = File(..., description="jpeg, png or webp, <= 5 MB"),
    filters: str | None = Form(default=None),
    caption: str | None = Form(default=None, max_length=200),
    photoWarnings: str | None = Form(default=None),
    withSources: bool = Form(default=True),
):
    """Public people recognised in a photo: one LLM call. photoWarnings lower the confidence.
    With withSources (default) the likely people also get social profiles and photos."""
    data, mime = await _read_upload(image)
    try:
        return await app.state.identify.by_image(
            data, mime, _form_filters(filters), caption, _form_warnings(photoWarnings), with_sources=withSources
        )
    except IdentifyError as e:
        raise HTTPException(e.status, str(e))


@app.post("/v1/sources/social", response_model=SourcesResponse)
async def social_sources(req: SocialSourcesRequest):
    """One profile per social network for a known person. 1-2 search calls, no LLM."""
    try:
        return await app.state.sources.social(req)
    except SourceError as e:
        raise HTTPException(e.status, str(e))


@app.post("/v1/sources/web", response_model=SourcesResponse)
async def web_sources(req: WebSourcesRequest):
    """Public web pages about the name (official site, Wikipedia, ...). 1 search call, no LLM."""
    try:
        return await app.state.sources.web(req)
    except SourceError as e:
        raise HTTPException(e.status, str(e))


@app.post("/v1/images/search", response_model=ImageSearchResponse)
async def image_search(req: ImageSearchRequest):
    """Raw image search, no face check. 1 search call, no LLM."""
    try:
        return await app.state.sources.images(req)
    except SourceError as e:
        raise HTTPException(e.status, str(e))


@app.post("/v1/photos/from-sources", response_model=SourcePhotosResponse)
async def photos_from_sources(req: SourcePhotosRequest):
    """Photos for a person with sources but no photo, taken from those pages (Wikipedia
    lead image, GitHub avatar, og:image). No search, no LLM; login-walled networks are skipped."""
    sources = [ClientSource(url=s.url, type=s.type, title=s.title, verified=s.verified) for s in req.sources]
    photos, checked = await app.state.source_photos.find(req.name, sources, limit=req.limit)
    return SourcePhotosResponse(photos=photos, faceChecked=checked, meta=StepMeta())


@app.post("/v1/profiles/enrich", response_model=EnrichResponse)
async def enrich_profile(req: EnrichRequest):
    """Full profile (biography, facts, career) of one candidate from its sources: one LLM
    call, no search. With historyId the result is saved, so the next call returns it for free."""
    try:
        return await app.state.enricher.run(req)
    except EnrichError as e:
        raise HTTPException(e.status, str(e))


@app.get("/v1/people-search/{search_id}", response_model=PeopleSearchResponse)
async def get_people_search(search_id: str):
    job = app.state.jobs.get(search_id)
    if not job:
        raise HTTPException(404, "search not found")
    if job.status == JobStatus.failed:
        raise HTTPException(int(job.extra.get("httpStatus") or 502), {"searchId": job.searchId, "error": job.error})
    if payload := _client_payload(job):
        return _json(payload)
    if job.status != JobStatus.complete or not job.result:
        return _json(_pending(job), 202)
    return _response(job)


@app.post("/v1/photos", response_model=MorePhotosResponse)
async def more_photos(req: MorePhotosRequest):
    """More photos of a candidate (search results carry one): an image search on its
    imageSearchQuery, keeping images whose face matches `referenceUrl`. No LLM call."""
    try:
        return await app.state.photos.more(req)
    except PhotoError as e:
        raise HTTPException(e.status, str(e))


def _chat_ready() -> None:
    if app.state.llm is None:
        raise HTTPException(503, "LLM provider is not configured")


async def _open_chat(session_id: str, candidate_id: str, user_id: str | None) -> dict:
    try:
        return await app.state.search_store.ensure_chat_session(session_id, candidate_id, user_id)
    except KeyError:
        raise HTTPException(404, "candidate not found") from None
    except PermissionError:
        raise HTTPException(403, "candidate belongs to another user") from None
    except ChatSessionConflict:
        raise HTTPException(409, "session bound to another candidate") from None


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask about a candidate in a client-declared session. Returns 202 immediately; poll GET resultUrl
    (same as search-profile) until the ChatResponse is ready. Cloudflare drops long
    SSE / held POSTs, so the UI must not wait on this connection."""
    _chat_ready()
    rec = await _open_chat(req.sessionId, req.candidateId, req.userId)
    job = app.state.jobs.create(SearchInput(name="chat"))
    task = asyncio.create_task(_run_chat_candidate(job, rec, req.question))
    app.state.jobs.attach(job, task)
    return await _await_client(job, task)


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """Same job as POST /v1/chat (202 + poll). Kept so old clients do not hang on SSE
    through trycloudflare, which cancels the HTTP/2 stream."""
    return await chat(req)
