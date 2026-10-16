"""RabbitMQ worker: consume a request, run search/deep/chat/image, publish the result.

The gateway publishes JSON to headers exchange `AI_SEARCH_EXCHANGE_INPUT`
(`type=process_search`). One queue `AI_SEARCH_QUEUE_IN` handles every search
action. Results go to `AI_SEARCH_EXCHANGE_RESULT` / `AI_SEARCH_QUEUE_OUT`.

    uv run python -m app.worker

Envelope:

    {"action": "search-profile", "requestId": "…", "payload": { …same as HTTP body… }}

`action` aliases: search-profile, people-search-image, deep-search, chat,
identify-image, identify, health. Chat requires sessionId + candidateId.
Image: `image` is a MinIO URL (preferred), a public http(s) image URL, or base64.
Optional `filters`, `caption`, `photoWarnings`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import signal
from typing import Any

from pydantic import BaseModel, ValidationError

from app.chat.chat import ChatError
from app.core.client_format import to_client_response
from app.core.models import (
    ClientCandidate,
    DeepSearchRequest,
    IdentifyRequest,
    JobStatus,
    PhotoWarning,
    SearchInput,
    SearchProfileRequest,
    _clean_filters,
)
from app.search.deep_search import DeepSearchError
from app.search.identify import MAX_UPLOAD_BYTES, IdentifyError, detect_image_mime, parse_photo_warnings
from app.infra.search_store import ChatSessionConflict, deep_metadata, persist_people, reuse_deep, reuse_search
from app.search.search_profile import SearchProfileError
from .runtime import Runtime, start_runtime

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("search-worker")

_ENVELOPE = frozenset({"action", "type", "requestId", "request_id", "id", "payload", "data"})

_ACTIONS = {
    "search-profile": "search-profile",
    "search_profile": "search-profile",
    "searchprofile": "search-profile",
    "people-search-image": "people-search-image",
    "people_search_image": "people-search-image",
    "search-image": "people-search-image",
    "search_image": "people-search-image",
    "image-search": "people-search-image",
    "image_search": "people-search-image",
    "deep-search": "deep-search",
    "deep_search": "deep-search",
    "deepsearch": "deep-search",
    "chat": "chat",
    "identify-image": "identify-image",
    "identify_image": "identify-image",
    "identify": "identify",
    "health": "health",
}


class WorkerError(Exception):
    def __init__(self, status: int, message: str, body: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.body = body or {"error": message}


def parse_message(body: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    action = body.get("action")
    header_type = body.get("type")
    if not action and header_type and str(header_type).strip().lower() in _ACTIONS:
        action = header_type
    request_id = body.get("requestId") or body.get("request_id") or body.get("id")
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else None
    if payload is None and isinstance(body.get("data"), dict):
        payload = body["data"]
    if payload is None:
        payload = {k: v for k, v in body.items() if k not in _ENVELOPE}
    if not action or not str(action).strip():
        raise WorkerError(422, "action required")
    key = str(action).strip().lower()
    mapped = _ACTIONS.get(key)
    if mapped is None:
        raise WorkerError(422, f"unknown action: {action}")
    return mapped, str(request_id) if request_id else None, payload


_COST_KEYS = frozenset({"costUsd", "costVnd", "costLines"})


def _strip_cost(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cost(v) for k, v in value.items() if k not in _COST_KEYS}
    if isinstance(value, list):
        return [_strip_cost(v) for v in value]
    return value


def dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = json.loads(value.model_dump_json())
    return _strip_cost(value)


def reply_body(*, ok: bool, action: str | None, request_id: str | None, status: int, result: Any = None, error: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": ok,
        "is_success": ok,
        "action": action,
        "requestId": request_id,
        "request_id": request_id,
        "status": status,
    }
    if ok:
        out["result"] = result
    else:
        out["error"] = error if error is not None else "failed"
    return out


def _need_search(rt: Runtime) -> None:
    if not rt.mcp.servers_for("web_search"):
        raise WorkerError(503, "no search MCP server connected")


def _need_llm(rt: Runtime) -> None:
    if rt.llm is None:
        raise WorkerError(503, "LLM provider is not configured")


def _filters(raw: Any) -> dict[str, str]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise WorkerError(422, "filters must be a JSON object") from e
    if not isinstance(raw, dict):
        raise WorkerError(422, "filters must be a JSON object")
    return _clean_filters(raw)


def _warnings(raw: Any) -> list[PhotoWarning]:
    if raw is None or raw == "" or raw == []:
        text = None
    elif isinstance(raw, list):
        text = ",".join(str(x) for x in raw)
    else:
        text = str(raw)
    warnings = parse_photo_warnings(text)
    if warnings is None:
        raise WorkerError(422, "photoWarnings: comma-separated no_face, multiple_faces, face_too_small")
    return warnings


def _image_key(payload: dict[str, Any]) -> str:
    for key in ("imageKey", "objectName", "minioObject", "minioKey"):
        value = payload.get(key)
        if value:
            return str(value).strip()
    return ""


def decode_image(payload: dict[str, Any]) -> tuple[bytes, str]:
    raw = payload.get("image")
    if raw is None or raw == "":
        raise WorkerError(422, "image required")
    if isinstance(raw, bytes):
        data = raw
    elif isinstance(raw, str):
        blob = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
        try:
            data = base64.b64decode(blob, validate=False)
        except Exception as e:
            raise WorkerError(422, "image must be base64") from e
    else:
        raise WorkerError(422, "image required")
    return _check_image_bytes(data)


def _check_image_bytes(data: bytes) -> tuple[bytes, str]:
    if not data:
        raise WorkerError(422, "image required")
    if len(data) > MAX_UPLOAD_BYTES:
        raise WorkerError(413, "image must be 5 MB or smaller")
    mime = detect_image_mime(data)
    if not mime:
        raise WorkerError(415, "image must be jpeg, png or webp")
    return data, mime


def _minio_host() -> str:
    from minio_handler import get_minio

    endpoint = (get_minio().endpoint or "").strip()
    if not endpoint:
        return ""
    if "://" in endpoint:
        from urllib.parse import urlsplit

        return (urlsplit(endpoint).hostname or "").lower()
    return endpoint.split("/")[0].split(":")[0].lower()


def _is_minio_url(url: str) -> bool:
    """True when the URL points at our MinIO endpoint (not a random CDN)."""
    host = _minio_host()
    if not host:
        return False
    try:
        from urllib.parse import urlsplit

        got = (urlsplit(url).hostname or "").lower()
    except Exception:
        return False
    return got == host or got.endswith("." + host)


def _minio_ref(payload: dict[str, Any]) -> str:
    raw = payload.get("image")
    if isinstance(raw, str) and raw.strip().startswith(("http://", "https://")) and _is_minio_url(raw.strip()):
        return raw.strip()
    return _image_key(payload)


async def _download_minio(ref: str) -> tuple[bytes, str]:
    from minio_handler import get_minio

    mini = get_minio()
    if not mini.is_available():
        raise WorkerError(503, "MinIO is not configured")
    try:
        data = await mini.download_async(ref)
    except Exception as e:
        raise WorkerError(502, f"MinIO download failed: {type(e).__name__}") from e
    return _check_image_bytes(data)


async def _download_http(url: str) -> tuple[bytes, str]:
    """Fetch a public image URL (e.g. Google thumbnail) for queue clients that skip MinIO upload."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent": "ai-service-search/1.0"}) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
    except httpx.HTTPError as e:
        raise WorkerError(502, f"image download failed: {type(e).__name__}") from e
    return _check_image_bytes(data)


async def load_image(payload: dict[str, Any]) -> tuple[bytes, str]:
    """`image` = MinIO URL (preferred), any public http(s) image URL, or base64."""
    ref = _minio_ref(payload)
    if ref:
        return await _download_minio(ref)
    raw = payload.get("image")
    if isinstance(raw, str) and raw.strip().startswith(("http://", "https://")):
        return await _download_http(raw.strip())
    if raw not in (None, ""):
        return decode_image(payload)
    raise WorkerError(422, "image required (MinIO URL, http(s) image URL, or base64)")


def _with_search_id(out: Any, search_id: str) -> Any:
    if isinstance(out, BaseModel) and "searchId" in type(out).model_fields:
        return out.model_copy(update={"searchId": search_id})
    return out


def _client_search_id(payload: dict[str, Any]) -> str:
    sid = payload.get("searchId") or payload.get("search_id")
    if not sid or not str(sid).strip():
        raise WorkerError(422, "searchId required")
    return str(sid).strip()[:128]


def _candidate_id(payload: dict[str, Any]) -> str:
    cid = payload.get("candidateId") or payload.get("candidate_id")
    if not cid or not str(cid).strip():
        raise WorkerError(422, "candidateId required")
    return str(cid).strip()[:128]


def _client_session_id(payload: dict[str, Any]) -> str:
    sid = payload.get("sessionId") or payload.get("session_id")
    if not sid or not str(sid).strip():
        raise WorkerError(422, "sessionId required")
    return str(sid).strip()[:128]


async def _open_chat(rt: Runtime, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return await rt.search_store.ensure_chat_session(
            _client_session_id(payload),
            _candidate_id(payload),
            payload.get("userId") or payload.get("user_id"),
        )
    except KeyError:
        raise WorkerError(404, "candidate not found") from None
    except PermissionError:
        raise WorkerError(403, "candidate belongs to another user") from None
    except ChatSessionConflict:
        raise WorkerError(409, "session bound to another candidate") from None


async def _session(rt: Runtime, candidate_id: str, user_id: str | None) -> dict[str, Any]:
    try:
        return await rt.search_store.session(candidate_id, user_id)
    except KeyError:
        raise WorkerError(404, "candidate not found") from None
    except PermissionError:
        raise WorkerError(403, "candidate belongs to another user") from None


async def _load_candidate(rt: Runtime, payload: dict[str, Any]) -> tuple[ClientCandidate, dict[str, Any]]:
    cid = payload.get("candidateId") or payload.get("candidate_id")
    if cid and str(cid).strip():
        rec = await _session(rt, str(cid).strip(), payload.get("userId") or payload.get("user_id"))
        return ClientCandidate.model_validate(rec["candidate"]), rec
    req = DeepSearchRequest.model_validate(payload)
    assert req.candidate is not None
    return req.candidate, {}


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


async def _save_people(rt: Runtime, search_id: str, query: str, filters: dict[str, str], mode: str, candidates: list, user_id: str | None) -> list:
    return await persist_people(
        rt.search_store,
        search_id=search_id,
        query=query,
        filters=filters,
        mode=mode,
        candidates=list(candidates),
        user_id=user_id,
    )


async def handle(rt: Runtime, action: str, payload: dict[str, Any]) -> Any:
    try:
        return await _handle(rt, action, payload)
    except WorkerError:
        raise
    except (SearchProfileError, DeepSearchError, IdentifyError, ChatError) as e:
        raise WorkerError(e.status, str(e)) from e
    except ValidationError as e:
        raise WorkerError(422, e.errors()[0]["msg"] if e.errors() else "invalid payload") from e


async def _handle(rt: Runtime, action: str, payload: dict[str, Any]) -> Any:
    if action == "health":
        return {"status": "ok", "mcp": rt.mcp.status(), "llm": getattr(rt.llm, "name", None)}

    if action == "search-profile":
        _need_search(rt)
        sid = _client_search_id(payload)
        req = SearchProfileRequest.model_validate(payload)
        job = rt.jobs.create(SearchInput(name=req.query, filters=req.filters))
        out = None
        if req.page == 1 and not (req.excludeUrls or req.excludeNames or req.refresh):
            out = await reuse_search(rt.search_store, req.query, req.filters, "searchprofile", rt.settings.reuse_search_max_age_h)
        if out is None:
            out = await rt.profile_search.run(req.query, req.filters, req.excludeUrls, req.excludeNames, req.page)
        saved = await _save_people(rt, sid, req.query, req.filters, "searchprofile", out.candidates, payload.get("userId"))
        return dump(_with_search_id(out.model_copy(update={"candidates": saved}), sid))

    if action == "deep-search":
        _need_search(rt)
        candidate, rec = await _load_candidate(rt, payload)
        filters = _filters(payload.get("filters") if payload.get("filters") not in (None, "") else rec.get("filters"))
        job = rt.jobs.create(SearchInput(name=candidate.displayName, filters=filters))
        out = None if _truthy(payload.get("refresh")) else await reuse_deep(rt.search_store, candidate, rt.settings.reuse_deep_max_age_h)
        if out is None:
            out = await rt.deep_search.run(candidate, filters)
        try:
            await rt.search_store.update_candidate(candidate.id, deep_metadata(out))
        except KeyError:
            log.warning("deep-search candidate %s not in store", candidate.id)
        return dump(_with_search_id(out, rec.get("searchId") or job.searchId))

    if action == "people-search-image":
        _need_search(rt)
        _need_llm(rt)
        sid = _client_search_id(payload)
        data, mime = await load_image(payload)
        filters = _filters(payload.get("filters"))
        caption = payload.get("caption")
        if caption is not None:
            caption = str(caption)[:200]
        warnings = _warnings(payload.get("photoWarnings"))
        job = rt.jobs.create(SearchInput(name="photo search", filters=filters))
        await rt.image_search.run(job, data, mime, caption, warnings)
        if job.status == JobStatus.failed:
            raise WorkerError(int(job.extra.get("httpStatus") or 502), job.error or "search failed", {"searchId": job.searchId, "error": job.error})
        if not job.result:
            raise WorkerError(502, job.error or "empty result", {"searchId": job.searchId, "error": job.error or "empty result"})
        out = to_client_response(job.result)
        out.identified = job.extra.get("identified", [])
        out.photoWarnings = job.extra.get("photoWarnings", [])
        out.lensInput = job.extra.get("lensInput")
        out.lensResults = job.extra.get("lensResults")
        out.lensImageUrl = job.extra.get("lensImageUrl")
        out.warnings = job.result.warnings or None
        saved = await _save_people(rt, sid, "photo search", filters, "image", out.candidates, payload.get("userId"))
        return dump(out.model_copy(update={"searchId": sid, "candidates": saved}))

    if action == "identify-image":
        data, mime = await load_image(payload)
        with_sources = payload.get("withSources", True)
        if isinstance(with_sources, str):
            with_sources = with_sources.strip().lower() not in ("0", "false", "no")
        caption = payload.get("caption")
        if caption is not None:
            caption = str(caption)[:200]
        out = await rt.identify.by_image(
            data, mime, _filters(payload.get("filters")), caption, _warnings(payload.get("photoWarnings")), with_sources=bool(with_sources)
        )
        return dump(out)

    if action == "identify":
        req = IdentifyRequest.model_validate(payload)
        return dump(await rt.identify.by_name(req))

    if action == "chat":
        _need_llm(rt)
        rec = await _open_chat(rt, payload)
        question = str(payload.get("question") or "").strip()
        if not question:
            raise WorkerError(422, "question is required")
        return dump(await rt.chat.ask_with(rec, rec["sessionId"], question, store=rt.search_store))

    raise WorkerError(422, f"unknown action: {action}")


def _json_body(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise WorkerError(422, "body must be JSON") from e
    if not isinstance(data, dict):
        raise WorkerError(422, "body must be a JSON object")
    return data


async def process_body(rt: Runtime, raw: bytes, *, correlation_id: str | None = None) -> dict[str, Any]:
    action: str | None = None
    request_id: str | None = correlation_id
    try:
        body = _json_body(raw)
        action, parsed_id, payload = parse_message(body)
        request_id = parsed_id or correlation_id
        result = await handle(rt, action, payload)
        return reply_body(ok=True, action=action, request_id=request_id, status=200, result=result)
    except WorkerError as e:
        return reply_body(ok=False, action=action, request_id=request_id, status=e.status, error=e.body if len(e.body) > 1 else e.body.get("error", str(e)))
    except Exception as e:
        log.exception("worker failed request_id=%s action=%s", request_id, action)
        return reply_body(ok=False, action=action, request_id=request_id, status=500, error=f"{type(e).__name__}: {str(e)[:300]}")


async def run_worker(rt: Runtime | None = None) -> None:
    from queue_handler import QueueHandler

    own = rt is None
    if own:
        rt = await start_runtime()
    assert rt is not None
    queues = QueueHandler()
    stop = asyncio.Event()

    def _stop(*_args: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    async def on_request(data: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(data, ensure_ascii=False).encode()
        rid = data.get("requestId") or data.get("request_id")
        return await process_body(rt, raw, correlation_id=str(rid) if rid else None)

    consumer = asyncio.create_task(queues.consume_messages_with_timeout_protection(on_request))
    stopper = asyncio.create_task(stop.wait())
    try:
        done, _pending = await asyncio.wait({consumer, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is consumer and task.exception():
                raise task.exception()
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass
    finally:
        await queues.close()
        if own:
            await rt.close()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
