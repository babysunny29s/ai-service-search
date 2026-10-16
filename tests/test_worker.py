import base64
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core.jobs import JobStore
from app.core.models import ClientCandidate, ClientMeta, JobStatus, PeopleSearchResponse, SearchResult, utcnow
from app.infra.search_store import MemorySearchStore
from app.runtime import Runtime
from app.worker import decode_image, handle, load_image, parse_message, process_body, WorkerError


class FakeMcp:
    def servers_for(self, _cap):
        return [("serper", "web_search")]

    def status(self):
        return {"connected": {"serper": ["web_search", "image_search"]}}


class FakeProfile:
    async def run(self, query, filters, exclude_urls, exclude_names, page):
        return PeopleSearchResponse(
            mode="searchprofile",
            status="ambiguous",
            candidates=[ClientCandidate(id="person_a", displayName=query, confidence=0.9, imageSearchQuery=query)],
            meta=ClientMeta(searchCalls=1, llmCalls=1),
        )


class FakeDeep:
    async def run(self, candidate, filters):
        from app.core.models import DeepSearchResponse

        return DeepSearchResponse(candidate=candidate, socialSources=[], articleSources=[], warnings=[], meta=ClientMeta(searchCalls=1))


class FakeImage:
    async def run(self, job, data, mime, caption, warnings):
        job.result = SearchResult(
            searchId=job.searchId,
            mode="image",
            status="no_match",
            input=job.input,
            candidates=[],
            sources=[],
            queries=[],
            completedAt=utcnow(),
        )
        job.extra["identified"] = []
        job.extra["photoWarnings"] = warnings
        job.touch("complete", JobStatus.complete)


class FakeIdentify:
    async def by_image(self, data, mime, filters, caption, warnings, with_sources=True):
        from app.core.models import IdentifyResponse, StepMeta

        return IdentifyResponse(candidates=[], warnings=[], meta=StepMeta(llmCalls=1))

    async def by_name(self, req):
        from app.core.models import IdentifyResponse, StepMeta

        return IdentifyResponse(candidates=[], warnings=[], meta=StepMeta(llmCalls=1))


class FakeHistory:
    def __init__(self):
        self.created = None

    async def create(self, user_id, query, filters, candidate, search_id):
        self.created = {"userId": user_id, "query": query, "candidate": candidate}
        return "hist_1"

    async def list(self, user_id, limit=50):
        return []

    async def messages(self, history_id, limit=None):
        return []

    async def delete(self, history_id):
        return None


class FakeChat:
    async def load(self, history_id, user_id):
        return {"id": history_id, "userId": user_id, "query": "A", "filters": {}, "candidate": {"displayName": "A"}, "searchId": None, "createdAt": "", "updatedAt": ""}

    async def ask(self, history_id, question, user_id):
        return {"historyId": history_id, "answer": "ok", "answeredFrom": "knowledge", "citations": [], "meta": {"searchQueries": [], "searchCalls": 0, "llmCalls": 1, "latencyMs": 1}}

    async def ask_with(self, rec, thread_id, question, store=None):
        return {
            "sessionId": rec.get("sessionId") or thread_id,
            "candidateId": rec.get("candidateId") or rec.get("id") or thread_id,
            "searchId": rec.get("searchId"),
            "answer": "ok",
            "answeredFrom": "knowledge",
            "citations": [],
            "meta": {"searchQueries": [], "searchCalls": 0, "llmCalls": 1, "latencyMs": 1},
        }


def runtime(**extra) -> Runtime:
    rt = Runtime.__new__(Runtime)
    rt.mcp = FakeMcp()
    rt.llm = SimpleNamespace(name="fake")
    rt.jobs = JobStore()
    rt.profile_search = FakeProfile()
    rt.deep_search = FakeDeep()
    rt.image_search = FakeImage()
    rt.identify = FakeIdentify()
    rt.history = FakeHistory()
    rt.chat = FakeChat()
    rt.search_store = MemorySearchStore()
    rt.settings = Settings(_env_file=None, llm_api_key="")
    for k, v in extra.items():
        setattr(rt, k, v)
    return rt


JPEG = b"\xff\xd8\xff" + b"\x00" * 16
PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")


def test_parse_message_payload_and_flat():
    action, rid, payload = parse_message({"action": "search-profile", "requestId": "r1", "payload": {"query": "Nguyen Van A"}})
    assert action == "search-profile" and rid == "r1" and payload["query"] == "Nguyen Van A"
    action, rid, payload = parse_message({"type": "deep_search", "id": "x", "candidate": {"displayName": "A"}})
    assert action == "deep-search" and rid == "x" and payload["candidate"]["displayName"] == "A"


def test_parse_message_rejects_unknown():
    with pytest.raises(WorkerError) as e:
        parse_message({"action": "wipe-disk"})
    assert e.value.status == 422


def test_process_search_header_is_not_an_action():
    with pytest.raises(WorkerError) as e:
        parse_message({"type": "process_search", "query": "Nguyen Van A"})
    assert e.value.status == 422
    action, rid, payload = parse_message({"type": "process_search", "action": "search-profile", "requestId": "r1", "query": "Nguyen Van A"})
    assert action == "search-profile" and rid == "r1" and payload["query"] == "Nguyen Van A"


@pytest.mark.asyncio
async def test_search_profile_sets_search_id():
    rt = runtime()
    out = await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "srch_client_1"})
    assert out["mode"] == "searchprofile"
    assert out["searchId"] == "srch_client_1"
    assert out["candidates"][0]["id"].startswith("cnd_")
    rec = await rt.search_store.get_candidate(out["candidates"][0]["id"])
    assert rec["searchId"] == "srch_client_1"
    assert rec["query"] == "Nguyen Van A"
    assert out["meta"]["searchCalls"] == 1
    assert "costUsd" not in out["meta"] and "costVnd" not in out["meta"] and "costLines" not in out["meta"]


@pytest.mark.asyncio
async def test_search_profile_requires_search_id():
    with pytest.raises(WorkerError) as e:
        await handle(runtime(), "search-profile", {"query": "Nguyen Van A"})
    assert e.value.status == 422


@pytest.mark.asyncio
async def test_deep_search_needs_candidate():
    cand = {"id": "c1", "displayName": "Nguyen Van A", "confidence": 0.9, "imageSearchQuery": "Nguyen Van A"}
    out = await handle(runtime(), "deep-search", {"candidate": cand})
    assert out["candidate"]["displayName"] == "Nguyen Van A"


@pytest.mark.asyncio
async def test_deep_search_and_chat_by_candidate_id():
    rt = runtime()
    out = await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "srch_2", "userId": "u1"})
    cid = out["candidates"][0]["id"]
    deep = await handle(rt, "deep-search", {"candidateId": cid, "userId": "u1"})
    assert deep["candidate"]["displayName"] == "Nguyen Van A"
    assert deep["searchId"] == "srch_2"
    chat = await handle(rt, "chat", {"sessionId": "sess_1", "candidateId": cid, "question": "ai vay", "userId": "u1"})
    assert chat["candidateId"] == cid
    assert chat["sessionId"] == "sess_1"
    assert chat["answer"] == "ok"
    with pytest.raises(WorkerError) as e:
        await handle(rt, "chat", {"candidateId": cid, "question": "ai vay"})
    assert e.value.status == 422
    with pytest.raises(WorkerError) as e:
        await handle(rt, "deep-search", {"candidateId": "cnd_missing"})
    assert e.value.status == 404


@pytest.mark.asyncio
async def test_same_query_from_another_user_reuses_the_saved_search():
    class Counting(FakeProfile):
        calls = 0

        async def run(self, *a):
            Counting.calls += 1
            return await super().run(*a)

    rt = runtime(profile_search=Counting())
    first = await handle(rt, "search-profile", {"query": "Nguyễn Văn A", "searchId": "s1", "userId": "u1"})
    again = await handle(rt, "search-profile", {"query": "nguyen  van a", "searchId": "s2", "userId": "u2"})
    assert Counting.calls == 1 and again["meta"]["cacheHit"] and again["meta"]["searchCalls"] == 0
    # The second user gets rows of their own for the same person.
    assert again["candidates"][0]["id"] != first["candidates"][0]["id"]
    rec = await rt.search_store.get_candidate(again["candidates"][0]["id"])
    assert rec["userId"] == "u2" and rec["searchId"] == "s2"
    await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "s3", "userId": "u3", "refresh": True})
    await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "s4", "userId": "u4", "filters": {"location": "Hà Nội"}})
    assert Counting.calls == 3


@pytest.mark.asyncio
async def test_deep_search_of_the_same_person_reuses_the_saved_dossier():
    class Counting(FakeDeep):
        calls = 0

        async def run(self, candidate, filters):
            Counting.calls += 1
            out = await super().run(candidate, filters)
            return out.model_copy(update={"candidate": candidate.model_copy(update={"headline": "Founder"})})

    rt = runtime(deep_search=Counting())
    a = await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "s1", "userId": "u1"})
    await handle(rt, "deep-search", {"candidateId": a["candidates"][0]["id"], "userId": "u1"})
    b = await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "s2", "userId": "u2", "refresh": True})
    cid = b["candidates"][0]["id"]
    deep = await handle(rt, "deep-search", {"candidateId": cid, "userId": "u2"})
    assert Counting.calls == 1 and deep["meta"]["cacheHit"] and deep["candidate"]["headline"] == "Founder"
    assert deep["candidate"]["id"] == cid and deep["searchId"] == "s2"
    await handle(rt, "deep-search", {"candidateId": cid, "userId": "u2", "refresh": "true"})
    assert Counting.calls == 2


@pytest.mark.asyncio
async def test_image_search_from_minio_url(monkeypatch):
    class Mini:
        endpoint = "minio.example"
        def is_available(self):
            return True

        async def download_async(self, ref):
            assert ref.startswith("https://")
            return JPEG

    monkeypatch.setattr("minio_handler.get_minio", lambda: Mini())
    out = await handle(
        runtime(),
        "people-search-image",
        {"searchId": "srch_img", "image": "https://minio.example/ai-inhouse/search/req.jpg", "filters": {"location": "Hanoi"}},
    )
    assert out["mode"] == "image"
    assert out["status"] == "no_match"


@pytest.mark.asyncio
async def test_load_image_minio_url(monkeypatch):
    class Mini:
        endpoint = "minio.example"
        def is_available(self):
            return True

        async def download_async(self, ref):
            assert ref == "https://minio.example/ai-inhouse/search/a.jpg"
            return JPEG

    monkeypatch.setattr("minio_handler.get_minio", lambda: Mini())
    data, mime = await load_image({"image": "https://minio.example/ai-inhouse/search/a.jpg"})
    assert mime == "image/jpeg" and data.startswith(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_load_image_public_http_url(monkeypatch):
    """Google CDN / arbitrary https must not go through MinIO."""
    class Mini:
        endpoint = "minios3.infinitydashboard.info"
        def is_available(self):
            return True

        async def download_async(self, ref):
            raise AssertionError(f"MinIO must not fetch {ref}")

    class Resp:
        content = JPEG
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def get(self, url):
            assert "gstatic.com" in url
            return Resp()

    monkeypatch.setattr("minio_handler.get_minio", lambda: Mini())
    monkeypatch.setattr("httpx.AsyncClient", Client)
    data, mime = await load_image({"image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTest&s=10"})
    assert mime == "image/jpeg" and data.startswith(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_image_search_from_base64():
    out = await handle(
        runtime(),
        "people-search-image",
        {"searchId": "srch_img2", "image": base64.b64encode(JPEG).decode(), "filters": {"location": "Hanoi"}},
    )
    assert out["mode"] == "image"
    assert out["status"] == "no_match"


def test_decode_image_errors():
    with pytest.raises(WorkerError) as e:
        decode_image({})
    assert e.value.status == 422
    with pytest.raises(WorkerError) as e:
        decode_image({"image": base64.b64encode(b"not-an-image").decode()})
    assert e.value.status == 415
    with pytest.raises(WorkerError) as e:
        decode_image({"image": base64.b64encode(b"\xff\xd8\xff" + b"x" * 5_000_001).decode()})
    assert e.value.status == 413
    data, mime = decode_image({"image": "data:image/png;base64," + base64.b64encode(PNG).decode()})
    assert mime == "image/png" and data.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_history_actions_are_gone():
    with pytest.raises(WorkerError) as e:
        parse_message({"action": "history.create", "query": "A"})
    assert e.value.status == 422


@pytest.mark.asyncio
async def test_process_body_error_and_ok():
    fail = await process_body(runtime(), b"not json")
    assert fail["ok"] is False and fail["is_success"] is False and fail["status"] == 422
    ok = await process_body(runtime(), json.dumps({"action": "health", "requestId": "r1"}).encode(), correlation_id="amqp")
    assert ok["ok"] is True and ok["is_success"] is True and ok["requestId"] == "r1"
    assert ok["result"]["llm"] == "fake"


@pytest.mark.asyncio
async def test_search_unavailable():
    rt = runtime()
    rt.mcp = SimpleNamespace(servers_for=lambda _c: [], status=lambda: {})
    with pytest.raises(WorkerError) as e:
        await handle(rt, "search-profile", {"query": "Nguyen Van A", "searchId": "srch_x"})
    assert e.value.status == 503
