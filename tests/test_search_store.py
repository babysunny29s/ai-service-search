import pytest

from app.core.models import ClientCandidate
from app.infra.search_store import ChatSessionConflict, MemorySearchStore, persist_people


def _person(pid: str, name: str) -> ClientCandidate:
    return ClientCandidate(id=pid, displayName=name, confidence=0.8, imageSearchQuery=name)


@pytest.mark.asyncio
async def test_persist_assigns_candidate_id_and_upserts():
    store = MemorySearchStore()
    first = await persist_people(
        store,
        search_id="srch_1",
        query="Nguyen Van A",
        filters={"location": "Hanoi"},
        mode="searchprofile",
        candidates=[_person("person_a", "Nguyen Van A")],
        user_id="u1",
    )
    assert first[0].id.startswith("cnd_")
    cid = first[0].id
    again = await persist_people(
        store,
        search_id="srch_1",
        query="Nguyen Van A",
        filters={"location": "Hanoi"},
        mode="searchprofile",
        candidates=[_person("person_a", "Nguyen Van A")],
        user_id="u1",
    )
    assert again[0].id == cid
    rec = await store.session(cid, "u1")
    assert rec["searchId"] == "srch_1"
    assert rec["candidate"]["displayName"] == "Nguyen Van A"
    with pytest.raises(PermissionError):
        await store.session(cid, "other")
    await store.update_candidate(cid, {**rec["candidate"], "headline": "CTO"})
    rec = await store.get_candidate(cid)
    assert rec["candidate"]["headline"] == "CTO"


@pytest.mark.asyncio
async def test_new_candidate_id_skips_ids_already_stored(monkeypatch):
    store = MemorySearchStore()
    occupied = "cnd_aaaaaaaaaaaaaaaa"
    store.candidates[occupied] = {
        "candidate_id": occupied,
        "search_id": "srch_other",
        "person_key": "someone_else",
        "candidate_information": {"id": occupied, "displayName": "Other"},
        "created_at": "t",
        "updated_at": "t",
    }
    drawn = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])
    monkeypatch.setattr("app.infra.search_store.secrets.token_hex", lambda _n: next(drawn))
    saved = await persist_people(
        store,
        search_id="srch_1",
        query="Nguyen Van A",
        filters={},
        mode="searchprofile",
        candidates=[_person("person_a", "Nguyen Van A")],
        user_id="u1",
    )
    assert saved[0].id == "cnd_bbbbbbbbbbbbbbbb"
    assert store.candidates[occupied]["candidate_information"]["displayName"] == "Other"


@pytest.mark.asyncio
async def test_chat_session_and_history():
    store = MemorySearchStore()
    people = await persist_people(
        store,
        search_id="srch_1",
        query="Nguyen Van A",
        filters={},
        mode="searchprofile",
        candidates=[_person("person_a", "Nguyen Van A")],
        user_id="u1",
    )
    cid = people[0].id
    other = await persist_people(
        store,
        search_id="srch_1",
        query="Nguyen Van A",
        filters={},
        mode="searchprofile",
        candidates=[_person("person_b", "Nguyen Van B")],
        user_id="u1",
    )
    opened = await store.ensure_chat_session("sess_1", cid, "u1")
    assert opened["sessionId"] == "sess_1" and opened["candidateId"] == cid
    again = await store.ensure_chat_session("sess_1", cid, "u1")
    assert again["sessionId"] == "sess_1"
    with pytest.raises(ChatSessionConflict):
        await store.ensure_chat_session("sess_1", other[0].id, "u1")
    await store.append("sess_1", [{"role": "user", "content": "ai vay"}, {"role": "assistant", "content": "A"}])
    msgs = await store.messages("sess_1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert await store.messages("sess_other") == []
