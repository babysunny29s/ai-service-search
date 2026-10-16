"""Offline tests for search history + chat (fake search and LLM, no network)."""

import asyncio
import json

import pytest

from app.chat.chat import ChatError, ChatService, person_block, search_query
from app.config import Settings
from app.chat.history import HistoryStore
from app.core.llm import ToolCall
from app.infra.search_adapter import SearchHit

CANDIDATE = {
    "id": "nguyen-van-a-1234",
    "displayName": "Nguyen Van A",
    "headline": "Founder of Acme",
    "confidence": 0.9,
    "imageSearchQuery": "Nguyen Van A Acme",
    "keyFacts": [{"key": "employer", "label": "Employer", "value": "Acme", "verified": True}],
    "sources": [{"url": "https://www.linkedin.com/in/nguyenvana", "title": "LinkedIn", "type": "linkedin", "verified": True}],
    "profile": ["Nguyen Van A founded Acme in 2015."],
}


class FakeSearch:
    def __init__(self):
        self.queries = []

    async def search(self, query, kind, logs, locale=None):
        self.queries.append(query)
        return [
            SearchHit(provider="fake", query=query, title="Acme team", url="https://acme.example/team", snippet="Nguyen Van A studied at HUST.", position=1),
            SearchHit(provider="fake", query=query, title="News", url="https://vnexpress.net/tin-tuc/ong-a-123456.html", snippet="news", position=2),
        ], []


class FakeChatLLM:
    """Answers `reply`; with `search_query`, the first call (tools offered) asks for a web search instead."""

    def __init__(self, reply="He studied at HUST.", search_query=None):
        self.reply = reply
        self.search_query = search_query
        self.calls = []
        self.tools = []

    async def chat_stream(self, messages, *, tools=None, max_tokens=800, trace_name="llm:chat", trace_metadata=None):
        self.calls.append(messages)
        self.tools.append(tools)
        if tools and self.search_query is not None:
            yield ToolCall(id="c1", name="web_search", arguments=json.dumps({"query": self.search_query}))
            return
        for word in self.reply.split(" "):
            yield word + " "


def _service(tmp_path, llm=None, search=None):
    store = HistoryStore(str(tmp_path / "h.db"))
    s = Settings(_env_file=None, llm_api_key="", history_db_path=str(tmp_path / "h.db"))
    return store, ChatService(s, store, search or FakeSearch(), llm or FakeChatLLM())


def test_history_crud(tmp_path):
    async def run():
        store, _ = _service(tmp_path)
        hid = await store.create("u1", "Nguyen Van A", {"location": "Hanoi"}, CANDIDATE, "s1")
        assert (await store.get(hid))["candidate"]["displayName"] == "Nguyen Van A"
        assert [r["id"] for r in await store.list("u1")] == [hid]
        assert await store.list("u2") == [] and await store.list(None) == []
        await store.append(hid, [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "citations": []}])
        assert [m["role"] for m in await store.messages(hid)] == ["user", "assistant"]
        assert (await store.list("u1"))[0]["messageCount"] == 2
        await store.delete(hid)
        assert await store.get(hid) is None and await store.messages(hid) == []

    asyncio.run(run())


def test_chat_turn_cost_is_only_that_turn(tmp_path):
    class Priced(FakeChatLLM):
        async def chat_stream(self, messages, *, tools=None, **kw):
            from app.core.tracing import record_llm

            record_llm(
                {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
                model="gemini-3.7-flash",
            )
            async for chunk in super().chat_stream(messages, tools=tools, **kw):
                yield chunk

    async def run():
        store, chat = _service(tmp_path, Priced("He founded Acme."))
        hid = await store.create("u1", "Nguyen Van A", {}, CANDIDATE, None)
        done = (await chat.ask(hid, "What company?", "u1"))
        assert done["meta"]["llmCalls"] == 1 and done["meta"]["searchCalls"] == 0
        assert done["meta"]["promptTokens"] == 1000 and done["meta"]["completionTokens"] == 100
        assert done["meta"]["costVnd"] > 0 and done["meta"]["costUsd"] > 0
        assert done["meta"]["costLines"][0]["model"] == "gemini-3.7-flash"
        assert done["meta"]["costLines"][0]["kind"] == "llm"

    asyncio.run(run())


def test_chat_answers_from_knowledge_without_searching(tmp_path):
    async def run():
        llm, search = FakeChatLLM("He founded Acme."), FakeSearch()
        store, chat = _service(tmp_path, llm, search)
        hid = await store.create("u1", "Nguyen Van A", {}, CANDIDATE, None)
        events = [e async for e in chat.stream(hid, "What company did he found?", "u1")]
        done = events[-1]
        assert done["event"] == "done" and done["answer"] == "He founded Acme."
        assert done["answeredFrom"] == "knowledge" and done["citations"] == []
        assert done["meta"]["llmCalls"] == 1 and done["meta"]["searchQueries"] == []
        assert search.queries == [] and len(llm.calls) == 1 and llm.tools[0]
        prompt = "\n".join(m["content"] for m in llm.calls[0])
        assert "Founder of Acme" in prompt and "linkedin.com/in/nguyenvana" in prompt

    asyncio.run(run())


def test_chat_searches_when_model_does_not_know(tmp_path):
    async def run():
        llm, search = FakeChatLLM("He studied at HUST.", search_query="university education"), FakeSearch()
        store, chat = _service(tmp_path, llm, search)
        hid = await store.create("u1", "Nguyen Van A", {}, CANDIDATE, None)
        events = [e async for e in chat.stream(hid, "Where did he study?", "u1")]
        stages = [e.get("stage") for e in events if e["event"] == "status"]
        assert stages == ["thinking", "searching", "answering"]
        done = events[-1]
        assert done["answer"] == "He studied at HUST." and done["answeredFrom"] == "web"
        assert search.queries == ["Nguyen Van A university education"]
        assert [c["url"] for c in done["citations"]] == ["https://acme.example/team", "https://vnexpress.net/tin-tuc/ong-a-123456.html"]
        assert done["meta"]["llmCalls"] == 2
        grounded = llm.calls[1][-1]["content"]
        assert "studied at HUST" in grounded and llm.tools[1] is None

        llm.search_query = None
        await chat.ask(hid, "And when?", "u1")
        third = llm.calls[2]
        assert {"role": "user", "content": "Where did he study?"} in third
        assert any(m["role"] == "assistant" and "HUST" in m["content"] for m in third)
        assert len(await store.messages(hid)) == 4

    asyncio.run(run())


def test_search_query_always_names_the_person():
    name = "Nguyễn Văn A"
    assert search_query(ToolCall("1", "web_search", '{"query": "Acme CEO"}'), name, "q") == "Nguyễn Văn A Acme CEO"
    assert search_query(ToolCall("1", "web_search", '{"query": "nguyen van a acme"}'), name, "q") == "nguyen van a acme"
    assert search_query(ToolCall("1", "web_search", "not json"), name, "who?") == "Nguyễn Văn A who?"


def test_chat_guards_owner_and_missing_history(tmp_path):
    async def run():
        store, chat = _service(tmp_path)
        hid = await store.create("u1", "Nguyen Van A", {}, CANDIDATE, None)
        with pytest.raises(ChatError) as e:
            await chat.ask(hid, "q", "u2")
        assert e.value.status == 403
        with pytest.raises(ChatError) as e:
            await chat.ask("hist_missing", "q", "u1")
        assert e.value.status == 404
        assert await store.messages(hid) == []

    asyncio.run(run())


def test_chat_failure_saves_nothing(tmp_path):
    class Broken(FakeChatLLM):
        async def chat_stream(self, messages, *, tools=None, **kw):
            raise RuntimeError("down")
            yield ""

    async def run():
        store, chat = _service(tmp_path, Broken())
        hid = await store.create(None, "Nguyen Van A", {}, CANDIDATE, None)
        events = [e async for e in chat.stream(hid, "q", None)]
        assert events[-1]["event"] == "error" and events[-1]["status"] == 502
        assert await store.messages(hid) == []

    asyncio.run(run())


def test_person_block_is_compact():
    block = person_block({**CANDIDATE, "profile": ["x" * 5000]})
    assert "Name: Nguyen Van A" in block and len(block) < 1200


def test_provider_stream_assembles_tool_calls():
    from types import SimpleNamespace as NS

    from app.core.llm import OpenAICompatibleProvider

    def chunk(content=None, tool_calls=None):
        return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])

    def tc(index, id=None, name=None, args=None):
        return NS(index=index, id=id, function=NS(name=name, arguments=args))

    chunks = [
        chunk(tool_calls=[tc(0, "c1", "web_search", '{"que')]),
        chunk(tool_calls=[tc(0, None, None, 'ry": "A CEO"}')]),
        chunk(tool_calls=[tc(1, "c2", "web_search", '{"query": "A school"}')]),
        NS(choices=[]),
    ]

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            async def gen():
                for c in chunks:
                    yield c

            return gen()

    async def run():
        p = OpenAICompatibleProvider(Settings(_env_file=None, llm_api_key="x", llm_reasoning_effort=""))
        p.client = NS(chat=NS(completions=FakeCompletions()))
        out = [o async for o in p.chat_stream([{"role": "user", "content": "q"}], tools=[{"type": "function"}])]
        assert out == [ToolCall("c1", "web_search", '{"query": "A CEO"}'), ToolCall("c2", "web_search", '{"query": "A school"}')]
        assert p.client.chat.completions.kwargs["tool_choice"] == "auto"
        assert p.client.chat.completions.kwargs["stream_options"] == {"include_usage": True}

    asyncio.run(run())


def test_chat_stream_reads_yescale_usage_on_last_chunk():
    from types import SimpleNamespace as NS

    from app.core.llm import OpenAICompatibleProvider

    usage = {
        "prompt_tokens": 6,
        "completion_tokens": 74,
        "total_tokens": 80,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    chunks = [
        NS(choices=[NS(delta=NS(content="ok", tool_calls=None))], usage=None),
        NS(choices=[], usage=usage),
    ]

    class FakeCompletions:
        async def create(self, **kwargs):
            async def gen():
                for c in chunks:
                    yield c

            return gen()

    async def run():
        p = OpenAICompatibleProvider(Settings(_env_file=None, llm_api_key="x", llm_reasoning_effort=""))
        p.client = NS(chat=NS(completions=FakeCompletions()))
        out = [o async for o in p.chat_stream([{"role": "user", "content": "q"}])]
        assert out == ["ok"]
        assert p.tokens_used == 80

    asyncio.run(run())


def test_grounded_answer_reads_result_pages(tmp_path):
    from app.infra.fetcher import FetchedPage

    class FakeFetcher:
        def __init__(self):
            self.urls = []

        async def fetch(self, url):
            self.urls.append(url)
            return FetchedPage(url, "fetched", text="Intro. Nguyen Van A is not married yet but has a boyfriend who co-runs a swim club. Outro.")

    async def run():
        llm, fetcher = FakeChatLLM("Not married.", search_query="husband"), FakeFetcher()
        store = HistoryStore(str(tmp_path / "h.db"))
        s = Settings(_env_file=None, llm_api_key="", chat_fetch_pages=1)
        chat = ChatService(s, store, FakeSearch(), llm, fetcher)
        hid = await store.create(None, "Nguyen Van A", {}, CANDIDATE, None)
        done = await chat.ask(hid, "chồng của người này", None)
        assert done["answeredFrom"] == "web" and fetcher.urls == ["https://acme.example/team"]
        assert "Page: Intro. Nguyen Van A is not married yet but has a boyfriend" in llm.calls[1][-1]["content"]

    asyncio.run(run())
