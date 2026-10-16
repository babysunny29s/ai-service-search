"""Building-block APIs (identify, sources, images, photo search, profile enrichment)
over the fake search MCP server and a fake LLM. No network."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.core.client_format import to_client_response
from app.config import Settings
from app.search.enrich import ENRICHMENT_VERSION, EnrichError, ProfileEnricher, needs_enrichment
from app.chat.history import HistoryStore
from app.search.identify import IdentifyService, ImagePeopleSearch, detect_image_mime, parse_photo_warnings
from app.core.llm import EnrichedCareer, EnrichedFact, EnrichOutput, RecallOutput, RecalledCandidate
from app.infra.mcp_manager import MCPClientManager, ServerConfig
from app.core.models import (
    ClientCandidate,
    ClientPhoto,
    ClientSource,
    EnrichRequest,
    IdentifyRequest,
    ImageSearchRequest,
    ImageSearchResponse,
    Job,
    JobStatus,
    SearchInput,
    SocialSourcesRequest,
    SourcesResponse,
    StepMeta,
    WebSourcesRequest,
)
from app.search.orchestrator import ProfileSearchOrchestrator
from app.infra.search_adapter import SearchAdapter
from app.search.sources import SourceLookup
from test_e2e import FakeFetcher, FakeLLM

JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 64


class PhotoLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.images: list[str] = []
        self.hints: list[list[str]] = []
        self.enrich_calls = 0

    async def recall_image(self, image_data_uri, filters, caption, hints):
        self.calls += 1
        self.images.append(image_data_uri)
        self.hints.append(hints)
        return RecallOutput(
            candidates=[
                RecalledCandidate(displayName="Cristiano Ronaldo", confidence=0.9, occupation="Footballer", imageSearchQuery="Cristiano Ronaldo footballer"),
                RecalledCandidate(displayName="Lookalike Person", confidence=0.1),
            ]
        )

    async def enrich(self, evidence):
        self.enrich_calls += 1
        await asyncio.sleep(0.05)
        assert evidence["name"] == "Nguyen Van A" and evidence["sources"][0]["url"].startswith("https://")
        return EnrichOutput(
            headline="Founder · Acme",
            profile="Nguyen Van A founded Acme in 2015.\n\nAcme builds payment software.",
            facts=[
                EnrichedFact(key="Company", label="Company", value="Acme Corp"),  # already known: verified value kept
                EnrichedFact(key="yearsActive", label="Years active", value="2015-present"),
                EnrichedFact(key="institution", label="Institution", value="Acme Holdings"),  # synonym of company
                EnrichedFact(key="primaryRole", label="Role", value="Founder"),
                EnrichedFact(key="award", label="Award", value="Fintech Award 2020"),
                EnrichedFact(key="email", label="Email", value="a@acme.com"),
                EnrichedFact(key="phoneHint", label="Phone", value="09xx"),
                EnrichedFact(key="age", label="Age", value="35"),
                EnrichedFact(key="birthDate", label="Born", value="June 2, 1989"),  # no Wikipedia/official source says so
                EnrichedFact(key="bad key!", label="x", value="y"),
            ],
            career=[EnrichedCareer(period="2015", title="Founder", organization="Acme"), EnrichedCareer()],
            tags=["fintech", "founder"],
        )


@asynccontextmanager
async def fake_search():
    manager = MCPClientManager(
        [ServerConfig(name="fake", transport="stdio", command="python", args=["tests/fake_search_mcp.py"], tools={"web_search": "web_search", "image_search": "image_search"})]
    )
    await manager.start()
    try:
        yield SearchAdapter(manager, 60, "vn", "vi", 10)
    finally:
        await manager.stop()


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, llm_api_key="", **kw)


async def test_social_sources_one_profile_per_network():
    async with fake_search() as search:
        out = await SourceLookup(_settings(), search).social(SocialSourcesRequest(name="Cristiano Ronaldo", occupation="Footballer"))
    # Fan groups, posts and Ronaldinho's account are not his profiles.
    assert [(s.type, s.url) for s in out.sources] == [("instagram", "https://www.instagram.com/cristiano/"), ("x", "https://x.com/Cristiano")]
    assert out.meta.searchCalls == 2 and out.meta.llmCalls == 0


async def test_web_sources_keep_pages_about_the_name():
    async with fake_search() as search:
        out = await SourceLookup(_settings(), search).web(WebSourcesRequest(name="Cristiano Ronaldo"))
    # The knowledge-panel link and the first result are the same page; namesakes are dropped.
    assert [(s.type, s.url) for s in out.sources] == [("wikipedia", "https://en.wikipedia.org/wiki/Cristiano_Ronaldo")]
    assert out.meta.searchCalls == 1


async def test_image_search_filters_by_name():
    async with fake_search() as search:
        lookup = SourceLookup(_settings(), search)
        hit = await lookup.images(ImageSearchRequest(query="Nguyễn Văn A Công ty X", name="Nguyễn Văn A"))
        again = await lookup.images(ImageSearchRequest(query="Nguyễn Văn A Công ty X", name="Nguyễn Văn A"))
        miss = await lookup.images(ImageSearchRequest(query="Nguyễn Văn A Công ty X", name="Someone Else"))
    assert [p.url for p in hit.photos] == ["https://img.example/a.jpg"] and hit.photos[0].fromSource is False
    assert again.meta.searchCalls == 0  # served from the search cache
    assert miss.photos == []


async def test_identify_by_name_drops_markdown_links():
    out = await IdentifyService(FakeLLM()).by_name(IdentifyRequest(query="Chipu", withSources=False))
    assert [p.displayName for p in out.candidates] == ["Chi Pu"]
    assert out.candidates[0].sourceUrls == ["https://en.wikipedia.org/wiki/Chi_Pu", "https://www.instagram.com/chipupu/"]
    assert out.meta.llmCalls == 1 and out.meta.searchCalls == 0 and out.warnings == []


async def test_identify_with_sources_adds_socials_and_photos_to_likely_people():
    async with fake_search() as search:
        service = IdentifyService(PhotoLLM(), SourceLookup(_settings(), search), _settings())
        out = await service.by_image(JPEG, "image/jpeg", {}, with_sources=True)
    ronaldo, lookalike = out.candidates
    assert [(s.type, s.url) for s in ronaldo.socialSources] == [("instagram", "https://www.instagram.com/cristiano/"), ("x", "https://x.com/Cristiano")]
    assert all("ronaldo" in f"{p.title} {p.pageUrl}".lower() for p in ronaldo.photos)
    # Below image_min_confidence: not searched.
    assert lookalike.socialSources == [] and lookalike.photos == []
    assert out.meta.searchCalls == 3 and out.warnings == []


def _photo(url, own=False):
    return ClientPhoto(url=url, fromSource=own)


class _Lookup:
    def __init__(self):
        self.image_queries = 0

    async def social(self, req):
        return SourcesResponse(sources=[ClientSource(url="https://www.youtube.com/@chipu", type="youtube", verified=True)], meta=StepMeta(searchCalls=1))

    async def images(self, req):
        self.image_queries += 1
        found = [
            _photo("https://img.example/wiki-copy.jpg"),
            _photo("https://img.example/copy.jpg"),
            _photo("https://img.example/1.jpg"),
            _photo("https://img.example/2.jpg"),
        ]
        return ImageSearchResponse(photos=found, meta=StepMeta(searchCalls=1))


class _Finder:
    def __init__(self, urls):
        self.urls = urls

    async def find(self, name, sources, limit=None):
        self.args = (name, [s.url for s in sources], limit)
        return [_photo(u, own=True) for u in self.urls][:limit], True

    async def distinct(self, photos):
        # "copy.jpg" is the same picture as the first source photo.
        return [p for p in photos if not p.url.endswith("/copy.jpg")]


async def test_identify_photos_from_source_pages_come_first_and_become_the_profile_photo():
    lookup, finder = _Lookup(), _Finder(["https://img.example/wiki-copy.jpg"])
    out = await IdentifyService(FakeLLM(), lookup, _settings(), finder).by_name(IdentifyRequest(query="Chipu"))  # type: ignore[arg-type]
    # Source photo first, then image search fills up to 3 (the same picture, by URL or by face, only once).
    assert [(p.url, p.fromSource) for p in out.candidates[0].photos] == [
        ("https://img.example/wiki-copy.jpg", True),
        ("https://img.example/1.jpg", False),
        ("https://img.example/2.jpg", False),
    ]
    # Verified social profiles plus the model's remembered links (its Wikipedia article).
    assert finder.args == ("Chi Pu", ["https://www.youtube.com/@chipu", "https://en.wikipedia.org/wiki/Chi_Pu", "https://www.instagram.com/chipupu/"], 3)
    assert out.meta.searchCalls == 2


async def test_identify_skips_the_image_search_with_three_source_photos():
    lookup = _Lookup()
    finder = _Finder([f"https://own.example/{i}.jpg" for i in range(5)])
    out = await IdentifyService(FakeLLM(), lookup, _settings(), finder).by_name(IdentifyRequest(query="Chipu"))  # type: ignore[arg-type]
    assert [p.url for p in out.candidates[0].photos] == [f"https://own.example/{i}.jpg" for i in range(3)]
    assert lookup.image_queries == 0 and out.meta.searchCalls == 1


async def test_identify_with_sources_without_search_still_answers():
    out = await IdentifyService(FakeLLM()).by_name(IdentifyRequest(query="Chipu"))
    assert [p.displayName for p in out.candidates] == ["Chi Pu"] and out.warnings


async def test_identify_by_image_applies_photo_warnings():
    llm = PhotoLLM()
    out = await IdentifyService(llm).by_image(JPEG, "image/jpeg", {}, None, ["multiple_faces"])
    assert llm.images[0].startswith("data:image/jpeg;base64,")
    assert llm.hints[0] and "several people" in llm.hints[0][0]
    assert out.candidates[0].confidence == 0.72 and out.photoWarnings == ["multiple_faces"]


async def test_photo_search_runs_deep_search_on_recognised_names():
    settings = _settings(search_strategy="search_first", photo_search_candidates=0, social_lookup_candidates=0, require_source=False)
    llm = PhotoLLM()
    job = Job(searchId="search_img", input=SearchInput(name="photo search"))
    async with fake_search() as search:
        orch = ProfileSearchOrchestrator(settings, search, FakeFetcher(5, 100000, 60), llm)
        runner = ImagePeopleSearch(settings, IdentifyService(llm), orch, None)
        await runner.run(job, JPEG, "image/jpeg", None, [])
    assert job.status == JobStatus.complete, job.error
    r = job.result
    # The 0.1-confidence lookalike gets no search.
    assert r.mode == "image" and r.searchId == "search_img"
    assert [q.query for q in r.queries if q.round == "discovery"] and all("Lookalike" not in q.query for q in r.queries)
    assert r.stats["llmCalls"] == 2  # identify + analyze
    out = to_client_response(r)
    assert out.mode == "image" and out.candidates[0].displayName == "Cristiano Ronaldo"
    assert [p.displayName for p in job.extra["identified"]] == ["Cristiano Ronaldo", "Lookalike Person"]
    assert out.meta.searchCalls == 1 and out.meta.costVnd == 26
    assert out.meta.costLines and out.meta.costLines[0].kind == "serper"


async def test_identify_image_meta_bills_the_llm_turn():
    from app.core.tracing import record_llm

    class Priced(PhotoLLM):
        async def recall_image(self, *a):
            record_llm(
                {"prompt_tokens": 800, "completion_tokens": 80, "total_tokens": 880},
                model="gemini-3.7-flash",
            )
            return await super().recall_image(*a)

    out = await IdentifyService(Priced(), None, _settings()).by_image(JPEG, "image/jpeg", {}, None, [], with_sources=False)
    assert out.meta.llmCalls == 1 and out.meta.searchCalls == 0
    assert out.meta.promptTokens == 800 and out.meta.completionTokens == 80
    assert out.meta.costVnd > 0 and out.meta.costLines[0].kind == "llm"


async def test_photo_search_with_nobody_recognised_is_no_match():
    class Nobody(PhotoLLM):
        async def recall_image(self, *a):
            return RecallOutput()

    llm = Nobody()
    runner = ImagePeopleSearch(_settings(), IdentifyService(llm), None, None)  # type: ignore[arg-type]
    job = Job(searchId="s", input=SearchInput(name="photo search"))
    await runner.run(job, JPEG, "image/jpeg", None, [])
    assert job.status == JobStatus.complete and job.result.status == "no_match" and job.result.candidates == []


class _LensSearch:
    def __init__(self):
        self.urls: list[str] = []

    def has_lens(self):
        return True

    async def lens(self, url, logs):
        from app.infra.search_adapter import SearchHit

        self.urls.append(url)
        rows = [
            ("https://www.linkedin.com/in/tran-thi-b", "Tran Thi B - Designer | LinkedIn", "LinkedIn"),
            ("https://shop.example/dress", "Red dress", "Shop"),
            ("https://www.linkedin.com/in/tran-thi-b/", "duplicate", "LinkedIn"),
        ]
        return [SearchHit(provider="serper", query="lens", title=t, url=u, snippet=s, position=i, image_url=f"https://img.example/{i}.jpg") for i, (u, t, s) in enumerate(rows, 1)]


class _Minio:
    def __init__(self):
        self.uploaded: list[tuple[str, str]] = []
        self.removed: list[str] = []

    def is_available(self):
        return True

    def upload_bytes(self, data, request_id, ext, content_type):
        self.uploaded.append((request_id + ext, content_type))
        return f"https://minio.example/search/{request_id}{ext}", f"search/{request_id}{ext}", True

    def remove(self, obj):
        self.removed.append(obj)


class _Pages:
    async def fetch(self, url):
        from app.infra.fetcher import FetchedPage

        return FetchedPage(url=url, status="fetched", title="Tran Thi B", text="Tran Thi B is a designer in Hanoi.")


async def test_photo_search_with_nobody_recognised_falls_back_to_google_lens():
    from app.core.llm import LensPeopleOutput, LensPerson

    class Nobody(PhotoLLM):
        async def recall_image(self, *a):
            return RecallOutput()

        async def lens_people(self, pages):
            self.pages = pages
            return LensPeopleOutput(
                identified=True,
                candidates=[
                    LensPerson(displayName="Tran Thi B", confidence=0.8, occupation="Designer", profileUrls=["https://www.linkedin.com/in/tran-thi-b", "https://invented.example/"]),
                    LensPerson(displayName="Nobody Linked", confidence=0.8, profileUrls=["https://invented.example/"]),
                ],
            )

    llm, search, minio = Nobody(), _LensSearch(), _Minio()
    runner = ImagePeopleSearch(_settings(), IdentifyService(llm, _Lookup(), _settings()), None, None, search, _Pages(), llm, minio)  # type: ignore[arg-type]
    job = Job(searchId="s", input=SearchInput(name="photo search"))
    await runner.run(job, JPEG, "image/jpeg", None, [])
    assert job.status == JobStatus.complete, job.error
    r = job.result
    # No face model: the whole photo is searched, and kept in MinIO.
    assert job.extra["lensInput"] == "photo" and minio.removed == [] and search.urls == [job.extra["lensImageUrl"]]
    assert [s.url for s in r.sources] == ["https://www.linkedin.com/in/tran-thi-b", "https://shop.example/dress"]
    assert [p["text"] for p in llm.pages][0].endswith("designer in Hanoi.")
    assert r.status == "resolved" and [c.displayName for c in r.candidates] == ["Tran Thi B"]
    out = to_client_response(r)
    # The Lens page, plus the accounts found by the name it gave (deep search starts from those).
    assert [s.url for s in out.candidates[0].sources] == ["https://www.linkedin.com/in/tran-thi-b", "https://www.youtube.com/@chipu"]
    assert [s.url for s in job.extra["lensResults"]] == [s.url for s in r.sources]
    assert r.stats["llmCalls"] == 2 and r.stats["searchCalls"] == 2


CANDIDATE = ClientCandidate.model_validate(
    {
        "id": "nguyen-van-a-1234",
        "displayName": "Nguyen Van A",
        "confidence": 0.9,
        "imageSearchQuery": "Nguyen Van A Acme",
        "keyFacts": [{"key": "company", "label": "Company", "value": "Acme", "verified": True}],
        "sources": [{"url": "https://www.linkedin.com/in/nguyenvana", "title": "LinkedIn", "type": "linkedin", "verified": True}],
    }
)


async def test_enrich_writes_profile_once_and_saves_it():
    store = HistoryStore(":memory:")
    hid = await store.create("u1", "Nguyen Van A", {}, CANDIDATE.model_dump(mode="json"), None)
    llm = PhotoLLM()
    enricher = ProfileEnricher(llm, store)

    # Two concurrent opens share one LLM call.
    first, twin = await asyncio.gather(
        enricher.run(EnrichRequest(historyId=hid, userId="u1")), enricher.run(EnrichRequest(historyId=hid, userId="u1"))
    )
    assert llm.enrich_calls == 1 and {first.meta.llmCalls, twin.meta.llmCalls} == {0, 1}
    c = first.candidate
    assert c.headline == "Founder · Acme" and c.profileSource == "llm" and len(c.profile) == 2
    facts = {f.key: (f.value, f.verified) for f in c.keyFacts}
    assert facts == {
        "company": ("Acme", True),
        "yearsActive": ("2015-present", False),
        "jobTitle": ("Founder", False),
        "awards": ("Fintech Award 2020", False),
        "email": ("a@acme.com", False),
        "phone": ("09xx", False),
        "age": ("35", False),
    }
    assert c.facts == {
        "company": "Acme",
        "yearsActive": "2015-present",
        "primaryRole": "Founder",
        "awards": "Fintech Award 2020",
        "email": "a@acme.com",
        "phone": "09xx",
        "age": "35",
        "socialProfiles": "LinkedIn · @nguyenvana",
    }
    assert [(x.period, x.title) for x in c.career] == [("2015", "Founder")]
    assert c.tags == ["fintech", "founder"] and c.enrichmentVersion == ENRICHMENT_VERSION and not needs_enrichment(c)

    # Saved on the history row: the next open is free.
    again = await enricher.run(EnrichRequest(historyId=hid, userId="u1"))
    assert again.cached and again.meta.llmCalls == 0 and llm.enrich_calls == 1
    assert (await store.get(hid))["candidate"]["enrichedAt"] == c.enrichedAt

    with pytest.raises(EnrichError) as e:
        await enricher.run(EnrichRequest(historyId=hid, userId="someone-else"))
    assert e.value.status == 403
    store.close()


async def test_enrich_keeps_the_wikipedia_profile():
    wiki = CANDIDATE.model_copy(update={"profile": ["Lead section."], "profileSource": "wikipedia"})
    out = await ProfileEnricher(PhotoLLM(), HistoryStore(":memory:")).run(EnrichRequest(candidate=wiki))
    assert out.candidate.profile == ["Lead section."] and out.candidate.profileSource == "wikipedia"
    assert out.candidate.career


def test_photo_upload_helpers():
    assert detect_image_mime(JPEG) == "image/jpeg"
    assert detect_image_mime(b"RIFF\0\0\0\0WEBPVP8 ") == "image/webp"
    assert detect_image_mime(b"<html>") is None
    assert parse_photo_warnings("face_too_small, no_face,no_face") == ["no_face", "face_too_small"]
    assert parse_photo_warnings("") == [] and parse_photo_warnings("blurry") is None
