"""End-to-end pipeline over a real MCP stdio connection with fake LLM/pages.
Also asserts the call budget (search calls / LLM calls) per request."""

from app.core.client_format import to_client_response
from app.config import Settings
from app.infra.fetcher import FetchedPage, PageFetcher
from app.core.llm import AnalysisOutput, AnalyzedCandidate, AnalyzedCareer, AnalyzedFact, KnowledgePlan, RecallOutput, RecalledCandidate
from app.infra.mcp_manager import MCPClientManager, ServerConfig
from app.core.models import Job, JobStatus, SearchInput
from app.search.orchestrator import ProfileSearchOrchestrator
from app.search.preview import PreviewSearch
from app.infra.search_adapter import SearchAdapter
from app.core.urls import can_fetch

FOUNDER_PAGE = (
    "Nguyễn Văn A là nhà sáng lập và CEO của Công ty X, một startup fintech tại Hồ Chí Minh.\n"
    "Trước đó anh làm kỹ sư phần mềm tại Ngân hàng Y (2015-2019)."
)
LECTURER_PAGE = "ThS. Nguyễn Văn A (sinh năm 1990) là giảng viên khoa Hóa học, Đại học Z, Hà Nội. Nghiên cứu về hóa hữu cơ và vật liệu mới."
NEWS_PAGE = "Nguyễn Văn A, nhà sáng lập Công ty X, tốt nghiệp Đại học Bách khoa TP.HCM năm 2014 và từng làm việc trong ngành ngân hàng."
CR7_PAGE = "Cristiano Ronaldo dos Santos Aveiro is a Portuguese professional footballer who plays as a forward for Al-Nassr and captains the Portugal national team."
R9_PAGE = "Ronaldo Luís Nazário de Lima is a Brazilian former professional footballer who played as a striker, twice winner of the FIFA World Player of the Year."


class FakeFetcher(PageFetcher):
    async def _fetch(self, url: str) -> FetchedPage:
        if not can_fetch(url):
            return FetchedPage(url, "blocked")
        pages = {
            "nguyenvana.dev": FetchedPage(url, "fetched", title="About", text=FOUNDER_PAGE, profile_links=["https://linkedin.com/in/nguyenvana"]),
            "daihocz": FetchedPage(url, "fetched", title="Giảng viên", text=LECTURER_PAGE),
            "hoidoanhnhan.org": FetchedPage(url, "fetched", title="CafeF", text=NEWS_PAGE),
            "Cristiano_Ronaldo": FetchedPage(url, "fetched", title="Cristiano Ronaldo - Wikipedia", text=CR7_PAGE, image="https://upload.wikimedia.org/cr7-og.jpg"),
            "Brazilian_footballer": FetchedPage(url, "fetched", title="Ronaldo - Wikipedia", text=R9_PAGE, image="https://upload.wikimedia.org/r9.jpg"),
        }
        return next((p for k, p in pages.items() if k in url), FetchedPage(url, "failed", error="nope"))

    async def wikipedia_summary(self, url: str) -> tuple[str | None, str]:
        if "wikipedia.org/wiki/" not in url:
            return None, ""
        title = url.rsplit("/", 1)[-1]
        extract = {"Cristiano_Ronaldo": f"{CR7_PAGE}\nHe has won five Ballon d'Or awards."}.get(title, "")
        return f"https://upload.wikimedia.org/thumb/{title}.jpg", extract


def _f(field, value, sid, ev):
    return AnalyzedFact(field=field, value=value, sourceId=sid, evidence=ev)


class FakeLLM:
    """Plays the single `analyze` call. Deliberately includes a few bad outputs
    the service must filter (fabricated facts, email, markdown query)."""

    name = "fake"

    def __init__(self):
        self.calls = 0

    async def plan(self, query_name, filters, max_queries):
        self.calls += 1
        if query_name == "Taylor Swift":
            ts = RecalledCandidate(
                displayName="Taylor Swift",
                headline="American singer-songwriter",
                confidence=0.99,
                nationality="American",
                occupation="Singer-songwriter",
                knownFor="Pop and country albums",
                sourceUrls=["https://en.wikipedia.org/wiki/Taylor_Swift", "https://www.instagram.com/taylorswift/"],
                imageSearchQuery="Taylor Swift American singer",
                profile="Taylor Swift is an American singer-songwriter.\n\nShe is known for **Folklore** [src_1].",
            )
            return KnowledgePlan(candidates=[ts], searchQueries=[], reason="famous singer")
        if query_name == "Ronaldo":
            known = [
                RecalledCandidate(
                    displayName="Cristiano Ronaldo",
                    confidence=0.95,
                    occupation="Footballer",
                    sourceUrls=["https://www.tiktok.com/@cristiano", "http://www.Instagram.com/cristiano/reels/"],
                ),
                RecalledCandidate(displayName="Ronaldo Testperson", confidence=0.9, occupation="Footballer", sourceUrls=["https://x.com/ronaldo_test"]),
            ]
            queries = ["Ronaldo footballer official", "Cristiano Ronaldo (site:instagram.com OR site:x.com)"]
            return KnowledgePlan(candidates=known, searchQueries=queries, reason="several footballers")
        return KnowledgePlan(candidates=[], searchQueries=['"Nguyễn Văn A" Công ty X', "fintech founder"], reason="not a public figure I know")

    async def recall(self, query_name, filters):
        self.calls += 1
        if "Chipu" not in query_name:
            return RecallOutput()
        return RecallOutput(
            candidates=[
                RecalledCandidate(
                    displayName="Chi Pu",
                    headline="Vietnamese singer and actress",
                    confidence=0.95,
                    nationality="Vietnamese",
                    occupation="Singer, actress",
                    knownFor="Pop music",
                    sourceUrls=["https://en.wikipedia.org/wiki/Chi_Pu", "https://www.instagram.com/chipupu/", "[bad](https://x.y)"],
                    imageSearchQuery="Chi Pu Vietnamese singer",
                )
            ]
        )

    async def analyze(self, query_name, filters, sources):
        self.calls += 1
        sid = {k: s["sourceId"] for s in sources for k in ("linkedin", "nguyenvana.dev", "hoidoanhnhan", "daihocz", "Cristiano_Ronaldo", "Brazilian_footballer", "Ronaldinho") if k in s["url"]}
        assert all("content" not in s for s in sources)
        assert not any("Private Person" in s["snippet"] for s in sources)
        out = []
        if "linkedin" in sid:
            s = sid["linkedin"]
            out.append(
                AnalyzedCandidate(
                    nameAsWritten="Nguyễn Văn A",
                    sourceIds=[s],
                    facts=[_f("jobTitle", "Founder", s, "Founder tại Công ty X"), _f("company", "Công ty X", s, "Founder tại Công ty X")],
                    headline="Founder · Công ty X",
                    imageSearchQuery="[Nguyễn Văn A](https://evil.example)",
                )
            )
        if "nguyenvana.dev" in sid:
            s = sid["nguyenvana.dev"]
            facts = [
                _f("company", "Công ty X", s, "nhà sáng lập và CEO của Công ty X"),
                _f("field", "fintech", s, "một startup fintech"),
                _f("location", "Hồ Chí Minh", s, "fintech tại Hồ Chí Minh"),
                _f("email", "a@x.com", s, "fabricated"),
                _f("bio", "Hallucinated", s, "this text does not exist"),
            ]
            ids = [s]
            if "hoidoanhnhan" in sid:
                c = sid["hoidoanhnhan"]
                ids.append(c)
                facts.append(_f("education", "Đại học Bách khoa TP.HCM", c, "tốt nghiệp Đại học Bách khoa TP.HCM"))
                facts.append(_f("company", "Công ty X", c, "nhà sáng lập Công ty X"))
            out.append(
                AnalyzedCandidate(
                    nameAsWritten="Nguyễn Văn A",
                    sourceIds=ids,
                    ownSiteSourceIds=[s],
                    facts=facts,
                    career=[AnalyzedCareer(title="Kỹ sư phần mềm", organization="Ngân hàng Y", period="2015-2019", sourceId=s, evidence="kỹ sư phần mềm tại Ngân hàng Y (2015-2019)")],
                    headline="Founder · Công ty X",
                    disambiguationNote="Vietnamese fintech founder of Công ty X.",
                    occupation="Founder",
                    knownFor="Công ty X",
                    imageSearchQuery="Nguyễn Văn A",
                    profile=[f"Nguyễn Văn A founded Công ty X [{s}] [src_999].", "He studied engineering [src_999]."],
                    tags=["Fintech", "fintech", "founder"],
                )
            )
        if "daihocz" in sid:
            s = sid["daihocz"]
            out.append(
                AnalyzedCandidate(
                    nameAsWritten="Nguyễn Văn A",
                    sourceIds=[s],
                    facts=[
                        _f("jobTitle", "Giảng viên", s, "giảng viên khoa Hóa học"),
                        _f("company", "Đại học Z", s, "Đại học Z"),
                        _f("location", "Hà Nội", s, "Hà Nội"),
                        _f("born", "1990", s, "sinh năm 1990"),
                    ],
                    headline="Giảng viên · Đại học Z",
                    imageSearchQuery="Nguyễn Văn A",
                )
            )
        if "Cristiano_Ronaldo" in sid:
            s = sid["Cristiano_Ronaldo"]
            out.append(
                AnalyzedCandidate(
                    nameAsWritten="Cristiano Ronaldo",
                    sourceIds=[s],
                    facts=[
                        _f("occupation", "footballer", s, "professional footballer"),
                        _f("nationality", "Portuguese", s, "Portuguese professional footballer"),
                        _f("knownFor", "Al-Nassr", s, "plays as a forward for Al-Nassr"),
                        _f("teams", "Al-Nassr", s, "plays as a forward for Al-Nassr"),
                        _f("teams", "Portugal national team", s, "captains the Portugal national team"),
                        _f("specialization", "Forward", s, "plays as a forward"),
                        _f("award", "Invented award", s, "never written anywhere"),
                        _f("born", "5 February 1985, Funchal, Portugal", s, "Born: 5 February 1985 (age 41), Funchal, Portugal"),
                        _f("fullName", "Cristiano Ronaldo dos Santos Aveiro", s, "Cristiano Ronaldo dos Santos Aveiro"),
                    ],
                    headline="Portuguese professional footballer · Al-Nassr",
                    occupation="Professional footballer",
                    knownFor="Al-Nassr",
                    imageSearchQuery="Cristiano Ronaldo footballer",
                )
            )
        if "Brazilian_footballer" in sid:
            s = sid["Brazilian_footballer"]
            out.append(
                AnalyzedCandidate(
                    nameAsWritten="Ronaldo Luís Nazário de Lima",
                    sourceIds=[s],
                    facts=[_f("occupation", "footballer", s, "former professional footballer"), _f("nationality", "Brazilian", s, "Brazilian former professional footballer")],
                    headline="Brazilian former footballer",
                    imageSearchQuery="Ronaldo Nazário footballer",
                )
            )
        if "Ronaldinho" in sid:
            out.append(AnalyzedCandidate(nameAsWritten="Ronaldinho", sourceIds=[sid["Ronaldinho"]]))
        return AnalysisOutput(candidates=out)


async def _run(job: Job, orch_holder: dict | None = None, **overrides) -> Job:
    multi_call = {"max_queries": 2, "enrich_candidates": 2, "social_lookup_candidates": 2, "search_strategy": "search_first", "photo_search_candidates": 0, "llm_max_calls": 3, "require_photo": False, "require_source": False}
    settings = Settings(_env_file=None, llm_api_key="", **{**multi_call, **overrides})
    manager = MCPClientManager(
        [ServerConfig(name="fake", transport="stdio", command="python", args=["tests/fake_search_mcp.py"], tools={"web_search": "web_search", "image_search": "image_search"})]
    )
    await manager.start()
    try:
        orch = (orch_holder or {}).get("orch") or ProfileSearchOrchestrator(settings, SearchAdapter(manager, 60, "vn", "vi", 10), FakeFetcher(5, 100000, 60), FakeLLM())
        orch.search.manager = manager
        if orch_holder is not None:
            orch_holder["orch"] = orch
        await orch.run(job)
    finally:
        await manager.stop()
    assert job.status == JobStatus.complete, job.error
    return job


async def test_name_only_discovery_and_conditional_enrichment():
    # Threshold raised so the thin founder profile triggers enrichment.
    job = await _run(Job(searchId="search_test", input=SearchInput(name="Nguyễn Văn A")), enrich_min_identifying=5)
    r = job.result
    assert r is not None
    # Budget: 2 discovery + up to 2 enrichment searches, exactly 2 LLM calls.
    assert r.stats["searchCalls"] <= 4
    assert r.stats["llmCalls"] == 2
    assert {q.round for q in r.queries} == {"discovery", "enrichment"}
    assert any(q.round == "enrichment" and '"Công ty X"' in q.query for q in r.queries)
    assert not any("example.com/b" in s.url for s in r.sources)

    assert len(r.candidates) == 2
    founder, lecturer = r.candidates
    # LinkedIn merged into the founder through the cross-link from their own site.
    assert {s.url for s in founder.sources} == {
        "https://linkedin.com/in/nguyenvana",
        "https://nguyenvana.dev/about",
        "https://hoidoanhnhan.org/hoi-vien/nguyen-van-a",
    }
    assert founder.fields["education"].value == "Đại học Bách khoa TP.HCM"
    assert founder.fields["company"].value == "Công ty X" and len(founder.fields["company"].sourceIds) == 3
    assert founder.fields["email"].status == "not_found" and founder.fields["website"].status == "not_found"
    # No Wikipedia article: the model's profile is kept.
    assert founder.narrative and founder.narrativeSource == "llm"
    assert founder.tags == ["fintech", "founder"]
    assert any(p.platform == "linkedin" for p in founder.socialProfiles)
    assert founder.careerArc and founder.careerArc[0].organization == "Ngân hàng Y"
    assert founder.nationality is None
    assert lecturer.fields["company"].value == "Đại học Z"
    # A birth year on an ordinary page is not taken; only Wikipedia/knowledge panel/official sites count.
    assert lecturer.fields["born"].status != "found"
    assert founder.imageSearchQuery != lecturer.imageSearchQuery

    assert r.status == "ambiguous"
    assert "Founder · Công ty X" in (r.clarificationQuestion or "")
    assert "Giảng viên · Đại học Z" in (r.clarificationQuestion or "")


async def test_default_thresholds_skip_enrichment():
    job = await _run(Job(searchId="s2", input=SearchInput(name="Nguyễn Văn A")))
    assert job.result.stats == {**job.result.stats, "searchCalls": 2, "llmCalls": 1}


async def test_one_llm_call_budget_skips_plan_and_enrichment():
    # llm_first and a thin profile would normally cost 2-3 LLM calls.
    job = await _run(
        Job(searchId="budget1", input=SearchInput(name="Nguyễn Văn A")), llm_max_calls=1, search_strategy="llm_first", enrich_min_identifying=5
    )
    r = job.result
    assert r.stats["llmCalls"] == 1
    assert not any(q.round == "enrichment" and q.kind == "web" and q.purpose != "socials" for q in r.queries)
    assert r.candidates


async def test_single_call_config_makes_exactly_one_search_call():
    single = {"max_queries": 1, "enrich_candidates": 0, "social_lookup_candidates": 0}
    for name in ("Nguyễn Văn A", "Ronaldo"):
        job = await _run(Job(searchId=f"one_{name}", input=SearchInput(name=name)), **single)
        assert job.result.stats["searchCalls"] == 1 and job.result.stats["llmCalls"] == 1
        assert job.result.candidates


async def test_llm_first_known_person_needs_no_search():
    job = await _run(Job(searchId="ts", input=SearchInput(name="Taylor Swift")), search_strategy="llm_first")
    r = job.result
    # The model asked for no search, so nothing is searched.
    assert r.queries == [] and r.stats["searchCalls"] == 0 and r.stats["llmCalls"] == 1
    # Nothing was searched, so no remembered link could be verified: a person without
    # any source is never returned.
    assert r.candidates == [] and to_client_response(r).candidates == []
    assert any("hidden" in w for w in r.warnings)


async def test_llm_first_searches_only_what_the_model_asks_for():
    # Default enrichment/social settings: the code must not add searches of its own.
    job = await _run(Job(searchId="nva", input=SearchInput(name="Nguyễn Văn A")), search_strategy="llm_first")
    r = job.result
    assert [q.query for q in r.queries] == ['"Nguyễn Văn A" Công ty X', "Nguyễn Văn A fintech founder"]
    assert r.stats["searchCalls"] == 2 and r.stats["llmCalls"] == 2
    assert any(c.fields["company"].value == "Công ty X" for c in r.candidates)


async def test_llm_first_keeps_unconfirmed_known_people_with_capped_confidence():
    job = await _run(Job(searchId="r", input=SearchInput(name="Ronaldo")), search_strategy="llm_first", enrich_candidates=0, social_lookup_candidates=0)
    r = job.result
    assert [q.purpose for q in r.queries] == ["model", "model"]
    by_name = {c.displayName: c for c in r.candidates}
    # Confirmed by search: scored from evidence, not duplicated.
    assert [n for n in by_name if "Cristiano" in n] == ["Cristiano Ronaldo"]
    assert by_name["Cristiano Ronaldo"].fields["nationality"].status == "found"
    # Remembered links: kept only when a search result points to the same profile.
    cr7 = [p.url for p in by_name["Cristiano Ronaldo"].socialProfiles]
    assert "https://instagram.com/cristiano/reels" not in cr7 and any("instagram.com/cristiano" in u for u in cr7)
    assert not any("tiktok" in u for u in cr7)
    # Known to the model only and its link was not found: no source left, so not returned.
    assert "Ronaldo Testperson" not in by_name


async def test_no_usable_search_results_falls_back_to_model_knowledge():
    job = await _run(Job(searchId="recall", input=SearchInput(name="Chipu")), max_queries=1, social_lookup_candidates=0)
    r = job.result
    assert r.stats["searchCalls"] == 1
    # The model knows the person but none of its links were found: no source, not returned.
    assert r.candidates == [] and any("hidden" in w for w in r.warnings)


async def test_public_figure_one_search_one_llm_and_client_contract():
    holder: dict = {}
    job = await _run(Job(searchId="search_cr", input=SearchInput(name="Ronaldo")), holder)
    r = job.result
    # Knowledge panel on the first query -> no further discovery, no enrichment;
    # only one social-profile lookup per strong candidate.
    assert r.stats["llmCalls"] == 1
    assert [q.purpose for q in r.queries].count("general") == 1
    assert r.stats["searchCalls"] == 1 + [q.purpose for q in r.queries].count("socials")

    cands = to_client_response(r).model_dump()["candidates"]
    names = [c["displayName"] for c in cands]
    assert names[:2] == ["Cristiano Ronaldo", "Ronaldo Luís Nazário de Lima"]
    assert "Ronaldinho" not in names
    cr7, r9 = cands[0], cands[1]
    assert cr7["profile"] == [CR7_PAGE, "He has won five Ballon d'Or awards."] and cr7["profileSource"] == "wikipedia"
    assert r9["profile"] == []
    assert cr7["confidence"] > r9["confidence"]
    assert cr7["nationality"] == "Portuguese"
    assert cr7["avatarUrl"] == "https://upload.wikimedia.org/cr7.jpg"
    # Social profiles first, then the Wikipedia article; posts, fan groups and other people's profiles are dropped.
    assert [s["url"] for s in cr7["sources"]][:3] == ["https://instagram.com/cristiano", "https://x.com/Cristiano", "https://en.wikipedia.org/wiki/Cristiano_Ronaldo"]
    # No social profile found -> the wiki page comes first.
    assert r9["platforms"] == [] and r9["sources"][0]["url"] == "https://en.wikipedia.org/wiki/Ronaldo_(Brazilian_footballer)"
    # No knowledge-panel image for him -> free Wikipedia lead image, no image search.
    assert r9["avatarUrl"] == "https://upload.wikimedia.org/thumb/Ronaldo_(Brazilian_footballer).jpg"
    for c in cands:
        q = c["imageSearchQuery"]
        assert "http" not in q and "[" not in q and len(q) <= 80
    assert len({c["imageSearchQuery"] for c in cands}) == len(cands)

    out = to_client_response(r)
    assert out.meta.searchCalls == r.stats["searchCalls"] and out.meta.llmCalls == 1 and not out.meta.cacheHit
    assert cr7["platforms"] == ["instagram", "x"]
    assert [s["type"] for s in cr7["sources"]][:3] == ["instagram", "x", "wikipedia"]
    assert all(s["verified"] for s in cr7["sources"])
    assert {"key": "nationality", "label": "Nationality", "value": "Portuguese", "verified": True} in cr7["keyFacts"]
    # Every onesearch Pro fact the sources support, keyed like its FactsTable; nothing unsupported.
    facts = cr7["facts"]
    assert set(facts["teams"].split(", ")) == {"Al-Nassr", "Portugal national team"}
    assert facts["specialization"] == "Forward" and facts["nationality"] == "Portuguese"
    assert facts["socialProfiles"] == "Instagram · @cristiano, X · @Cristiano"
    assert facts["audience"] == "Instagram: 650M followers"
    assert facts["born"] == "5 February 1985, Funchal, Portugal"
    assert facts["fullName"] == "Cristiano Ronaldo dos Santos Aveiro"
    assert "awards" not in facts and "award" not in facts

    # Same query again: served from the result cache, zero new calls.
    llm = holder["orch"].llm
    before = llm.calls
    again = await _run(Job(searchId="search_cr2", input=SearchInput(name="  ronaldo ")), holder)
    assert again.toolCalls == [] and llm.calls == before
    assert again.result.searchId == "search_cr2"
    out_again = to_client_response(again.result)
    assert out_again.meta.cacheHit and out_again.meta.searchCalls == 0 and out_again.meta.llmCalls == 0
    # The stable id survives across searches.
    assert [c.id for c in out_again.candidates] == [c.id for c in out.candidates]


async def test_preview_mode_needs_no_llm():
    settings = Settings(_env_file=None, llm_api_key="")
    manager = MCPClientManager(
        [ServerConfig(name="fake", transport="stdio", command="python", args=["tests/fake_search_mcp.py"], tools={"web_search": "web_search", "image_search": "image_search"})]
    )
    await manager.start()
    try:
        preview = PreviewSearch(settings, SearchAdapter(manager, 60, "vn", "vi", 10))
        out = await preview.run("Cristiano Ronaldo", {}, max_queries=2)
        again = await preview.run("cristiano  ronaldo", {}, max_queries=2)
    finally:
        await manager.stop()
    assert out.mode == "preview" and out.meta.llmCalls == 0 and out.meta.searchCalls == 2
    # Instagram and X share the handle -> one person; fan groups, posts and Ronaldinho are not candidates.
    assert [c.displayName for c in out.candidates] == ["Cristiano Ronaldo"]
    c = out.candidates[0]
    assert c.id.startswith("wp-") and c.platforms == ["instagram", "x"]
    assert [s.url for s in c.sources] == ["https://www.instagram.com/cristiano/", "https://x.com/Cristiano"]
    assert again.meta.cacheHit and again.meta.searchCalls == 0



async def test_people_without_photo_or_source_are_hidden():
    def shown(c):
        return c.photos and any(s.provider != "llm_knowledge" for s in c.sources)

    kept = (await _run(Job(searchId="search_kept", input=SearchInput(name="Ronaldo")), require_photo=True, require_source=True)).result
    assert len(kept.candidates) == 2 and all(shown(c) for c in kept.candidates)
    r = (await _run(Job(searchId="search_hidden", input=SearchInput(name="Nguyễn Văn A")), require_photo=True, require_source=True)).result
    assert all(shown(c) for c in r.candidates)
    assert any("hidden" in w for w in r.warnings)
    # By default people without a photo are kept; a search-found source is still required.
    loose = (await _run(Job(searchId="search_loose", input=SearchInput(name="Nguyễn Văn A")), require_source=True)).result
    assert len(loose.candidates) >= len(r.candidates)
    assert all(any(s.provider != "llm_knowledge" for s in c.sources) for c in loose.candidates)
