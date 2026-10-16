import pytest
from pydantic import ValidationError

from app.core.models import PeopleSearchRequest, SearchInput
from app.search.planner import discovery_queries, enrichment_queries
from app.search.ranking import build_sources
from app.infra.search_adapter import SearchHit
from app.core.client_format import _about_person
from app.core.urls import classify_source, fold, is_profile_url, link_key, name_match, normalize_url


def test_onesearch_filter_names_are_accepted():
    req = PeopleSearchRequest(
        query="Nguyễn Văn A",
        filters={"title": "CTO", "school": "HUST", "location": " Hà  Nội ", "company": "FPT", "industry": "Fintech", "age": "30"},
    )
    assert req.filters == {"jobTitle": "CTO", "education": "HUST", "location": "Hà Nội", "company": "FPT", "field": "Fintech"}


def test_filters_match_related_facts_and_past_roles():
    from app.core.models import CareerItem, FieldValue
    from app.search.resolution import _filter_match

    fields = {
        "industry": FieldValue(value="Fintech", status="found"),
        "company": FieldValue(value="Công ty X", status="found"),
        "education": FieldValue(value="Đại học Bách khoa Hà Nội", status="found"),
    }
    careers = [CareerItem(title="Kỹ sư phần mềm", organization="FPT Software")]
    assert _filter_match(fields, {"field": "fintech", "company": "FPT Software", "jobTitle": "kỹ sư phần mềm"}, careers) == 1.0
    assert _filter_match(fields, {"company": "Viettel", "education": "Bách khoa Hà Nội"}, careers) == 0.5


def test_image_search_query_is_pinned_to_the_profile():
    from app.core.models import ClientSource
    from app.core.urls import image_on_sources, source_image_query

    fb = ClientSource(url="https://www.facebook.com/tramanh.cao.71", title="Cao Thị Trâm Anh", type="facebook", verified=True)
    other = ClientSource(url="https://www.instagram.com/tram.anh.cute/", title="Cao Thị Trâm Anh", type="instagram", verified=True)
    q = source_image_query("Cao Thị Trâm Anh", [fb])
    assert q.startswith("Cao Thị Trâm Anh") and "tramanh.cao.71" in q and "site:facebook.com" in q
    both = source_image_query("Cao Thị Trâm Anh", [fb, other])
    assert "tramanh.cao.71" in both and "tram.anh.cute" in both
    wiki = ClientSource(url="https://en.wikipedia.org/wiki/Hoang_Thuy_Linh", title="Hoàng Thùy Linh", type="wikipedia", verified=True)
    wq = source_image_query("Hoàng Thùy Linh", [wiki, fb])
    assert "site:en.wikipedia.org/wiki/Hoang_Thuy_Linh" in wq
    assert source_image_query("Ai Đó", []) is None
    assert image_on_sources("https://www.facebook.com/tramanh.cao.71/photos/123", "https://scontent.xx.fbcdn.net/a.jpg", [fb])
    assert not image_on_sources("https://www.facebook.com/someone.else/", "https://scontent.xx.fbcdn.net/b.jpg", [fb])
    assert image_on_sources("https://en.wikipedia.org/wiki/Hoang_Thuy_Linh", "https://upload.wikimedia.org/wikipedia/commons/h.jpg", [wiki])


def test_link_key_matches_same_profile_or_page():
    assert link_key("http://www.Instagram.com/leomessi/reels/") == link_key("https://instagram.com/leomessi")
    assert link_key("https://vi.wikipedia.org/wiki/Tr%E1%BA%A7n_Minh_Hi%E1%BA%BFu") == link_key("https://vi.wikipedia.org/wiki/Trần Minh Hiếu")
    assert link_key("https://instagram.com/leomessi") != link_key("https://instagram.com/leomessi_fan")


def test_wikipedia_links_only_for_articles_about_the_person():
    names = ["Trần Minh Hiếu", "HIEUTHUHAI"]
    assert _about_person("https://vi.wikipedia.org/wiki/Hieuthuhai", names)
    assert _about_person("https://en.wikipedia.org/wiki/Ronaldo_(Brazilian_footballer)", ["Ronaldo"])
    assert not _about_person("https://uk.wikipedia.org/wiki/commons:Category:Hieuthuhai", names)
    assert not _about_person("https://vi.wikipedia.org/wiki/2_ng%C3%A0y_1_%C4%91%C3%AAm_(m%C3%B9a_1)", names)


def test_normalize_url_strips_tracking_and_variants():
    assert normalize_url("https://vn.linkedin.com/in/abc/?trk=x&utm_source=y") == "https://linkedin.com/in/abc"
    assert normalize_url("http://www.Example.com/a//b/?q=1&fbclid=z#frag") == "https://example.com/a/b?q=1"
    assert normalize_url("https://twitter.com/abc") == "https://x.com/abc"


def test_profile_and_classify():
    assert is_profile_url("https://linkedin.com/in/abc")
    assert not is_profile_url("https://linkedin.com/posts/abc")
    assert is_profile_url("https://github.com/abc")
    assert classify_source("https://github.com/abc") == "code"
    assert classify_source("https://vnexpress.net/x") == "news"
    assert classify_source("https://daihocz.edu.vn/x") == "academic"


def test_name_match_diacritics():
    assert fold("Nguyễn Văn Đức") == "nguyen van duc"
    assert name_match("Nguyễn Văn A", "ông NGUYEN VAN A là") == 1.0
    assert name_match("Nguyễn Văn A", "Trần Thị B") < 0.4
    assert name_match("Zhang Yuxua", "Zhang Yuxuan - Wikipedia") >= 0.8
    assert name_match("Zhang Yuxua", "Yuxuan Zhang | LinkedIn") >= 0.8
    assert name_match("Nguyễn Minh Khang", "Nguyễn Minh Khoa") < 0.8
    from app.core.urls import name_score

    assert name_score("Đồng Nam Khánh", "Nam Khánh") > name_score("Đồng Nam Khánh", "Bùi Nam Khánh")
    assert name_score("Đồng Nam Khánh", "Nam Khánh") >= 0.7
    assert name_score("Đồng Nam Khánh", "Bùi Nam Khánh") >= 0.5
    assert name_score("Đồng Nam Khánh", "Nguyễn Văn A") < 0.4


def test_name_match_ignores_spacing():
    assert name_match("Chipu", "Chi Pu (@chipupu) • Instagram") == 1.0
    assert name_match("Chi Pu", "Ca sĩ Chipu ra MV mới") == 1.0
    assert name_match("Sơn Tùng MTP", "Sơn Tùng M-TP") == 1.0
    assert name_match("Chipu", "Chipotle menu") < 0.4
    assert name_match("Anh", "An Hà") < 0.4


def test_locale_from_filters():
    from app.infra.search_locale import locale_for

    default = ("us", "en")
    assert locale_for({}, default) == default
    assert locale_for({"location": "San Francisco, USA"}, ("vn", "vi")) == ("us", "en")
    assert locale_for({"location": "Hà Nội"}, default) == ("vn", "vi")
    assert locale_for({"nationality": "Japanese"}, default) == ("jp", "ja")
    assert locale_for({"location": "Mars"}, default) == default
    # No filter: a Vietnamese-spelled name searches Vietnam; accents shared with other languages do not.
    assert locale_for({}, default, "Phùng Thanh Độ") == ("vn", "vi")
    assert locale_for({}, default, "José Mourinho") == default
    assert locale_for({"location": "Tokyo"}, default, "Nguyễn Văn A") == ("jp", "ja")


def test_identity_words_and_more_profile_platforms():
    from app.core.urls import identity_words, mentions_identity, profile_root

    words = identity_words("Professional footballer", "Al-Nassr")
    assert "footballer" in words and "professional" not in words
    assert mentions_identity("Leo Messi, football player for Inter Miami", words)
    assert not mentions_identity("Messi Nguyen, nail salon in Texas", words)
    assert profile_root("https://reddit.com/user/someone") == "https://reddit.com/user/someone"
    assert profile_root("https://pinterest.com/chipupu") == "https://pinterest.com/chipupu"
    assert profile_root("https://pinterest.com/pin/123") is None


def test_handle_must_carry_the_name():
    from app.core.urls import handle_matches_name

    assert handle_matches_name("https://x.com/Cristiano", "Cristiano Ronaldo dos Santos Aveiro", "Ronaldo")
    assert handle_matches_name("https://instagram.com/leomessi", "Lionel Messi", "Messi")
    assert handle_matches_name("https://instagram.com/chipupu", "Chi Pu", "Chipu")
    assert handle_matches_name("https://youtube.com/channel/UClyA28", "Sơn Tùng M-TP")
    assert not handle_matches_name("https://x.com/iamshorgo", "Cristiano Ronaldo dos Santos Aveiro", "Ronaldo")


def test_fan_accounts_are_detected():
    from app.core.urls import looks_like_fan_account

    assert looks_like_fan_account("https://x.com/CRonaldoNews")
    assert looks_like_fan_account("https://facebook.com/MTP.Fan")
    assert looks_like_fan_account("https://instagram.com/abc", "Messi Fans Club")
    assert not looks_like_fan_account("https://x.com/TeamMessi", "Leo Messi (@TeamMessi)")
    assert not looks_like_fan_account("https://youtube.com/user/chipuofficial", "Chi Pu Official")


def test_input_is_name_only():
    assert SearchInput(name="  Nguyễn   Văn A ").name == "Nguyễn Văn A"
    assert set(PeopleSearchRequest.model_fields) == {"query", "filters", "mode", "maxSearchQueries"}
    assert SearchInput(name="A B", filters={"company": " X ", "email": "a@b.c", "location": ""}).filters == {"company": "X"}
    with pytest.raises(ValidationError):
        SearchInput(name=" ")


def test_discovery_uses_only_name_within_budget():
    done: set[str] = set()
    plan = discovery_queries("Nguyễn Văn A", budget=2, done=done)
    assert [p.query for p in plan][0] == '"Nguyễn Văn A"'
    assert len(plan) == 2 and "site:linkedin.com/in" in plan[1].query
    assert all('"Nguyễn Văn A"' in p.query for p in plan)
    with_filters = discovery_queries("Nguyễn Văn A", budget=2, done=set(), filters={"company": "Công ty X"})
    assert with_filters[0].query == '"Nguyễn Văn A" Công ty X'


def test_excerpt_keeps_passages_around_name():
    from app.infra.fetcher import excerpt

    text = "x" * 3000 + " Ông NGUYEN VAN A là CEO. " + "y" * 3000
    out = excerpt(text, "Nguyễn Văn A", limit=500, window=50)
    assert "NGUYEN VAN A là CEO" in out and len(out) <= 500


def test_enrichment_uses_discovered_seeds_and_skips_done():
    done: set[str] = set()
    seeds = {"company": "Công ty X", "jobTitle": "Founder"}
    plan = enrichment_queries("Nguyễn Văn A", seeds, budget=3, done=done, have_platforms={"linkedin"}, include_image=True)
    assert plan[0].query == '"Nguyễn Văn A" "Công ty X"'
    assert not any(p.purpose == "platform:linkedin" for p in plan)
    assert any(p.kind == "image" for p in plan)
    assert enrichment_queries("Nguyễn Văn A", seeds, budget=3, done=done, have_platforms={"linkedin"}, include_image=True) == []
    assert enrichment_queries("Nguyễn Văn A", {}, budget=3, done=set()) == []


def test_build_sources_dedup_filter_and_exclude():
    hits = [
        SearchHit("s", "q1", "Nguyễn Văn A - LinkedIn", "https://vn.linkedin.com/in/a?trk=1", "Founder", 1),
        SearchHit("s", "q2", "Nguyễn Văn A - LinkedIn", "https://linkedin.com/in/a", "Founder at X", 2),
        SearchHit("s", "q1", "Trần Thị B", "https://b.com", "nothing", 3),
        SearchHit("s", "q1", "Nguyễn Văn A", "https://seen.com/a", "x", 4),
        SearchHit("s", "q1", "Nguyễn Văn A nhận giải", "https://vnexpress.net/nguyen-van-a", "Nguyễn Văn A, founder", 5),
    ]
    sources = build_sources(hits, "Nguyễn Văn A", exclude_urls={"https://seen.com/a"}, start_index=5)
    assert [s.url for s in sources] == ["https://linkedin.com/in/a"]
    assert sources[0].id == "src_5"


def test_parse_json_tolerates_raw_newlines_in_strings_and_keys():
    from app.core.llm import _parse_json

    raw = '```json\n{\n  "candidates\n": [{"nameAsWritten": "Tran\nMinh Hieu"}]}\n```'
    assert _parse_json(raw) == {"candidates": [{"nameAsWritten": "Tran\nMinh Hieu"}]}


def test_profile_of_canonical_profiles_only():
    from app.core.urls import profile_of

    p = profile_of("https://m.facebook.com/people/Le-Minh-Son/100012345/")
    assert (p.platform, p.handle, p.url) == ("facebook", "people/le-minh-son/100012345", "https://www.facebook.com/people/Le-Minh-Son/100012345/")
    assert profile_of("https://facebook.com/groups/abc/user/42/").url == "https://www.facebook.com/profile.php?id=42"
    assert profile_of("https://www.facebook.com/profile.php?id=99").handle == "id/99"
    assert profile_of("https://vn.linkedin.com/in/Son-LM?trk=x").url == "https://www.linkedin.com/in/Son-LM/"
    assert profile_of("https://www.instagram.com/leomessi/reels/").url == "https://www.instagram.com/leomessi/"
    assert profile_of("https://www.tiktok.com/@chipu").handle == "chipu"
    assert profile_of("https://www.pinterest.co.uk/someone/").url == "https://www.pinterest.com/someone/"
    assert profile_of("https://www.youtube.com/user/abc/videos").handle == "user/abc"
    assert profile_of("https://youtube.com/legacyname").handle == "legacyname"
    for not_profile in (
        "https://instagram.com/p/Cxyz/",
        "https://facebook.com/espn/videos/999",
        "https://facebook.com/groups/123",
        "https://x.com/user/status/1",
        "https://github.com/user/repo",
        "https://youtube.com/watch?v=1",
        "https://linkedin.com/posts/abc",
    ):
        assert profile_of(not_profile) is None, not_profile


def test_facebook_slug_profiles_count_as_profiles():
    assert is_profile_url("https://facebook.com/people/Le-Minh-Son/100012345")
    assert is_profile_url("https://facebook.com/p/Le-Minh-Son-100012345")
    from app.core.urls import handle_matches_name

    assert handle_matches_name("https://facebook.com/people/Le-Minh-Son/100012345", "Lê Minh Sơn")


def test_official_website_of():
    from app.core.urls import official_website_of

    assert official_website_of("https://www.taylorswift.com/", "Home", "Taylor Swift") == "https://www.taylorswift.com/"
    assert official_website_of("https://www.taylorswift.com/tour", "Tour", "Taylor Swift") is None
    assert official_website_of("https://abc.com/", "Lê Minh Sơn - Official Website", "Lê Minh Sơn") == "https://abc.com/"
    assert official_website_of("https://abc.com/about", "x", "Someone", knowledge_graph=True) == "https://abc.com/"
    assert official_website_of("https://instagram.com/taylorswift", "x", "Taylor Swift") is None


def test_stable_image_url_avoids_social_cdns():
    from app.core.urls import stable_image_url

    assert stable_image_url("https://upload.wikimedia.org/a.jpg", "https://encrypted-tbn0.gstatic.com/t") == "https://upload.wikimedia.org/a.jpg"
    assert stable_image_url("https://scontent.xx.fbcdn.net/a.jpg", "https://encrypted-tbn0.gstatic.com/t") == "https://encrypted-tbn0.gstatic.com/t"
    assert stable_image_url("https://lookaside.fbsbx.com/a", None) is None


def test_name_from_title():
    from app.core.urls import name_from_title

    assert name_from_title("Zhang Yuxuan - Wikipedia", "Zhang Yuxua") == "Zhang Yuxuan"
    assert name_from_title("Lê Minh Sơn - Software Engineer - FPT | LinkedIn", "Le Minh Son") == "Lê Minh Sơn"
    assert name_from_title("Chi Pu (@chipupu) • Instagram photos", "Chi Pu") == "Chi Pu"
    assert name_from_title("Ronaldinho (@ronaldinho) • Instagram", "Cristiano Ronaldo") is None


def test_snippet_facts_and_preview_grouping():
    from app.search.preview import Mention, build_preview, extract_snippet_facts

    li = Mention(
        "Lê Minh Sơn - Software Engineer - FPT Software | LinkedIn",
        "https://vn.linkedin.com/in/sonlm",
        "Experience: FPT Software · Education: Đại học Bách khoa Hà Nội · Location: Hà Nội · 500+ connections",
    )
    assert extract_snippet_facts(li, "linkedin", "Lê Minh Sơn") == {
        "role": "Software Engineer",
        "company": "FPT Software",
        "education": "Đại học Bách khoa Hà Nội",
        "location": "Hà Nội",
    }
    fb = Mention("Lê Minh Sơn | Facebook", "https://facebook.com/son.lm", "Làm việc tại FPT Software. Sống tại Hà Nội. 1,2K người theo dõi")
    assert extract_snippet_facts(fb, "facebook", "Lê Minh Sơn")["audience"] == "1,2K followers"
    other = Mention("Lê Minh Sơn | Facebook", "https://facebook.com/leminhson.hcm", "Sống tại Hồ Chí Minh")
    web = Mention("Lê Minh Sơn - Kỹ sư FPT Software", "https://vnexpress.net/son", "Kỹ sư Lê Minh Sơn tại FPT Software chia sẻ...")
    cands, total = build_preview("Lê Minh Sơn", [li, fb, other], [web])
    assert total == 4
    # LinkedIn + the Facebook profile sharing the company -> one person; the other namesake stays separate.
    assert len(cands) == 2
    first = cands[0]
    assert first.platforms == ["facebook", "linkedin"]
    assert [s.type for s in first.sources] == ["facebook", "linkedin", "web"]
    assert {f.key for f in first.keyFacts} >= {"role", "company", "location", "education"}
    assert cands[1].platforms == ["facebook"] and len(cands[1].sources) == 1


def test_slug_id_is_stable_and_salted():
    from app.core.client_format import slug_id

    assert slug_id("Lê Minh Sơn", "a") == slug_id("Lê Minh Sơn", "a")
    assert slug_id("Lê Minh Sơn", "a") != slug_id("Lê Minh Sơn", "b")
    assert slug_id("Đặng Văn Lâm").startswith("dang-van-lam-")


def test_select_photos_keeps_only_the_persons_face():
    import numpy as np

    from app.media.faces import Face, select_photos

    def face(*v):
        e = np.array(v, dtype=float)
        return Face(box=(0, 0, 100, 100), score=0.9, embedding=e / np.linalg.norm(e))

    me, me2, other = face(1, 0.1, 0), face(0.95, 0.2, 0.05), face(0, 1, 0)
    # Trusted avatar is the reference; no-face, unreadable and other-person photos are dropped.
    found = [[me], [], None, [other], [me2], [other, me2]]
    kept = select_photos(found, [True, False, False, False, False, False], 0.363)
    assert [i for i, _ in kept] == [0, 4, 5] and kept[0][1] == 1.0
    # A trusted group photo cannot be the reference: no trusted single face -> consensus.
    found = [[me, other], [other], [me2], [face(0.9, 0, 0.1)]]
    assert [i for i, _ in select_photos(found, [True, False, False, False], 0.363)] == [0, 2, 3]
    # Nothing agrees and nothing is trusted: no photo can be confirmed.
    assert select_photos([[me], [other]], [False, False], 0.363) == []


def test_article_pages_are_not_sources():
    from app.core.urls import is_news

    for u in (
        "https://viettimes.vn/fpt-sap-phat-hanh-co-phieu-chu-tich-truong-gia-binh-post205626.html",
        "https://chungta.vn/cong-nghe/anh-truong-gia-binh-1141585.html",
        "https://vannguyen.edu.vn/6-bai-hoc-cua-chu-tich-fpt-truong-gia-binh",
        "https://example.com/2024/05/12/some-post",
        "https://company.com/blog/meet-our-founder",
    ):
        assert is_news(u), u
    for u in (
        "https://mn2s.com/booking-agency/talent-roster/chi-pu",
        "https://www.imdb.com/name/nm6679580",
        "https://en.wikipedia.org/wiki/Chi_Pu",
        "https://nguyenvana.dev/about",
        "https://fpt.com/vi/ve-fpt/ban-lanh-dao/truong-gia-binh",
        "https://www.linkedin.com/in/a",
        "https://daihocz.edu.vn/giang-vien/nguyen-van-a",
    ):
        assert not is_news(u), u


def test_same_face_groups():
    import numpy as np

    from app.media.faces import Face, same_face_groups

    def face(v):
        v = np.array(v, dtype=np.float32)
        return Face((0, 0, 1, 1), 1.0, v / np.linalg.norm(v))

    refs = [face([1, 0, 0]), face([0, 1, 0]), face([0.95, 0.1, 0]), None, face([0, 0.9, 0.2])]
    assert same_face_groups(refs, 0.55) == [[0, 2], [1, 4]]
    assert same_face_groups(refs, 0.999) == []


def test_more_photos_keeps_matching_faces_only():
    import asyncio

    import numpy as np

    from app.config import Settings
    from app.media.faces import Face
    from app.core.models import MorePhotosRequest
    from app.media.photos import PhotoService
    from app.infra.search_adapter import SearchHit

    def face(v):
        v = np.array(v, dtype=np.float32)
        return Face((0, 0, 10, 10), 1.0, v / np.linalg.norm(v))

    me, other = face([1, 0, 0]), face([0, 1, 0])
    # copy.jpg is the reference picture under another URL.
    faces_by_url = {
        "https://x/ref.jpg": [me],
        "https://x/copy.jpg": [me],
        "https://x/a.jpg": [face([1, 0.5, 0])],
        "https://x/b.jpg": [other],
        "https://x/c.jpg": [other, face([0.9, 0.3, 0])],
        "https://x/d.jpg": None,
    }

    class Faces:
        async def ready(self):
            return True

        async def faces_in(self, urls):
            return [faces_by_url.get(u) for u in urls]

        async def faces_or_fallback(self, urls, fallbacks):
            return await self.faces_in(urls), list(urls)

    class Search:
        async def search(self, q, kind, logs, locale=None):
            assert kind == "image"
            names = ["ref", "copy", "a", "b", "c", "d", "known"]
            return [SearchHit(provider="f", query=q, title=f"Nguyen Van A {n}", url=f"https://p/{n}", snippet="", position=i, image_url=f"https://x/{n}.jpg") for i, n in enumerate(names)], []

    svc = PhotoService(Settings(_env_file=None, llm_api_key=""), Search(), Faces())
    req = MorePhotosRequest(query="Nguyen Van A", referenceUrl="https://x/ref.jpg", name="Nguyen Van A", exclude=["https://x/known.jpg"])
    out = asyncio.run(svc.more(req))
    # The reference picture under another URL is not offered again.
    assert [p.url for p in out.photos] == ["https://x/c.jpg", "https://x/a.jpg"]
    assert out.meta.faceChecked and out.photos[1].faceMatch == 0.894


def test_more_photos_by_alias_without_reference():
    import asyncio

    import numpy as np

    from app.config import Settings
    from app.media.faces import Face
    from app.core.models import MorePhotosRequest
    from app.media.photos import PhotoService
    from app.infra.search_adapter import SearchHit

    def face(v):
        v = np.array(v, dtype=np.float32)
        return Face((0, 0, 10, 10), 1.0, v / np.linalg.norm(v))

    faces_by_url = {"https://x/1.jpg": [face([1, 0, 0])], "https://x/2.jpg": [face([0, 1, 0])], "https://x/3.jpg": None}

    class Faces:
        async def ready(self):
            return True

        async def faces_in(self, urls):
            return [faces_by_url.get(u) for u in urls]

        async def faces_or_fallback(self, urls, fallbacks):
            return await self.faces_in(urls), list(urls)

    class Search:
        async def search(self, q, kind, logs, locale=None):
            titles = ["Low G - Rap Việt", "Rapper Low G live", "Low G album cover", "Another singer"]
            return [SearchHit(provider="f", query=q, title=t, url=f"https://p/{i}", snippet="", position=i, image_url=f"https://x/{i}.jpg") for i, t in enumerate(titles, 1)], []

    svc = PhotoService(Settings(_env_file=None, llm_api_key=""), Search(), Faces())
    out = asyncio.run(svc.more(MorePhotosRequest(query="Nguyen Hoang Long Low G rapper", name="Nguyễn Hoàng Long", aliases=["Low G"])))
    # Titles carry only the stage name; no two faces agree, so single-face images come back unverified.
    assert [p.url for p in out.photos] == ["https://x/1.jpg", "https://x/2.jpg"]
    assert not out.meta.faceChecked and all(p.faceMatch is None for p in out.photos)


def test_deep_search_keeps_avatar_when_no_face_photos():
    from app.search.deep_search import gallery_and_avatar
    from app.core.models import ClientPhoto

    ava = ClientPhoto(url="https://scontent.xx.fbcdn.net/v/ava.jpg", fromSource=True)
    face = ClientPhoto(url="https://scontent.xx.fbcdn.net/v/face.jpg", fromSource=True, faceMatch=0.9)
    photos, avatar = gallery_and_avatar([face, ava], [ava, face], None)
    assert [p.url for p in photos] == [face.url, ava.url] and avatar == face.url
    photos, avatar = gallery_and_avatar([], [ava], "https://old/avatar.jpg")
    assert photos == [] and avatar == ava.url
    photos, avatar = gallery_and_avatar([], [], "https://old/avatar.jpg")
    assert photos == [] and avatar == "https://old/avatar.jpg"


def test_facebook_photo_permalink_is_listed_as_a_website():
    from app.search.deep_search import classify_links
    from app.infra.social_crawl import photos_from_profile
    from app.core.urls import looks_like_image_url

    photo = "https://www.facebook.com/photo.php?fbid=123&set=a.456&id=789"
    slash = "https://www.facebook.com/photo/?fbid=123"
    profile = "https://www.facebook.com/profile.php?id=1000"

    assert not looks_like_image_url(photo) and not looks_like_image_url(slash)
    profiles, articles = classify_links(
        [photo, slash, "https://www.facebook.com/leminhson", "https://tuoitre.vn/le-minh-son.htm"],
        "Lê Minh Sơn",
        profile,
    )
    assert [s.type for s in profiles] == ["facebook"]
    assert any("facebook.com/photo" in s.url for s in articles)
    assert articles[-1].url.startswith("https://tuoitre.vn/")

    pics = photos_from_profile(
        {
            "display_name": "Lê Minh Sơn",
            "avatar_url": "https://scontent.xx.fbcdn.net/v/t1.0-1/ava.jpg",
            "links": [photo],
        },
        profile,
        "Lê Minh Sơn",
    )
    assert [p.url for p in pics] == ["https://scontent.xx.fbcdn.net/v/t1.0-1/ava.jpg"]


def test_image_key_keeps_identifying_query_params():
    from app.core.urls import image_key

    g = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpZ4XT&s=10"
    assert image_key(g) != image_key("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5LGgW&s=10")
    assert image_key(g) == image_key("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRpZ4XT&s")
    fb = "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=1"
    assert image_key(fb) != image_key("https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=2")
    news = "https://media-cdn-v2.laodong.vn/storage/a/Pham-Hue1.jpeg"
    assert image_key(news + "?w=800&h=420&crop=auto&scale=both") == image_key(news)
    wiki = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/X.jpg/330px-X.jpg"
    assert image_key(wiki + "?utm_source=vi") == image_key("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/X.jpg/1280px-X.jpg")


def test_more_photos_from_own_profile_without_name_in_caption():
    import asyncio

    import numpy as np

    from app.config import Settings
    from app.media.faces import Face
    from app.core.models import MorePhotosRequest
    from app.media.photos import PhotoService
    from app.infra.search_adapter import SearchHit

    def face(v):
        v = np.array(v, dtype=np.float32)
        return Face((0, 0, 10, 10), 1.0, v / np.linalg.norm(v))

    me, other = face([1, 0, 0]), face([0, 1, 0])
    thumbs = [f"https://encrypted-tbn0.gstatic.com/images?q=tbn:{i}&s=10" for i in range(4)]
    faces_by_url = {thumbs[0]: [me], thumbs[1]: [face([1, 0.5, 0])], thumbs[2]: [other], thumbs[3]: [face([1, 0, 0.5])]}

    class Faces:
        async def ready(self):
            return True

        async def faces_in(self, urls):
            return [faces_by_url.get(u) for u in urls]

        async def faces_or_fallback(self, urls, fallbacks):
            return await self.faces_in(urls), list(urls)

    class Search:
        async def search(self, q, kind, logs, locale=None):
            rows = [
                ("Thêm một câu chuyện tình yêu cực kỳ dễ thương", "https://www.facebook.com/longines.daika/"),
                ("Nguyễn Hoàng Long (@longines.daika) • Facebook", "https://www.facebook.com/longines.daika/"),
                ("Simple Crowd Favorite Game", "https://www.tiktok.com/@jolley.hoang.nguy/video/1"),
                ("Muốn đập chai thì phải đập hộp | Nguyễn Hoàng Long | Facebook", "https://www.facebook.com/someone.else/"),
            ]
            return [SearchHit(provider="f", query=q, title=t, url=u, snippet="", position=i, image_url=thumbs[i]) for i, (t, u) in enumerate(rows)], []

    svc = PhotoService(Settings(_env_file=None, llm_api_key=""), Search(), Faces())
    req = MorePhotosRequest(query="Nguyen Hoang Long bilingual wedding mc", name="Nguyễn Hoàng Long", profileUrls=["https://facebook.com/longines.daika"])
    out = asyncio.run(svc.more(req))
    # Own-page images are kept without the name in the caption; Google thumbnails are not merged
    # into one picture; another Facebook account with the same name is dropped.
    assert [p.url for p in out.photos] == [thumbs[0], thumbs[1]]
    assert out.photos[0].fromSource and out.photos[0].faceMatch == 1.0 and out.meta.faceChecked


def test_image_fallback_to_google_thumbnail():
    import asyncio

    from app.media.faces import FaceVerifier
    from app.core.urls import stable_image_url

    thumb = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9Gc&s=10"
    # TikTok / Instagram / Facebook pages answer non-browsers with HTML: use the thumbnail.
    for blocked in ("https://www.tiktok.com/api/img/?itemId=1", "https://lookaside.instagram.com/seo/x?media_id=1", "https://lookaside.fbsbx.com/x?media_id=1"):
        assert stable_image_url(blocked, thumb) == thumb
    assert stable_image_url("https://cosari.vn/a.png", thumb) == "https://cosari.vn/a.png"

    fv = FaceVerifier("models", "test")
    readable = {thumb: [], "https://ok/b.jpg": []}

    async def fake_faces_in(urls, concurrency=8):
        return [readable.get(u) for u in urls]

    fv.faces_in = fake_faces_in  # type: ignore[method-assign]
    found, used = asyncio.run(fv.faces_or_fallback(["https://broken/a.jpg", "https://ok/b.jpg", "https://broken/c.jpg"], [thumb, thumb, None]))
    assert used == [thumb, "https://ok/b.jpg", "https://broken/c.jpg"] and found == [[], [], None]


def test_search_profile_queries_merge_and_require_a_profile_page():
    from app.core.llm import MergedPerson
    from app.search.search_profile import ProfileHit, assemble_candidates, is_profile_page, more_profile_queries, profile_queries

    queries = profile_queries("Chi Dân", {"location": "Hồ Chí Minh", "occupation": "ca sĩ", "company": "", "field": "nhạc", "jobTitle": "singer"})
    assert len(queries) == 2
    assert all(q.startswith("Chi Dân") and "site:" in q for q in queries)
    near = profile_queries("Đồng Nam Khánh", {})
    assert all(q.startswith("Đồng Nam Khánh") for q in near)
    assert "site:en.wikipedia.org" in queries[0] and "site:wikipedia.org" in queries[0]
    assert all("wikipedia" not in q for q in queries[1:])
    assert "Hồ Chí Minh" in queries[0] and "Hồ Chí Minh" in queries[1]
    assert "ca sĩ" not in queries[0] and "nhạc" not in queries[0]
    assert "site:linkedin.com/in" in queries[0] and "site:facebook.com" in queries[0]
    assert "site:instagram.com" in queries[1] and "site:tiktok.com" in queries[1] and "site:x.com" in queries[1]
    assert "ca sĩ" in queries[1] and "nhạc" in queries[1]
    assert "company" not in queries[0]
    more = more_profile_queries("Chi Dân", {"location": "Hồ Chí Minh", "occupation": "ca sĩ", "company": "", "field": "nhạc", "jobTitle": "singer"})
    assert more == [(queries[0], 2), (queries[1], 2)]
    assert more_profile_queries("Chi Dân", {}, 3) == [(q, 3) for q in profile_queries("Chi Dân", {})]

    from app.search.search_profile import wiki_hit, wiki_page_about

    assert is_profile_page("https://www.instagram.com/nguyentrunghieuuu/", "Chi Dân (@nguyentrunghieuuu)", "Chi Dân")
    assert is_profile_page("https://en.wikipedia.org/wiki/Chi_D%C3%A2n", "Chi Dân - Wikipedia", "Chi Dân")
    assert is_profile_page("https://en.wikipedia.org/wiki/Jennie_(singer)", "Jennie (singer) - Wikipedia", "Kim Jennie")
    assert not is_profile_page("https://en.wikipedia.org/wiki/File:Jennie.jpg", "File:Jennie.jpg", "Kim Jennie")
    assert not is_profile_page("https://vnexpress.net/chi-dan-hat-moi-1234567.html", "Chi Dân ra mắt bài mới", "Chi Dân")
    assert not is_profile_page("https://www.facebook.com/chi.dan/posts/123", "Chi Dân posted a photo", "Chi Dân")
    assert not is_profile_page("https://www.youtube.com/watch?v=abc", "Chi Dân - Điều Anh Biết", "Chi Dân")
    # Stage-name Wikipedia: keep for merge / scrape, confirm from article body.
    mixi = "https://vi.wikipedia.org/wiki/%C4%90%E1%BB%99_Mixi"
    assert wiki_hit(mixi, "Độ Mixi - Wikipedia", "Phùng Thanh Độ", "Streamer MixiGaming")
    assert is_profile_page(mixi, "Độ Mixi - Wikipedia", "Phùng Thanh Độ", aliases=["Độ Mixi"])
    assert is_profile_page(mixi, "Độ Mixi - Wikipedia", "Phùng Thanh Độ")
    assert wiki_page_about(
        "Độ Mixi",
        "Phùng Thanh Độ (sinh ngày 12 tháng 9 năm 1989), thường được biết đến với nghệ danh Độ Mixi, là một streamer.",
        "Phùng Thanh Độ",
    )
    assert not wiki_page_about("Độ Mixi", "", "Phùng Thanh Độ")
    assert not wiki_page_about("Lionel Messi", "Lionel Messi is an Argentine footballer.", "Phùng Thanh Độ")

    li = ProfileHit("src_1", "Lê Minh Sơn - Engineer - FPT | LinkedIn", "https://www.linkedin.com/in/sonlm", "Experience: FPT Software · Location: Hà Nội")
    fb = ProfileHit("src_2", "Lê Minh Sơn | Facebook", "https://www.facebook.com/son.lm", "Làm việc tại FPT Software")
    wiki = ProfileHit("src_6", "Lê Minh Sơn - Wikipedia", "https://vi.wikipedia.org/wiki/Lê_Minh_Sơn", "Kỹ sư phần mềm")
    news = ProfileHit("src_3", "Lê Minh Sơn trên báo", "https://vnexpress.net/le-minh-son-123456.html", "bài phỏng vấn")
    other = ProfileHit("src_4", "Trần Thị B - Designer | LinkedIn", "https://www.linkedin.com/in/tranthib", "Designer tại VNG")
    people = [
        MergedPerson(displayName="Lê Minh Sơn", headline="Engineer · FPT", confidence=0.9, sourceIds=["src_1", "src_2", "src_6", "src_99"], imageSearchQuery="Le Minh Son FPT"),
        MergedPerson(displayName="Lê Minh Sơn", confidence=0.4, sourceIds=["src_3"]),
        MergedPerson(displayName="Trần Thị B", confidence=0.8, sourceIds=["src_4"]),
    ]
    cands = assemble_candidates("Lê Minh Sơn", [li, fb, wiki, news, other], people, 10)
    assert len(cands) == 1
    assert cands[0].displayName == "Lê Minh Sơn"
    assert [s.type for s in cands[0].sources] == ["wikipedia", "facebook", "linkedin"]
    vi = ProfileHit("src_7", "Chi Dân - Wikipedia", "https://vi.wikipedia.org/wiki/Chi_Dân")
    en = ProfileHit("src_8", "Chi Dân - Wikipedia", "https://en.wikipedia.org/wiki/Chi_Dan")
    fb2 = ProfileHit("src_9", "Chi Dân | Facebook", "https://www.facebook.com/chidan")
    both = assemble_candidates(
        "Chi Dân",
        [vi, en, fb2],
        [MergedPerson(displayName="Chi Dân", confidence=0.9, sourceIds=["src_7", "src_8", "src_9"])],
        5,
    )
    assert [s.url for s in both[0].sources] == [en.url, fb2.url]
    assert cands[0].location == "Hà Nội"
    assert "src_99" not in {s.url for s in cands[0].sources}
    assert cands[0].avatarUrl is None

    pictured = ProfileHit(
        "src_5",
        "Lê Minh Sơn | Facebook",
        "https://www.facebook.com/son.lm.photo",
        image="https://img.example/son.jpg",
    )
    with_photo = assemble_candidates(
        "Lê Minh Sơn",
        [pictured],
        [MergedPerson(displayName="Lê Minh Sơn", confidence=0.8, sourceIds=["src_5"])],
        5,
    )
    assert with_photo[0].avatarUrl == "https://img.example/son.jpg"
    assert with_photo[0].photos[0].fromSource is True

    li_ava = ProfileHit(
        "src_li",
        "Haoran Chen | LinkedIn",
        "https://www.linkedin.com/in/haoranchen-uva/",
        image="https://media.licdn.com/dms/image/v2/C4E03AQ/profile-displayphoto-shrink_200_200/0/haoran.jpg",
        image_fallback="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTest",
    )
    from_li = assemble_candidates(
        "Haoran Chen",
        [li_ava],
        [MergedPerson(displayName="Haoran Chen", confidence=0.8, sourceIds=["src_li"])],
        5,
    )
    assert from_li[0].avatarUrl.endswith("haoran.jpg")
    assert from_li[0].photos[0].fromSource is True
    assert from_li[0].photos[0].fallbackUrl.startswith("https://encrypted-tbn0.gstatic.com/")

    nam = ProfileHit("src_n", "Nam Khánh | Facebook", "https://www.facebook.com/namkhanh")
    bui = ProfileHit("src_b", "Bùi Nam Khánh | Instagram", "https://www.instagram.com/buinamkhanh/")
    ranked = assemble_candidates(
        "Đồng Nam Khánh",
        [nam, bui],
        [
            MergedPerson(displayName="Bùi Nam Khánh", confidence=0.9, sourceIds=["src_b"]),
            MergedPerson(displayName="Nam Khánh", confidence=0.7, sourceIds=["src_n"]),
        ],
        5,
    )
    assert [c.displayName for c in ranked] == ["Bùi Nam Khánh", "Nam Khánh"]
    assert [c.confidence for c in ranked] == [0.9, 0.7]

    # Legal name on the card, stage name on Wikipedia — still attach the article.
    mixi_wiki = ProfileHit("src_mixi", "Độ Mixi - Wikipedia", "https://vi.wikipedia.org/wiki/%C4%90%E1%BB%99_Mixi", "Streamer người Việt")
    mixi = assemble_candidates(
        "Phùng Thanh Độ",
        [mixi_wiki],
        [MergedPerson(displayName="Phùng Thanh Độ", aliases=["Độ Mixi"], confidence=0.9, sourceIds=["src_mixi"])],
        5,
    )
    assert len(mixi) == 1
    assert [s.type for s in mixi[0].sources] == ["wikipedia"]
    assert "Độ Mixi" in mixi[0].aliases


def test_search_profile_merge_gets_already_listed_names():
    import asyncio

    from app.config import Settings
    from app.core.llm import MergedPerson, MergeProfilesOutput
    from app.infra.search_adapter import SearchHit
    from app.search.search_profile import SearchProfile

    seen: dict[str, list[str] | None] = {}

    class Search:
        async def search(self, query, kind, logs, locale, page=1):
            return [SearchHit("s", query, "A | Facebook", "https://www.facebook.com/a.person", "bio", 1)], []

    class LLM:
        async def merge_profiles(self, query, filters, sources, already_listed=None):
            seen["already_listed"] = already_listed
            return MergeProfilesOutput(candidates=[MergedPerson(displayName="A", confidence=0.9, sourceIds=["src_1"])])

    svc = SearchProfile(Settings(_env_file=None, llm_api_key="x"), Search(), LLM(), None)
    asyncio.run(svc.run("A", {}, exclude_names=["Người đã có"]))
    assert seen["already_listed"] == ["Người đã có"]


def test_wikipedia_query_and_wikidata_fill_the_profile():
    import asyncio

    from app.config import Settings
    from app.core.models import ClientCandidate, ClientSource
    from app.search.search_profile import SearchProfile
    from app.infra.wiki import WikiPage, facts_from_entity, images_from_generator, parse_query, social_from_entity

    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Jennie (singer)",
                    "extract": "Jennie Kim is a South Korean singer and rapper.\n\nShe is a member of the girl group Blackpink.",
                    "original": {"source": "https://upload.wikimedia.org/jennie.jpg"},
                    "pageprops": {"wikibase_item": "Q210835"},
                    "langlinks": [{"url": "https://en.wikipedia.org/wiki/Jennie_(singer)"}],
                    "extlinks": [
                        {"*": "https://www.instagram.com/jennierubyjane/"},
                        {"*": "https://www.youtube.com/channel/UCNYiF2xVdTTbYuan2lX7How"},
                        {"*": "https://www.youtube.com/watch?v=abc123"},
                    ],
                }
            }
        }
    }
    page = parse_query(payload, "https://en.wikipedia.org/wiki/Jennie_(singer)")
    assert page.qid == "Q210835" and page.image.endswith("jennie.jpg")
    gallery = images_from_generator(
        {
            "query": {
                "pages": {
                    "10": {"title": "File:Fernanda_Montenegro2019.jpg", "imageinfo": [{"url": "https://upload.wikimedia.org/wikipedia/commons/2/2a/Fernanda_Montenegro2019.jpg", "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/f1.jpg", "mime": "image/jpeg"}]},
                    "11": {"title": "File:Flag_of_Brazil.svg", "imageinfo": [{"url": "https://upload.wikimedia.org/flag.svg", "mime": "image/svg+xml"}]},
                    "12": {"title": "File:Fernanda_Montenegro_1999.jpg", "imageinfo": [{"url": "https://upload.wikimedia.org/wikipedia/commons/f/f9/Fernanda_1999.jpg", "mime": "image/jpeg"}]},
                }
            }
        }
    )
    assert gallery == [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/f/f9/Fernanda_1999.jpg",
    ]
    assert "Blackpink" in page.extract
    assert "https://www.instagram.com/jennierubyjane/" in page.links
    assert "https://www.youtube.com/channel/UCNYiF2xVdTTbYuan2lX7How" in page.links
    assert "https://www.youtube.com/watch?v=abc123" in page.links

    entity = {
        "claims": {
            "P569": [{"rank": "preferred", "mainsnak": {"snaktype": "value", "datavalue": {"type": "time", "value": {"time": "+1996-01-16T00:00:00Z", "precision": 11}}}}],
            "P19": [{"rank": "preferred", "mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q8684"}}}}],
            "P27": [{"rank": "preferred", "mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q884"}}}}],
            "P106": [{"rank": "preferred", "mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q177220"}}}}],
            "P2003": [{"rank": "preferred", "mainsnak": {"snaktype": "value", "datavalue": {"type": "string", "value": "jennierubyjane"}}}],
            "P856": [{"rank": "preferred", "mainsnak": {"snaktype": "value", "datavalue": {"type": "string", "value": "https://www.jenn.ie"}}}],
        }
    }
    facts = facts_from_entity(entity, {"Q8684": "Seoul", "Q884": "South Korea", "Q177220": "singer"})
    assert facts["born"] == "16 January 1996, Seoul"
    assert facts["nationality"] == "South Korea" and facts["occupation"] == "singer"
    assert facts["website"] == "https://www.jenn.ie"
    assert "https://www.instagram.com/jennierubyjane/" in social_from_entity(entity)

    class Fetch:
        async def wikipedia_page(self, url):
            return WikiPage(
                url=url,
                title="Jennie (singer)",
                extract="Jennie Kim is a South Korean singer and rapper.\n\nShe is a member of Blackpink.",
                image="https://upload.wikimedia.org/jennie.jpg",
                links=["https://www.instagram.com/jennierubyjane/", "https://www.youtube.com/channel/UCNYiF2xVdTTbYuan2lX7How", "https://www.youtube.com/watch?v=abc123"],
                facts={"born": "16 January 1996, Seoul", "nationality": "South Korea", "occupation": "singer, rapper"},
                english_url="https://en.wikipedia.org/wiki/Jennie_(singer)",
            )

    person = ClientCandidate(
        id="jennie",
        displayName="Jennie Kim",
        confidence=0.9,
        imageSearchQuery="Jennie Kim",
        sources=[ClientSource(url="https://ko.wikipedia.org/wiki/제니", title="제니", type="wikipedia", verified=True)],
    )
    svc = SearchProfile(Settings(_env_file=None, llm_api_key=""), None, None, None, Fetch())  # type: ignore[arg-type]
    asyncio.run(svc._fill_wiki([person]))
    assert person.profileSource == "wikipedia" and any("Blackpink" in p for p in person.profile)
    assert person.occupation == "singer, rapper"
    by_key = {f.key: f.value for f in person.keyFacts}
    assert by_key["born"] == "16 January 1996, Seoul" and by_key["nationality"] == "South Korea"
    assert person.photos[0].fromSource and person.photos[0].url.endswith("jennie.jpg")
    assert any(s.type == "instagram" for s in person.sources)
    assert any(s.type == "youtube" and "/channel/" in s.url for s in person.sources)
    assert not any("watch?v=" in s.url for s in person.sources)
    wiki_urls = [s.url for s in person.sources if s.type == "wikipedia"]
    assert wiki_urls == ["https://en.wikipedia.org/wiki/Jennie_(singer)"]

    class MixiFetch:
        async def wikipedia_page(self, url):
            return WikiPage(
                url=url,
                title="Độ Mixi",
                extract="Phùng Thanh Độ (sinh ngày 12 tháng 9 năm 1989), thường được biết đến với nghệ danh Độ Mixi, là một streamer người Việt Nam.",
                image="https://upload.wikimedia.org/mixi.jpg",
                links=["https://www.facebook.com/mixigaming"],
                facts={"occupation": "streamer", "nationality": "Vietnam"},
                english_url=None,
            )

    mixi = ClientCandidate(
        id="mixi",
        displayName="Phùng Thanh Độ",
        aliases=["Độ Mixi"],
        confidence=0.9,
        imageSearchQuery="Phung Thanh Do Mixi",
        sources=[ClientSource(url="https://vi.wikipedia.org/wiki/%C4%90%E1%BB%99_Mixi", title="Độ Mixi", type="wikipedia", verified=True)],
    )
    asyncio.run(SearchProfile(Settings(_env_file=None, llm_api_key=""), None, None, None, MixiFetch())._fill_wiki([mixi]))  # type: ignore[arg-type]
    assert mixi.profileSource == "wikipedia"
    assert any(s.type == "wikipedia" for s in mixi.sources)
    assert any(s.type == "facebook" for s in mixi.sources)

    class WrongWiki:
        async def wikipedia_page(self, url):
            return WikiPage(url=url, title="Lionel Messi", extract="Lionel Messi is an Argentine professional footballer.", facts={})

    wrong = ClientCandidate(
        id="wrong",
        displayName="Phùng Thanh Độ",
        confidence=0.5,
        imageSearchQuery="Phung Thanh Do",
        sources=[
            ClientSource(url="https://en.wikipedia.org/wiki/Lionel_Messi", title="Lionel Messi", type="wikipedia", verified=True),
            ClientSource(url="https://www.facebook.com/mixigaming", title="MixiGaming", type="facebook", verified=True),
        ],
    )
    asyncio.run(SearchProfile(Settings(_env_file=None, llm_api_key=""), None, None, None, WrongWiki())._fill_wiki([wrong]))  # type: ignore[arg-type]
    assert not any(s.type == "wikipedia" for s in wrong.sources)
    assert any(s.type == "facebook" for s in wrong.sources)


def test_search_profile_leaves_a_photo_empty_when_the_source_has_none():
    import asyncio

    from app.config import Settings
    from app.core.models import ClientCandidate
    from app.search.search_profile import SearchProfile

    class Photos:
        async def find(self, name, sources, limit=None, **kwargs):
            return [], False

    svc = SearchProfile(Settings(_env_file=None, llm_api_key=""), None, None, Photos())  # type: ignore[arg-type]
    person = ClientCandidate(
        id="giang",
        displayName="Nguyen Truong Giang",
        confidence=0.85,
        headline="Internship Student at German Aerospace Center (DLR)",
        imageSearchQuery="Nguyen Truong Giang DLR",
        sources=[],
    )
    asyncio.run(svc._attach_photos([person]))
    assert person.avatarUrl is None
    assert person.photos == []


def test_search_profile_keeps_source_photos_without_an_image_search():
    import asyncio

    from app.config import Settings
    from app.core.models import ClientCandidate, ClientPhoto, ClientSource
    from app.search.search_profile import SearchProfile

    class Search:
        def __init__(self):
            self.queries: list[str] = []

        async def search(self, query, kind, logs, locale):
            self.queries.append(query)
            raise AssertionError("search-profile must not run an image search")

    class Photos:
        async def find(self, name, sources, limit=None, **kwargs):
            return [], False

    search = Search()
    svc = SearchProfile(Settings(_env_file=None, llm_api_key=""), search, None, Photos())  # type: ignore[arg-type]
    people = [
        ClientCandidate(
            id=str(i),
            displayName=f"Person {i}",
            confidence=0.8,
            imageSearchQuery=f"person {i}",
            sources=[ClientSource(url=f"https://www.facebook.com/person{i}", title=f"Person {i}", type="facebook", verified=True)] if i < 3 else [],
        )
        for i in range(4)
    ]
    people[1].photos = [ClientPhoto(url="https://img.example/have.jpg", fromSource=True)]
    people[1].avatarUrl = people[1].photos[0].url
    asyncio.run(svc._attach_photos(people))
    assert people[0].photos == [] and people[0].avatarUrl is None
    assert people[1].photos[0].url == "https://img.example/have.jpg"
    assert people[2].photos == []
    assert people[3].avatarUrl is None
    assert search.queries == []


def test_search_profile_keeps_source_photos_including_a_logo():
    import asyncio

    from app.config import Settings
    from app.core.models import ClientCandidate, ClientPhoto, ClientSource
    from app.search.search_profile import SearchProfile

    class Photos:
        async def find(self, name, sources, limit=None, **kwargs):
            return [], False

    svc = SearchProfile(Settings(_env_file=None, llm_api_key=""), None, None, Photos())  # type: ignore[arg-type]
    person = ClientCandidate(
        id="cha",
        displayName="Cha Eun Woo",
        confidence=0.9,
        imageSearchQuery="Cha Eun Woo",
        sources=[ClientSource(url="https://en.wikipedia.org/wiki/Cha_Eun-woo", title="Cha Eun-woo", type="wikipedia", verified=True)],
        photos=[
            ClientPhoto(url="https://img.example/logo.png", fromSource=True),
            ClientPhoto(url="https://img.example/me.jpg", fromSource=True, faceMatch=1.0),
            ClientPhoto(url="https://img.example/other.jpg", fromSource=True, faceMatch=0.37),
        ],
    )
    asyncio.run(svc._attach_photos([person]))
    assert [p.url for p in person.photos] == [
        "https://img.example/logo.png",
        "https://img.example/me.jpg",
        "https://img.example/other.jpg",
    ]
    assert person.avatarUrl == "https://img.example/logo.png"


def test_search_profile_keeps_duplicate_source_photos_on_the_card():
    import asyncio

    from app.config import Settings
    from app.core.models import ClientCandidate, ClientPhoto, ClientSource
    from app.search.search_profile import SearchProfile

    class Photos:
        async def find(self, name, sources, limit=None, **kwargs):
            return [], False

    person = ClientCandidate(
        id="yang",
        displayName="Yang Mi",
        confidence=0.9,
        imageSearchQuery="Yang Mi",
        sources=[ClientSource(url="https://en.wikipedia.org/wiki/Yang_Mi", title="Yang Mi", type="wikipedia", verified=True)],
        photos=[
            ClientPhoto(url="https://img.example/wiki.jpg", fromSource=True),
            ClientPhoto(url="https://img.example/wiki-cdn.jpg", fromSource=True),
        ],
    )
    asyncio.run(SearchProfile(Settings(_env_file=None, llm_api_key=""), None, None, Photos())._attach_photos([person]))
    assert [p.url for p in person.photos] == ["https://img.example/wiki.jpg", "https://img.example/wiki-cdn.jpg"]
    assert person.avatarUrl == "https://img.example/wiki.jpg"


def test_pages_yield_social_accounts_and_wikipedia_extlinks():
    from app.search.deep_search import linked_profiles
    from app.infra.fetcher import extlinks_from_payload

    links = linked_profiles(
        [
            "https://www.facebook.com/ChiDanSinger",
            "https://www.instagram.com/nguyentrunghieuuu/",
            "https://www.facebook.com/ChiDanSinger/posts/123",
            "https://www.youtube.com/watch?v=abc",
            "https://tuoitre.vn/chi-dan.htm",
        ],
        "Chi Dân",
        "https://en.wikipedia.org/wiki/Chi_Dan",
    )
    assert [s.type for s in links] == ["facebook", "instagram"]
    assert links[0].verified and "wikipedia.org" in (links[0].snippet or "")

    # Every account written on the profile is kept. An article URL is not a profile.
    from app.search.deep_search import classify_links

    profiles, articles = classify_links(
        [
            "https://www.facebook.com/MissLeNguyenBaoNgoc",
            "https://www.facebook.com/baodientuvov.vn",
            "https://www.instagram.com/bn_lenguyen/",
            "https://www.tiktok.com/@bn_lenguyen",
            "https://www.youtube.com/@baodientuvov",
            "https://tuoitre.vn/le-nguyen-bao-ngoc.htm",
        ],
        "Lê Nguyễn Bảo Ngọc",
        "https://www.youtube.com/@lenguyenbaongoc",
    )
    assert [s.type for s in profiles] == ["facebook", "facebook", "instagram", "tiktok", "youtube"]
    assert articles[0].url.startswith("https://tuoitre.vn/")

    payload = {"query": {"pages": {"1": {"extlinks": [{"*": "https://www.instagram.com/nguyentrunghieuuu/"}, {"url": "https://x.com/chidan20"}]}}}}
    assert extlinks_from_payload(payload) == ["https://www.instagram.com/nguyentrunghieuuu/", "https://x.com/chidan20"]
    assert extlinks_from_payload({}) == []


def test_channel_about_keeps_contact_and_social_links():
    from app.infra.about_profiles import github_from_payload, public_text, youtube_from_info
    from app.search.deep_search import profiles_copied_from_text

    description = """
    Chào bạn! Mình là Lê Phương Anh – HLV Yoga, Body Coach.
    📩 Liên hệ: https://www.facebook.com/Lephuonganhfityoungbody/
    Nguồn: https://www.facebook.com/baodientuvov.vn
    ☎️ 0974 34 3333
    mail me at le.phuonganh@example.com
    """
    text, audience = youtube_from_info({"description": description, "channel_follower_count": 431})
    assert "0974" in text and "le.phuonganh@example.com" in text
    assert "https://www.facebook.com/Lephuonganhfityoungbody/" in text
    assert "HLV Yoga" in text and "431 subscribers" in text
    assert audience == "431 subscribers"
    assert public_text("431 subscribers") == "431 subscribers"
    copied = profiles_copied_from_text(
        [
            "https://www.facebook.com/Lephuonganhfityoungbody/",
            "https://www.facebook.com/baodientuvov.vn",
            "https://www.facebook.com/not-in-the-text",
        ],
        "Lê Phương Anh",
        {"yt": text},
    )
    assert [s.url for s in copied] == [
        "https://www.facebook.com/Lephuonganhfityoungbody",
        "https://www.facebook.com/baodientuvov.vn",
    ]

    bio, followers = github_from_payload(
        {
            "bio": "Yoga coach. https://www.instagram.com/thefitandyoungbody/",
            "blog": "lephuonganh.vn",
            "twitter_username": "lephuonganh",
            "company": "The Fit & Young Body",
            "location": "Hồ Chí Minh",
            "followers": 120,
            "email": "hidden@example.com",
        }
    )
    assert "hidden@example.com" in bio and "https://x.com/lephuonganh" in bio and "https://www.instagram.com/thefitandyoungbody/" in bio
    assert "The Fit & Young Body" in bio and followers == "120 followers"
    accounts = profiles_copied_from_text(
        ["https://www.instagram.com/thefitandyoungbody/", "https://x.com/lephuonganh", "https://www.facebook.com/invented"],
        "Lê Phương Anh",
        {"gh": bio},
    )
    assert [s.type for s in accounts] == ["instagram", "x"]


def test_deep_search_crawls_the_selected_account_even_when_the_title_uses_another_name():
    from app.search.deep_search import _crawlable
    from app.core.models import ClientSource
    from app.core.urls import link_key

    ig = ClientSource(url="https://www.instagram.com/namkhanh/", title="Bùi Nam Khánh · Digital creator", type="instagram", verified=True)
    trusted = {link_key(ig.url)}
    assert _crawlable(ig, "Nam Khánh", trusted)
    assert not _crawlable(
        ClientSource(url="https://www.instagram.com/someoneelse/", title="Someone Else", type="instagram", verified=True),
        "Nam Khánh",
        trusted,
    )


def test_deep_search_stays_on_the_selected_card():
    from app.search.deep_search import article_query, fits_selected, markers
    from app.core.models import ClientCandidate, ClientKeyFact, ClientSource

    intern = ClientCandidate(
        id="tung-ig",
        displayName="Bùi Thanh Tùng",
        confidence=0.8,
        imageSearchQuery="Bui Thanh Tung ttung_167 intern",
        occupation="Intern",
        headline="Internship Student",
        sources=[ClientSource(url="https://www.instagram.com/ttung_167/", title="ttung_167", type="instagram", verified=True)],
        keyFacts=[ClientKeyFact(key="company", label="Company", value="VNU IS", verified=True)],
    )
    assert "ttung" in markers(intern) or "167" in "".join(markers(intern))
    assert "vnu" in markers(intern) or "intern" in markers(intern)
    assert "ttung_167" in article_query(intern)
    assert fits_selected(intern, "Bùi Thanh Tùng intern at VNU IS", "ttung_167")
    assert not fits_selected(intern, "PGS.TS Bùi Thanh Tùng ngành dược", "ứng viên giáo sư trẻ nhất ngành dược học")
    assert not fits_selected(intern, "Tung Bui Thanh Apple developer", "https://apps.apple.com/developer/tung-bui-thanh")


def test_deep_search_uses_social_photos_without_serper():
    from app.config import Settings
    from app.search.deep_search import DeepSearch
    from app.core.models import ClientCandidate, ClientPhoto, ClientSource

    class Search:
        async def search(self, *a, **k):
            raise AssertionError("deep search must not image-search; photos come from social/source pages")

    svc = DeepSearch(Settings(_env_file=None, llm_api_key=""), Search(), None, None, None)
    ig = ClientSource(url="https://www.instagram.com/ttung_167/", title="ttung_167", type="instagram", verified=True)
    c = ClientCandidate(id="tung", displayName="Bùi Thanh Tùng", confidence=0.8, imageSearchQuery="Bui Thanh Tung intern", sources=[ig])
    one = [ClientPhoto(url="https://cdn.example/ava.jpg", pageUrl="https://www.instagram.com/ttung_167/", fromSource=True)]
    photos, _, note, avatar = __import__("asyncio").run(svc._photos(c, [ig], [], ("vn", "vi"), one))
    assert [p.url for p in photos] == ["https://cdn.example/ava.jpg"]
    assert avatar == "https://cdn.example/ava.jpg" and note is None

    full = [
        ClientPhoto(url=f"https://cdn.example/{i}.jpg", pageUrl="https://www.instagram.com/ttung_167/", fromSource=True)
        for i in range(10)
    ]
    photos, _, note, avatar = __import__("asyncio").run(
        DeepSearch(Settings(_env_file=None, llm_api_key=""), Search(), None, None, None)._photos(c, [], [], ("vn", "vi"), full)
    )
    assert [p.url for p in photos] == [p.url for p in full] and avatar == full[0].url and note is None


def test_deep_search_without_profile_page_looks_up_profiles_by_name():
    import asyncio

    from app.config import Settings
    from app.search.deep_search import DeepSearch
    from app.core.models import ClientCandidate, ClientSource, SourcesResponse, StepMeta

    class Lookup:
        async def social(self, req):
            self.req = req
            tt = ClientSource(url="https://www.tiktok.com/@quynhalee", title="Quỳnh Alee (@quynhalee)", type="tiktok", verified=True)
            return SourcesResponse(sources=[tt], meta=StepMeta(searchCalls=2))

    lookup = Lookup()
    svc = DeepSearch(Settings(_env_file=None, llm_api_key=""), None, None, None, None, sources=lookup)  # type: ignore[arg-type]
    news = ClientSource(url="https://kenh14.vn/quynh-alee.html", title="Quỳnh Alee - Tin tức", type="web", verified=True)
    c = ClientCandidate(id="q", displayName="Quỳnh Alee", confidence=0.9, occupation="TikToker", imageSearchQuery="Quỳnh Alee", sources=[news])
    found, calls = asyncio.run(svc._find_profiles(c, {}))
    assert [s.url for s in found.sources] == ["https://www.tiktok.com/@quynhalee", "https://kenh14.vn/quynh-alee.html"]
    assert calls == 2 and lookup.req.name == "Quỳnh Alee" and lookup.req.occupation == "TikToker"


def test_deep_search_splits_social_profiles_from_articles():
    from app.core.client_format import social_profiles_markdown
    from app.search.deep_search import apply_dossier, split_sources
    from app.core.llm import DeepProfileOutput, EnrichedFact
    from app.core.models import ClientCandidate, ClientSource
    from app.core.urls import link_key

    social = ClientSource(url="https://www.instagram.com/chidan/", title="Chi Dân (@chidan)", type="instagram", snippet="2399355 Followers", verified=True)
    wiki = ClientSource(url="https://en.wikipedia.org/wiki/Chi_Dan", title="Chi Dân - Wikipedia", type="wikipedia", snippet="Vietnamese singer and songwriter", verified=True)
    article = ClientSource(url="https://tuoitre.vn/chi-dan-ra-mat-bai-hat-20240101120000000.htm", title="Chi Dân ra mắt bài hát", type="web", snippet="Chi Dân", verified=True)
    invented = "https://evil.example/nope"
    profiles, articles = split_sources(
        "Chi Dân",
        [social, wiki],
        [article],
        [social.url, invented],
        [article.url, wiki.url, invented],
        trusted={link_key(social.url), link_key(wiki.url)},
    )
    assert [s.type for s in profiles] == ["wikipedia", "instagram"]
    assert {s.url for s in articles} == {article.url}
    assert social_profiles_markdown(profiles) == "[Instagram · @chidan](https://www.instagram.com/chidan/)"

    candidate = ClientCandidate(
        id="chi-dan",
        displayName="Chi Dân",
        confidence=0.8,
        imageSearchQuery="Chi Dan singer",
        sources=[social, wiki],
        location="Hồ Chí Minh",
    )
    out = DeepProfileOutput(
        headline="Singer and songwriter",
        profile=["Chi Dân is a Vietnamese singer.", "He releases V-Pop songs."],
        facts=[
            EnrichedFact(key="jobTitle", label="Role", value="Singer and songwriter"),
            EnrichedFact(key="born", label="Born", value="June 2, 1989"),
            EnrichedFact(key="notableWorks", label="Notable works", value="Điều Anh Biết, 1234"),
        ],
        socialUrls=[social.url],
        articleUrls=[article.url, wiki.url],
    )
    merged = apply_dossier(candidate, out, profiles, articles)
    by_key = {f.key: f for f in merged.keyFacts}
    assert by_key["jobTitle"].label == "Current role" and by_key["jobTitle"].verified is False
    assert "born" not in by_key  # the wikipedia snippet does not state 1989
    stated = apply_dossier(
        candidate,
        out,
        profiles,
        articles,
        {"ig": "Sinh năm 1999. Sống tại Hà Nội. Học tại Đại học Sân khấu. Sống tại 12 đường Lê Lợi."},
    )
    stated_by = {f.key: f for f in stated.keyFacts}
    assert stated_by["born"].value == "1999" and stated_by["born"].verified
    assert stated_by["location"].value == "Hà Nội"
    assert "Đại học" in stated_by["education"].value
    assert "Lê Lợi" not in stated_by["location"].value
    assert by_key["notableWorks"].value == "Điều Anh Biết, 1234"
    assert by_key["socialProfiles"].value == "Instagram · @chidan"
    assert "2399355" in by_key["audience"].value
    assert merged.profileSource == "llm" and len(merged.profile) == 2
    assert merged.facts["socialProfiles"] == by_key["socialProfiles"].value
    assert [s.type for s in merged.sources] == ["wikipedia", "instagram", "web"]


def test_certain_profile_rejects_a_shared_token_and_an_opaque_id():
    from app.search.deep_search import allows_name_image_search, is_public_figure, split_sources
    from app.core.models import ClientSource
    from app.core.urls import certain_profile, link_key

    assert certain_profile("https://x.com/Cristiano", "Cristiano Ronaldo")
    assert certain_profile("https://instagram.com/leomessi", "Lionel Messi", "Messi")
    assert certain_profile("https://www.facebook.com/MissLeNguyenBaoNgoc", "Lê Nguyễn Bảo Ngọc")
    assert not certain_profile("https://instagram.com/bn_lenguyen/", "Lê Nguyễn Bảo Ngọc")
    assert not certain_profile("https://facebook.com/nguyenminhtuan", "Nguyễn Minh Khang")
    assert not certain_profile("https://youtube.com/channel/UClyA28", "Sơn Tùng M-TP")
    assert not certain_profile("https://www.facebook.com/profile.php?id=100099", "Lê Nguyễn Bảo Ngọc")

    namesake = ClientSource(url="https://facebook.com/nguyenminhtuan", title="Nguyễn Minh Khang · facebook", type="facebook", verified=True)
    channel = ClientSource(url="https://www.youtube.com/channel/UClyA28", title="Sơn Tùng M-TP - YouTube", type="youtube", verified=True)
    chosen = ClientSource(url="https://www.youtube.com/@khanhlinhff02", title="Trần Khánh Linh - YouTube", type="youtube", verified=True)
    kept, _ = split_sources("Nguyễn Minh Khang", [namesake], [], [], [])
    assert kept == []
    kept, _ = split_sources("Sơn Tùng M-TP", [channel], [], [], [], trusted={link_key(channel.url)})
    assert [s.url for s in kept] == [channel.url]
    kept, _ = split_sources("Trần Khánh Linh", [chosen], [], [], [], trusted={link_key(chosen.url)})
    assert [s.url for s in kept] == [chosen.url]
    stage = ClientSource(url="https://www.instagram.com/bbrightvc/", title="bbrightvc", type="instagram", verified=True)
    wiki_bright = ClientSource(url="https://en.wikipedia.org/wiki/Vachirawit_Chivaaree", title="Vachirawit Chivaaree", type="wikipedia", verified=True)
    kept, arts = split_sources(
        "Vachirawit Chivaaree",
        [stage, wiki_bright],
        [ClientSource(url="https://tuoitre.vn/bright.htm", title="tin", type="web", verified=True)],
        [],
        ["https://tuoitre.vn/bright.htm"],
        trusted={link_key(stage.url), link_key(wiki_bright.url)},
    )
    assert [s.type for s in kept] == ["wikipedia", "instagram"]
    assert arts[0].url.startswith("https://tuoitre.vn/")
    es = ClientSource(url="https://es.wikipedia.org/wiki/Vachirawit_Chiva-aree", title="Vachirawit Chiva-aree", type="wikipedia", verified=True)
    en = ClientSource(url="https://en.wikipedia.org/wiki/Vachirawit_Chivaaree", title="Vachirawit Chivaaree", type="wikipedia", verified=True)
    ig = ClientSource(url="https://www.instagram.com/bbrightvc/", title="bbrightvc", type="instagram", verified=True)
    kept, _ = split_sources(
        "Vachirawit Chivaaree",
        [es, ig, en],
        [],
        [],
        [en.url],
        trusted={link_key(es.url), link_key(ig.url), link_key(en.url)},
    )
    assert [s.url for s in kept] == [en.url, ig.url]
    assert allows_name_image_search(False, True) and not allows_name_image_search(True, True)
    assert allows_name_image_search(False, False)
    wiki = ClientSource(url="https://vi.wikipedia.org/wiki/Son_Tung_M-TP", title="Sơn Tùng M-TP", type="wikipedia", verified=True)
    assert is_public_figure("Sơn Tùng M-TP", [wiki])
    assert not is_public_figure("Nguyễn Minh Khang", [namesake])


def test_a_missing_profile_is_dead_and_a_block_is_not():
    from app.infra.fetcher import link_is_dead
    from app.infra.social_crawl import SocialCrawl

    assert link_is_dead(404) and link_is_dead(410) and link_is_dead(0)
    assert not link_is_dead(200) and not link_is_dead(403) and not link_is_dead(None)
    crawl = SocialCrawl("key", "https://api.crawlfeed.dev", 5, 60)
    crawl.cache.set(("facebook", "gone"), ({}, 0))
    crawl.cache.set(("instagram", "live"), ({"display_name": "Live"}, 1))
    assert crawl.known_missing("https://www.facebook.com/gone")
    assert crawl.known_live("https://www.instagram.com/live/")
    assert not crawl.known_missing("https://www.instagram.com/live/")
    assert not crawl.known_missing("https://www.tiktok.com/@never-looked-up")


def test_scrape_creators_profile_request_uses_handle_or_url():
    from app.infra.social_crawl import scrape_request, unwrap_profile

    path, q = scrape_request("facebook", "mantraindianfolsom", "https://www.facebook.com/mantraindianfolsom")
    assert path == "/v1/facebook/profile" and q["url"].startswith("https://www.facebook.com/")
    path, q = scrape_request("instagram", "zuck")
    assert path == "/v1/instagram/profile" and q["handle"] == "zuck" and q["trim"] == "true"
    path, q = scrape_request("x", "Austen")
    assert path == "/v1/twitter/profile" and q["handle"] == "Austen"
    path, q = scrape_request("linkedin", "parrsam", "https://www.linkedin.com/in/parrsam/")
    assert path == "/v1/linkedin/profile" and "linkedin.com/in/parrsam" in q["url"]
    path, q = scrape_request("threads", "yangmi")
    assert path == "/v1/threads/profile" and q["handle"] == "yangmi"

    ig = unwrap_profile({"success": True, "credits_charged": 1, "data": {"user": {"full_name": "Ada", "biography": "Hi", "profile_pic_url": "https://scontent.cdninstagram.com/v/ava"}}})
    assert ig["display_name"] == "Ada" and ig["bio"] == "Hi" and ig["avatar_url"].startswith("https://")
    tw = unwrap_profile({"credits_charged": 1, "legacy": {"name": "Austen", "description": "CEO", "profile_image_url_https": "https://pbs.twimg.com/a_normal.jpg", "followers_count": 9}})
    assert tw["display_name"] == "Austen" and tw["bio"] == "CEO" and tw["followers"] == 9


def test_crawlfeed_profile_text_keeps_contact_and_targets_walled_networks():
    from app.core.models import ClientSource
    from app.infra.social_crawl import SocialCrawl, crawl_target, profile_text
    from app.core.urls import link_key

    assert crawl_target("https://www.tiktok.com/@lephuonganh") == ("tiktok", "lephuonganh")
    assert crawl_target("https://www.linkedin.com/in/le-phuong-anh/") == ("linkedin", "le-phuong-anh")
    assert crawl_target("https://www.youtube.com/@LePhuongAnhFitYoungBody") is None
    assert crawl_target("https://www.facebook.com/profile.php?id=1000123") == (
        "facebook",
        "https://www.facebook.com/profile.php?id=1000123",
    )
    text = profile_text(
        {
            "display_name": "Lê Phương Anh",
            "handle": "lephuonganh",
            "bio": "Coach. Call 0974 34 3333 or mail a@b.com",
            "followers": 431,
            "following": 12,
            "category": "Fitness Coach",
            "links": ["https://www.facebook.com/Lephuonganhfityoungbody/"],
            "raw": {"work": "The Fit & Young Body", "education": "HCMC", "hometown": "Huế"},
        }
    )
    assert "0974" in text and "a@b.com" in text
    assert "https://www.facebook.com/Lephuonganhfityoungbody/" in text
    assert "431" in text and "Fitness Coach" in text and "The Fit & Young Body" in text
    assert "Huế" in text and "lephuonganh" in text
    from app.infra.search_adapter import result_image
    from app.infra.social_crawl import photos_from_profile

    url, fb = result_image({"imageUrl": "https://media.licdn.com/dms/image/v2/C4E03/haoran.jpg", "thumbnailUrl": "https://encrypted-tbn0.gstatic.com/x"}, "web")
    assert url.endswith("haoran.jpg") and fb.startswith("https://encrypted-tbn0.gstatic.com/")
    crawled = photos_from_profile(
        {"raw": {"profilePicture": {"displayImage": "https://media.licdn.com/dms/image/v2/C4E03/from-raw.jpg"}}},
        "https://www.linkedin.com/in/haoranchen-uva/",
        "Haoran Chen",
    )
    assert crawled[0].url.endswith("from-raw.jpg") and crawled[0].fromSource is True

    class Fake(SocialCrawl):
        async def _profile(self, platform, handle):
            assert platform == "tiktok" and handle == "lephuonganh"
            return {
                "display_name": "Lê Phương Anh",
                "bio": "Coach",
                "avatar_url": "https://cdn.example/a.jpg",
                "links": ["https://www.instagram.com/thefitandyoungbody/"],
            }, 1

    source = ClientSource(url="https://www.tiktok.com/@lephuonganh", title="Lê Phương Anh", type="tiktok", verified=True)

    class FakeIg(Fake):
        async def _profile(self, platform, handle):
            if platform == "instagram":
                return {"display_name": "Phương Anh", "bio": "IG bio", "category": "Creator"}, 1
            return await super()._profile(platform, handle)

    batch = __import__("asyncio").run(FakeIg("key", "https://api.crawlfeed.dev", 5, 60).read([source]))
    assert batch.calls == 2 and batch.credits == 2
    assert "Coach" in batch.text[link_key("https://www.tiktok.com/@lephuonganh")]
    assert any("IG bio" in t for t in batch.text.values())
    assert batch.photos[0].url == "https://cdn.example/a.jpg" and batch.photos[0].fromSource is True
    assert __import__("asyncio").run(SocialCrawl("", "https://api.crawlfeed.dev", 5, 60).read([source])).calls == 0

    class Two(SocialCrawl):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.seen: list[str] = []

        async def _profile(self, platform, handle):
            self.seen.append(platform)
            return {"display_name": platform, "avatar_url": f"https://cdn.example/{platform}.jpg"}, 1

    many = [
        ClientSource(url="https://www.linkedin.com/in/a/", title="A", type="linkedin", verified=True),
        ClientSource(url="https://www.tiktok.com/@a", title="A", type="tiktok", verified=True),
        ClientSource(url="https://x.com/a", title="A", type="x", verified=True),
        ClientSource(url="https://www.instagram.com/a/", title="A", type="instagram", verified=True),
        ClientSource(url="https://www.facebook.com/a", title="A", type="facebook", verified=True),
    ]
    two = Two("key", "https://api.crawlfeed.dev", 5, 60)
    batch = __import__("asyncio").run(two.read(many))
    assert set(two.seen) == {"facebook", "instagram"} and batch.calls == 2
    assert [p.url for p in batch.photos] == ["https://cdn.example/facebook.jpg", "https://cdn.example/instagram.jpg"]

    from app.infra.social_crawl import hrefs_from_profile, linked_from_profile, photos_from_profile
    from app.core.urls import looks_like_image_url, profile_of

    ig = {
        "display_name": "Yang Mi",
        "avatar_url": "https://scontent.cdninstagram.com/v/t51.2885-19/s150x150/ava",
        "links": [
            {"title": "Facebook", "url": "https://l.instagram.com/?u=https%3A%2F%2Fwww.facebook.com%2Fyangmi%2F"},
            {"title": "Threads", "url": "https://www.threads.net/@yangmi"},
            "https://scontent.cdninstagram.com/v/t51.2885-15/post",
        ],
        "raw": {"bio_links": [{"url": "https://www.facebook.com/yangmi/"}], "display_url": "https://scontent.cdninstagram.com/v/t51.2885-15/another"},
    }
    hrefs = hrefs_from_profile(ig)
    assert any(profile_of(u) and profile_of(u).platform == "facebook" for u in hrefs)
    assert any(profile_of(u) and profile_of(u).platform == "threads" for u in hrefs)
    assert crawl_target("https://www.threads.net/@yangmi") == ("threads", "yangmi")
    assert profile_of("https://l.instagram.com/?u=https%3A%2F%2Fwww.facebook.com%2Fyangmi%2F").url == "https://www.facebook.com/yangmi"
    linked = linked_from_profile(ig, "")
    assert any("facebook.com/yangmi" in u for u in linked)
    assert any("threads.net/@yangmi" in u for u in linked)
    pics = photos_from_profile(ig, "https://www.instagram.com/yangmi/", "Yang Mi")
    assert pics[0].fromSource is True
    assert any(looks_like_image_url(p.url) for p in pics)
    assert len(pics) >= 2


def test_yescale_usage_reads_prompt_tokens_not_zero_aliases():
    from types import SimpleNamespace as NS

    from app.core.tracing import extract_usage, parse_usage

    # Live Yescale /chat/completions body (gemini-3.7-flash): real counts sit on
    # prompt_tokens / completion_tokens. input_tokens / output_tokens are always 0.
    yescale = {
        "prompt_tokens": 6,
        "completion_tokens": 74,
        "total_tokens": 80,
        "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 6, "audio_tokens": 0, "image_tokens": 0},
        "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "image_tokens": 0, "reasoning_tokens": 0},
        "input_tokens": 0,
        "output_tokens": 0,
        "input_tokens_details": None,
        "claude_cache_creation_5_m_tokens": 0,
        "claude_cache_creation_1_h_tokens": 0,
    }
    assert parse_usage(yescale) == (6, 74, 80)
    last_chunk = NS(choices=[], usage=yescale)
    assert parse_usage(extract_usage(last_chunk)) == (6, 74, 80)
    wrapped = NS(usage=NS(**{k: v for k, v in yescale.items() if k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")}))
    assert parse_usage(extract_usage(wrapped)) == (6, 74, 80)


def test_yescale_cost_matches_dashboard():
    from app.config import Settings
    from app.core.cost import yescale_llm

    # Dashboard: 1052 * 0.75$/1M + 114 * 3.75$/1M = 0.0012165 credit × 8000 = 9.73 ₫
    billed = yescale_llm(1052, 114, input_per_1m=0.75, output_per_1m=3.75, group_ratio=1.0, vnd_per_credit=8000)
    assert abs(billed.input_usd - 0.000789) < 1e-12
    assert abs(billed.output_usd - 0.0004275) < 1e-12
    assert abs(billed.credits - 0.0012165) < 1e-12
    assert round(billed.vnd, 2) == 9.73
    s = Settings(_env_file=None, llm_price_input_per_1m="", llm_price_output_per_1m="", cost_currency="")
    assert s.llm_price_input_per_1m == 0.75
    assert s.llm_price_output_per_1m == 3.75
    assert s.yescale_vnd_per_credit == 8000
    assert s.cost_currency == "VND"


def test_mark_vnd_is_noop_without_langfuse():
    from app.core.tracing import _Noop, mark_vnd

    mark_vnd(_Noop(), 37.4, on_trace=True)


def test_serper_token_is_26_vnd():
    from app.config import Settings
    from app.core.cost import extra_vnd

    s = Settings(_env_file=None, serper_price_per_search=26, scrapecreators_price_per_credit=49)
    assert extra_vnd(serper_calls=2, settings=s) == 52
    assert extra_vnd(scrape_credits=1, settings=s) == 49
    assert extra_vnd(serper_calls=1, scrape_credits=2, settings=s) == 26 + 98
    from app.core.cost import scrape_cost_details, serper_cost_details, usd_from_vnd

    usage, cost = serper_cost_details(2, settings=s)
    assert usage == {"input": 2, "output": 0, "total": 2}
    assert cost["total"] == usd_from_vnd(52, settings=s)
    usage, cost = scrape_cost_details(1, settings=s)
    assert usage == {"input": 1, "output": 0, "total": 1}
    assert cost["total"] == usd_from_vnd(49, settings=s)


def test_parallel_serper_queries_each_bill_one_token():
    import asyncio

    from app.core import tracing
    from app.infra.mcp_manager import ToolCallResult
    from app.core.models import ToolCallLog
    from app.infra.search_adapter import SearchAdapter

    class Mgr:
        def servers_for(self, cap):
            return [("serper", "web_search")]

        async def call(self, server, tool, args):
            await asyncio.sleep(0.02)
            return ToolCallResult(
                server=server,
                tool=tool,
                data={"provider": "serper", "results": [{"title": "t", "url": "https://x.com/a"}]},
                latency_ms=1,
            )

    adapter = SearchAdapter(Mgr(), cache_ttl_s=60, gl="vn", hl="vi", num=10)
    logs: list[ToolCallLog] = []

    async def run():
        with tracing.span("test-search"):
            await asyncio.gather(
                adapter.search("Nguyễn Thanh Tùng wiki", "web", logs),
                adapter.search("Nguyễn Thanh Tùng social", "web", logs),
            )
            st = tracing.current_stats()
            assert st is not None
            assert st.serper_calls == 2
            assert sorted(l.detail for l in st.lines) == ["Nguyễn Thanh Tùng social", "Nguyễn Thanh Tùng wiki"]
            assert all(l.token_in == 1 for l in st.lines)

    asyncio.run(run())
    assert len(logs) == 2


def test_failed_scrape_still_shows_on_the_cost_line(monkeypatch):
    import asyncio

    import httpx

    from app.core import tracing
    from app.core.models import ClientSource
    from app.infra.social_crawl import SocialCrawl

    class FailClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a, **k):
            return False

        async def get(self, *a, **k):
            raise httpx.ReadTimeout("")

    monkeypatch.setattr("app.infra.social_crawl.httpx.AsyncClient", FailClient)
    src = ClientSource(url="https://www.instagram.com/lsnguyenthanhtung/", title="Tung", type="instagram", verified=True)

    async def run():
        with tracing.span("test-scrape"):
            batch = await SocialCrawl("key", "https://api.scrapecreators.com", 5, 60).read([src])
            st = tracing.current_stats()
            assert batch.calls == 1 and batch.credits == 0
            assert batch.errors and "ReadTimeout" in batch.errors[0]
            assert st is not None and st.scrape_calls == 1
            assert st.lines and "instagram" in st.lines[0].detail and "ReadTimeout" in st.lines[0].detail

    asyncio.run(run())
