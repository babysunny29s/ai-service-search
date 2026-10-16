"""Photos from a person's source pages (no search, no LLM), with fake pages and faces. No network."""

import numpy as np

from app.config import Settings
from app.media.faces import Face
from app.infra.fetcher import FetchedPage, named_images, parse_html
from app.core.models import ClientSource
from app.media.source_photos import SourcePhotoFinder


def face(v):
    v = np.array(v, dtype=np.float32)
    return Face((0, 0, 10, 10), 1.0, v / np.linalg.norm(v))


# Other photos of the same person: close faces, not identical (identical = the same picture).
ME, OTHER = face([1, 0, 0]), face([0, 1, 0])
ME_B, ME_C, ME_D = face([1, 0.5, 0]), face([1, 0, 0.5]), face([1, 0.4, 0.4])
FACES = {
    "https://upload.wikimedia.org/a.jpg": [ME],
    "https://nguyenvana.dev/logo.png": [],
    "https://news.example/a-portrait.jpg": [ME_B],
    "https://blog.example/someone.jpg": [OTHER],
    "https://github.com/nguyenvana.png": [ME],
}
PAGES = {
    "https://nguyenvana.dev/": "https://nguyenvana.dev/logo.png",
    "https://news.example/interview": "https://news.example/a-portrait.jpg",
    "https://blog.example/post": "https://blog.example/someone.jpg",
}


class Fetcher:
    def __init__(self):
        self.fetched: list[str] = []

    async def fetch(self, url):
        self.fetched.append(url)
        return FetchedPage(url, "fetched", image=PAGES.get(url)) if url in PAGES else FetchedPage(url, "failed")

    async def wikipedia_thumbnail(self, url):
        self.fetched.append(url)
        return "https://upload.wikimedia.org/a.jpg" if url.endswith("/Nguyen_Van_A") else "https://upload.wikimedia.org/other.jpg"

    async def wikipedia_images(self, url, limit=10):
        thumb = await self.wikipedia_thumbnail(url)
        return [thumb] if thumb else []


class Faces:
    def __init__(self):
        self.checked: list[str] = []

    async def ready(self):
        return True

    async def faces_in(self, urls):
        self.checked += urls
        return [FACES.get(u) for u in urls]


def src(url, type_="web", verified=True):
    return ClientSource(url=url, type=type_, verified=verified)


SOURCES = [
    src("https://www.instagram.com/nguyenvana/", "instagram"),
    src("https://blog.example/post"),
    src("https://news.example/interview"),
    src("https://nguyenvana.dev/", "website"),
    src("https://vi.wikipedia.org/wiki/Nguyen_Van_A", "wikipedia"),
]


def finder(fetcher, faces=None, **kw):
    return SourcePhotoFinder(Settings(_env_file=None, llm_api_key="", **kw), fetcher, faces)


async def test_keeps_images_showing_the_person_from_their_pages():
    fetcher = Fetcher()
    photos, checked = await finder(fetcher, Faces()).find("Nguyen Van A", SOURCES)
    # Wikipedia gives the reference face; the logo (no face) and someone else's face are dropped.
    assert [(p.url, p.pageUrl, p.faceMatch) for p in photos] == [
        ("https://upload.wikimedia.org/a.jpg", "https://vi.wikipedia.org/wiki/Nguyen_Van_A", 1.0),
        ("https://news.example/a-portrait.jpg", "https://news.example/interview", 0.894),
    ]
    assert checked and all(p.fromSource for p in photos)
    assert fetcher.fetched[0].endswith("/wiki/Nguyen_Van_A")  # most likely portrait first
    assert not any("instagram.com" in u for u in fetcher.fetched)  # login-walled: never fetched


async def test_without_face_model_only_the_persons_own_pages():
    photos, checked = await finder(Fetcher()).find("Nguyen Van A", SOURCES)
    assert [p.url for p in photos] == ["https://upload.wikimedia.org/a.jpg", "https://nguyenvana.dev/logo.png"]
    assert not checked and all(p.faceMatch is None for p in photos)


async def test_unverified_wikipedia_link_counts_only_when_named_after_the_person():
    links = [src("https://vi.wikipedia.org/wiki/Nguyen_Van_A", verified=False), src("https://en.wikipedia.org/wiki/Someone_Else", verified=False)]
    photos, _ = await finder(Fetcher()).find("Nguyễn Văn A", links)
    assert [p.url for p in photos] == ["https://upload.wikimedia.org/a.jpg"]


async def test_github_avatar_and_page_limit():
    sources = [src("https://github.com/nguyenvana", "github"), *SOURCES[1:4]]
    fetcher = Fetcher()
    photos, checked = await finder(fetcher, Faces(), source_photo_pages=2).find("Nguyen Van A", sources)
    # GitHub then the official site: only 2 pages fetched (the avatar URL needs no fetch).
    assert fetcher.fetched == ["https://nguyenvana.dev/"]
    assert [p.url for p in photos] == ["https://github.com/nguyenvana.png"] and checked


STAFF_PAGE = """<html><head><title>TS. Nguyễn Trường Giang</title>
<meta property="og:image" content="https://scls.hust.edu.vn/uploads/scls/truong-hoa-va-khss-04_750_669.png"></head><body>
<header><img src="/uploads/scls/logo-dhbk-1-02_130_191.png" alt="Trường Hóa và Khoa học Sự sống"></header>
<main><h1>TS. Nguyễn Trường Giang</h1><p>Giảng viên Khoa Kỹ thuật Sinh học, Trường Hóa và Khoa học Sự sống, Đại học Bách khoa Hà Nội.</p>
<img src="/assets/scls/organs/2024_01/scls.ktsh.nguyen-truong-giang.jpg" class="img-thumbnail"
     data-src="/uploads/scls/organs/2024_01/scls.ktsh.nguyen-truong-giang.jpg">
<img src="/uploads/scls/organs/2024_01/scls.ktsh.tran-van-b.jpg" alt="TS. Trần Văn B">
<img alt="cron" src="/index.php?second=cronjobs" width="1" height="1"></main></body></html>"""
STAFF_URL = "https://scls.hust.edu.vn/vi/organs/person/Khoa-Ky-thuat-Sinh-hoc-15/TS-Nguyen-Truong-Giang-111/"
PORTRAIT = "https://scls.hust.edu.vn/uploads/scls/organs/2024_01/scls.ktsh.nguyen-truong-giang.jpg"


def test_staff_page_portrait_is_found_in_the_body():
    p = parse_html(STAFF_PAGE, STAFF_URL)
    # Full-size lazy-load file first; the logo and the cron pixel are not photos.
    assert [u for u, _ in p.images] == [PORTRAIT, "https://scls.hust.edu.vn/uploads/scls/organs/2024_01/scls.ktsh.tran-van-b.jpg"]
    page = FetchedPage(STAFF_URL, "fetched", image=p.image, images=p.images)
    assert named_images(page, "Nguyễn Trường Giang") == [PORTRAIT]
    assert named_images(page, "TS. Nguyễn Trường Giang") == [PORTRAIT]


async def test_named_portrait_on_a_staff_page_counts_as_the_persons_photo():
    p = parse_html(STAFF_PAGE, STAFF_URL)

    class StaffFetcher(Fetcher):
        async def fetch(self, url):
            return FetchedPage(url, "fetched", image=p.image, images=p.images)

    faces = Faces()
    FACES[PORTRAIT] = [ME]
    FACES[p.image] = []
    try:
        photos, checked = await finder(StaffFetcher(), faces).find("Nguyễn Trường Giang", [src(STAFF_URL, "academic")])
        # Academic pages are not the person's own, but the portrait carries the name; the og:image logo has no face.
        assert [(ph.url, ph.faceMatch) for ph in photos] == [(PORTRAIT, 1.0)] and checked
        unchecked, _ = await finder(StaffFetcher()).find("Nguyễn Trường Giang", [src(STAFF_URL, "academic")])
        assert [ph.url for ph in unchecked] == [PORTRAIT]
    finally:
        del FACES[PORTRAIT], FACES[p.image]


async def test_preview_source_photo_is_the_profile_photo_and_the_face_reference():
    from app.core.models import ClientCandidate, ClientPhoto, ClientSource
    from app.search.preview import PreviewSearch
    from app.infra.search_adapter import SearchHit

    own = "https://upload.wikimedia.org/a.jpg"
    # copy.jpg is the source photo again under another URL.
    extra = {"https://img.example/copy.jpg": [ME], "https://img.example/me1.jpg": [ME_B], "https://img.example/me2.jpg": [ME_C], "https://img.example/me3.jpg": [ME_D]}
    hits = ["https://img.example/copy.jpg", "https://img.example/me1.jpg", "https://blog.example/someone.jpg", "https://img.example/me2.jpg", "https://img.example/me3.jpg"]
    faces = Faces()
    FACES.update(extra)

    class Search:
        async def search(self, q, kind, logs, locale=None):
            return [
                SearchHit(
                    provider="fake",
                    query=q,
                    title="Nguyen Van A",
                    url="https://en.wikipedia.org/wiki/Nguyen_Van_A" if "someone" not in u else f"https://page.example/{i}",
                    snippet="",
                    position=i,
                    image_url=u,
                )
                for i, u in enumerate(hits, 1)
            ], []

    async def faces_or_fallback(urls, fallbacks):
        return await faces.faces_in(urls), list(urls)

    faces.faces_or_fallback = faces_or_fallback
    try:
        preview = PreviewSearch(Settings(_env_file=None, llm_api_key=""), Search(), faces)  # type: ignore[arg-type]
        c = ClientCandidate(
            id="x",
            displayName="Nguyen Van A",
            confidence=0.8,
            imageSearchQuery="Nguyen Van A",
            sources=[ClientSource(url="https://en.wikipedia.org/wiki/Nguyen_Van_A", title="Nguyen Van A", type="wikipedia", verified=True)],
            photos=[ClientPhoto(url=own, fromSource=True)],
        )
        await preview._photos(c, [], ("vn", "en"))
    finally:
        for u in extra:
            del FACES[u]
    # The source photo leads and is the avatar; its copy and someone else's face are dropped; 3 photos at most.
    assert [p.url for p in c.photos] == [own, "https://img.example/me1.jpg", "https://img.example/me2.jpg"]
    assert c.avatarUrl == own and c.photos[1].faceMatch == 0.894


def test_same_picture_under_another_url_is_dropped():
    from app.media.faces import distinct_pictures

    crop = 0b1011_0110 << 40
    a = Face((0, 0, 50, 50), 1.0, ME.embedding, crop_hash=crop)
    resized = Face((0, 0, 20, 20), 1.0, ME_B.embedding, crop_hash=crop ^ 0b111)  # 3 bits off: same shot
    other_shot = Face((0, 0, 50, 50), 1.0, ME_C.embedding, crop_hash=~crop & (2**64 - 1))
    another = Face((0, 0, 50, 50), 1.0, face([0.83, 0.558, 0]).embedding, crop_hash=crop ^ ((1 << 14) - 1))
    assert distinct_pictures([[a], [resized], [another], None, [], [other_shot], [face([1, 0.01, 0])]]) == [0, 2, 3, 4, 5]


async def test_no_fetchable_page():
    photos, checked = await finder(Fetcher(), Faces()).find("Nguyen Van A", SOURCES[:1])
    assert photos == [] and not checked
