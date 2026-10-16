"""URL normalization, source typing, deduplication and ranking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|msclkid|mc_cid|mc_eid|igshid|ref|ref_src|trk|trkInfo|originalSubdomain|_hsenc|_hsmkt|si|feature|locale|hl|lang)$", re.I)

SOCIAL_HOSTS = {
    "linkedin.com": "linkedin",
    "github.com": "github",
    "facebook.com": "facebook",
    "x.com": "x",
    "twitter.com": "x",
    "instagram.com": "instagram",
    "youtube.com": "youtube",
    "tiktok.com": "tiktok",
    "scholar.google.com": "scholar",
    "researchgate.net": "researchgate",
    "medium.com": "medium",
    "pinterest.com": "pinterest",
    "reddit.com": "reddit",
    "threads.net": "threads",
    "threads.com": "threads",
}
NEWS_HINTS = ("vnexpress", "tuoitre", "thanhnien", "dantri", "vietnamnet", "cafef", "cafebiz", "zingnews", "znews", "vneconomy", "baomoi", "nld.com.vn", "laodong", "tienphong", "forbes", "reuters", "bloomberg", "techcrunch", "e27", "techinasia", "news", "bao")
ACADEMIC_HINTS = (".edu", "edu.vn", "ac.", "orcid.org", "scholar.google", "researchgate", "semanticscholar", "dblp", "arxiv")
DIRECTORY_HINTS = ("crunchbase", "rocketreach", "zoominfo", "signalhire", "apollo.io", "theorg.com", "masothue", "infodoanhnghiep")
BLOCKED_FETCH_HOSTS = ("linkedin.com", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com", "pinterest.com", "reddit.com", "threads.net", "threads.com")

SOURCE_QUALITY = {
    "official_website": 0.9,
    "news": 0.8,
    "academic": 0.85,
    "wiki": 0.75,
    "social_profile": 0.7,
    "code": 0.7,
    "directory": 0.45,
    "other": 0.5,
}


def host_of(url: str) -> str:
    h = urlsplit(url).hostname or ""
    return h[4:] if h.startswith("www.") else h


def _match_host(host: str, base: str) -> bool:
    return host == base or host.endswith("." + base)


_REDIRECT_HOSTS = {
    "translate.google.com",
    "translate.googleusercontent.com",
    "l.instagram.com",
    "lm.instagram.com",
    "l.facebook.com",
    "lm.facebook.com",
    "l.threads.net",
}


def _unwrap(url: str, depth: int = 0) -> str:
    """Google Translate and Instagram/Facebook `?u=` wrappers point at the real page."""
    if depth > 3:
        return url
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in _REDIRECT_HOSTS:
        return url
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    target = query.get("u") or query.get("url") or ""
    if not target.startswith(("http://", "https://")):
        target = unquote(target)
    if target.startswith(("http://", "https://")):
        return _unwrap(target, depth + 1)
    return url


def normalize_url(url: str) -> str:
    parts = urlsplit(_unwrap(url).strip())
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") and any(_match_host(host[2:], b) for b in SOCIAL_HOSTS):
        host = host[2:]
    if re.match(r"^[a-z]{2}\.linkedin\.com$", host):
        host = "linkedin.com"
    if host == "twitter.com":
        host = "x.com"
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not TRACKING_PARAMS.match(k)])
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, query, ""))


def social_platform(url: str) -> str | None:
    host = host_of(url)
    for base, platform in SOCIAL_HOSTS.items():
        if _match_host(host, base):
            return platform
    return None


def is_profile_url(url: str) -> bool:
    """Heuristic: URL points to a person's profile page, not a post/search."""
    platform = social_platform(url)
    path = urlsplit(url).path.strip("/")
    segs = [s for s in path.split("/") if s]
    if platform == "linkedin":
        return len(segs) == 2 and segs[0] == "in"
    if platform == "github":
        return len(segs) == 1 and segs[0] not in {"topics", "search", "orgs", "explore", "features"}
    if platform == "reddit":
        return len(segs) == 2 and segs[0] in ("user", "u")
    if platform == "pinterest":
        return len(segs) == 1 and segs[0] not in {"pin", "search", "ideas", "today", "explore"}
    if platform in ("x", "instagram", "tiktok", "medium"):
        return len(segs) == 1 and segs[0] not in {"search", "explore", "hashtag", "tag", "i", "home"}
    if platform == "facebook":
        if segs and segs[0] in ("people", "p"):
            return len(segs) == (3 if segs[0] == "people" else 2)
        return len(segs) == 1 and segs[0] not in {"groups", "pages", "watch", "events", "search"} or path.startswith("profile.php")
    if platform == "youtube":
        return (len(segs) == 1 and segs[0].startswith("@")) or (len(segs) == 2 and segs[0] in {"c", "channel", "user"})
    if platform == "scholar":
        return "citations" in path
    if platform == "researchgate":
        return len(segs) >= 2 and segs[0] == "profile"
    return False


_PROFILE_TABS = {
    "x": {"with_replies", "media", "likes", "highlights"},
    "instagram": {"reels", "tagged", "reel"},
    "facebook": {"photos", "videos", "about", "posts", "reels"},
    "youtube": {"videos", "featured", "shorts", "streams", "playlists", "about", "community"},
    "tiktok": set(),
}


def profile_root(url: str) -> str | None:
    """Profile URL for `url`: itself if it already is one, or the owner's profile
    when it is a tab of that profile (x.com/user/with_replies -> x.com/user).
    Individual posts are not mapped, since they may belong to someone else."""
    if is_profile_url(url):
        return url
    platform = social_platform(url)
    parts = urlsplit(url)
    segs = [s for s in parts.path.split("/") if s]
    if platform in _PROFILE_TABS and len(segs) == 2 and segs[1] in _PROFILE_TABS[platform]:
        root = urlunsplit((parts.scheme, parts.netloc, "/" + segs[0], "", ""))
        return root if is_profile_url(root) else None
    return None


def link_key(url: str) -> str:
    """Comparison key: the same profile (any of its tabs) or the same page gets
    the same key, ignoring scheme, `www.`, case and percent-encoding."""
    u = normalize_url(url)
    if social_platform(u):
        u = profile_root(u) or u
    parts = urlsplit(u)
    return (parts.netloc + unquote(parts.path).rstrip("/")).lower().replace(" ", "_")


_FAN_HANDLE = re.compile(r"fan|news|daily|update|fc$|^fc|club|lover|army|team_|_team|tribute|fake|parody", re.I)
_FAN_TITLE = re.compile(r"\b(fan ?page|fans?|fan club|fanclub|news|updates?|parody|tribute)\b", re.I)


def looks_like_fan_account(url: str, title: str = "") -> bool:
    """Fan/news/parody accounts about a person rather than the person's own."""
    segs = [s for s in urlsplit(url).path.split("/") if s]
    handle = (segs[-1] if segs and segs[0] in ("in", "c", "channel", "user") else segs[0] if segs else "").lstrip("@")
    return bool(_FAN_HANDLE.search(handle) or _FAN_TITLE.search(title))


_IDENTITY_STOPWORDS = {"professional", "former", "known", "public", "figure", "person", "official", "people", "member", "national"}


def identity_words(*texts: str | None) -> list[str]:
    """Distinctive words (>= 5 letters) describing who a person is: role, field, organization."""
    out: list[str] = []
    for t in texts:
        for w in re.findall(r"\w+", fold(t or "")):
            if len(w) >= 5 and w not in _IDENTITY_STOPWORDS and w not in out:
                out.append(w)
    return out


def mentions_identity(text: str, words: list[str]) -> bool:
    """`text` describes the person: shares a 5-letter stem with an identity word (music / musician)."""
    tokens = [w for w in re.findall(r"\w+", fold(text)) if len(w) >= 5]
    return any(t[:5] == w[:5] for w in words for t in tokens)


def handle_matches_name(url: str, *names: str) -> bool:
    """The account handle carries part of the person's name (x.com/Cristiano,
    instagram.com/leomessi). Opaque IDs (YouTube channel IDs, profile.php) pass."""
    segs = [s for s in urlsplit(url).path.split("/") if s]
    if not segs or segs[0] in ("channel", "profile.php"):
        return True
    if segs[0] in ("people", "p") and len(segs) >= 2:
        raw = segs[1]
    else:
        raw = segs[-1] if segs[0] in ("in", "c", "user", "u") else segs[0]
    handle = re.sub(r"\W|_", "", fold(unquote(raw)))
    parts = {t for n in names for t in fold(n).split() if len(t) >= 4}
    parts |= {re.sub(r"\W", "", fold(n)) for n in names if len(re.sub(r"\W", "", fold(n))) >= 4}
    return any(p in handle for p in parts)


# A surname on its own is not enough to claim an account.
_SURNAMES = {"nguyen", "tran", "pham", "hoang", "huynh", "phan", "dang", "bui", "ngo", "duong", "truong", "dinh", "trinh", "doan", "luu", "luong", "trieu", "phung", "vuong"}


def opaque_profile(url: str) -> bool:
    """A profile with no readable username: YouTube /channel/UC… or Facebook profile.php."""
    profile = profile_of(url)
    if not profile:
        return False
    segs = [s for s in urlsplit(profile.url).path.split("/") if s]
    return bool(segs) and segs[0] in ("channel", "profile.php")


def certain_profile(url: str, *names: str) -> bool:
    """The handle is this person, not someone who shares one name token.

    One shared piece such as "nguyen" or "minh" is not enough, and an opaque
    id is not a match. The whole name has to sit in the handle, or every
    name token of 4+ letters does, or the handle is exactly one uncommon token
    (x.com/Cristiano)."""
    if opaque_profile(url):
        return False
    raw = account_handle(url)
    if not raw:
        return False
    handle = re.sub(r"\W|_", "", fold(raw))
    if not handle:
        return False
    for name in names:
        if not name or not name.strip():
            continue
        folded = re.sub(r"\W", "", fold(name))
        tokens = [t for t in fold(name).split() if len(t) >= 4]
        whole_name = len(folded) >= 4 and folded in handle and (len(tokens) >= 2 or (len(folded) >= 5 and folded not in _SURNAMES))
        every_token = len(tokens) >= 2 and all(t in handle for t in tokens)
        exact = handle in {t for t in fold(name).split() if len(t) >= 5 and t not in _SURNAMES}
        if whole_name or every_token or exact:
            return True
    return False


def account_handle(url: str) -> str | None:
    """"@handle" of a profile URL (instagram.com/leomessi -> @leomessi); None for opaque IDs."""
    segs = [unquote(s) for s in urlsplit(url).path.split("/") if s]
    if not segs or segs[0] in ("channel", "profile.php"):
        return None
    if segs[0] in ("people", "p", "in", "c", "user", "u") and len(segs) >= 2:
        raw = segs[1]
    else:
        raw = segs[0]
    raw = raw.lstrip("@")
    return f"@{raw}" if re.fullmatch(r"[\w.\-]{2,60}", raw) else None


def classify_source(url: str) -> str:
    host = host_of(url)
    platform = social_platform(url)
    if platform == "github":
        return "code"
    if platform in ("scholar", "researchgate"):
        return "academic"
    if platform:
        return "social_profile"
    if "wikipedia.org" in host:
        return "wiki"
    if any(h in host for h in ACADEMIC_HINTS):
        return "academic"
    if any(h in host for h in DIRECTORY_HINTS):
        return "directory"
    if any(h in host for h in NEWS_HINTS):
        return "news"
    return "other"


_ARTICLE_SECTION = re.compile(
    r"/(news|tin-?tuc|tin|article|articles|bai-?viet|blog|blogs|posts?|story|stories|press|press-release|newsroom|"
    r"thoi-su|kinh-doanh|giai-tri|cong-nghe|the-thao|doi-song|xa-hoi|phap-luat|van-hoa|du-lich|suc-khoe|"
    r"chuyen-muc|category|tag|tags|topic|su-kien)(/|$)",
    re.I,
)
_ARTICLE_DATE = re.compile(r"/(19|20)\d\d/\d{1,2}(/\d{1,2})?/")
_ARTICLE_ID = re.compile(r"(-|_|post|p|a|ar|news|bv)\d{5,}(\.\w{2,5})?$", re.I)
_PAGE_EXT = re.compile(r"\.(html?|shtml|chn|aspx?|php)$", re.I)


def looks_like_article(url: str) -> bool:
    """An article/news page on any site, judged from the URL path: news
    sections, dates, numeric article ids, or a long sentence-like slug.
    Profiles ("/name/nm123", "/talent-roster/chi-pu"), social networks,
    Wikipedia and home pages are not articles."""
    if social_platform(url) or classify_source(url) in ("wiki", "code"):
        return False
    path = unquote(urlsplit(url).path).rstrip("/")
    if not path:
        return False
    last = path.rsplit("/", 1)[-1]
    words = [w for w in re.split(r"[-_+]", _PAGE_EXT.sub("", last)) if w]
    return bool(
        _ARTICLE_SECTION.search(path)
        or _ARTICLE_DATE.search(path + "/")
        or _ARTICLE_ID.search(last)
        or len(words) >= 6
        or (_PAGE_EXT.search(last) and len(words) >= 4)
    )


def is_news(url: str) -> bool:
    """News sites and article pages are not used as sources."""
    return classify_source(url) == "news" or looks_like_article(url)


def can_fetch(url: str) -> bool:
    host = host_of(url)
    return not any(_match_host(host, b) for b in BLOCKED_FETCH_HOSTS)


def fold(text: str) -> str:
    """Lowercase and strip Vietnamese diacritics for fuzzy matching."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).lower().split())


def spacing_variant(name: str, text: str) -> bool:
    """`name` appears in `text` written with different spacing or punctuation
    ("Chipu" / "Chi Pu", "Sơn Tùng MTP" / "Sơn Tùng M-TP")."""
    target = re.sub(r"\W", "", fold(name))
    if len(target) < 4:
        return False
    toks = re.findall(r"\w+", fold(text))
    for i in range(len(toks)):
        acc = ""
        for tok in toks[i : i + 6]:
            acc += tok
            if acc == target:
                return True
            if len(acc) >= len(target):
                break
    return False


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Names are short, so a 2-row DP is enough."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    if abs(len(a) - len(b)) > 8:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def tokens_close(a: str, b: str) -> bool:
    """Same word, or a small typo / extra letter (Yuxua ≈ Yuxuan). Short tokens stay exact."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    limit = 2 if min(len(a), len(b)) >= 5 else 1
    return abs(len(a) - len(b)) <= limit and edit_distance(a, b) <= limit


def _token_in_text(tok: str, text: str) -> bool:
    if re.search(rf"(?<!\w){re.escape(tok)}(?!\w)", text):
        return True
    if len(tok) < 5:
        return False
    return any(tokens_close(tok, w) for w in re.findall(r"\w+", text) if len(w) >= 5)


def name_match(name: str, text: str) -> float:
    """1.0 exact (diacritic-insensitive) phrase match, partial for token overlap."""
    n, t = fold(name), fold(text)
    if not n or not t:
        return 0.0
    if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", t) or spacing_variant(n, t):
        return 1.0
    tokens = n.split()
    present = sum(1 for tok in tokens if _token_in_text(tok, t))
    # Vietnamese names: given name (last token) is the most discriminative.
    if tokens and _token_in_text(tokens[-1], t):
        return 0.4 + 0.4 * present / len(tokens)
    return 0.3 * present / len(tokens)


def _tok_sim(a: str, b: str) -> float:
    if tokens_close(a, b):
        return 1.0
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 0.8
    if min(len(a), len(b)) < 4:
        return 0.0
    dist = edit_distance(a, b)
    return max(0.0, 1 - dist / max(len(a), len(b))) if dist <= 3 else 0.0


def name_score(query: str, name: str) -> float:
    """How closely `name` is the person we typed. Extra/missing words and small
    typos are OK; the given name matters most. Used to keep near-matches and
    to rank them (Đồng Nam Khánh → Nam Khánh before Bùi Nam Khánh)."""
    q, n = name_tokens(query), name_tokens(name)
    if not q or not n:
        return 0.0
    if fold(query) == fold(name):
        return 1.0
    q_hits = [max(_tok_sim(qt, nt) for nt in n) for qt in q]
    n_hits = [max(_tok_sim(nt, qt) for qt in q) for nt in n]
    q_cov = sum(q_hits) / len(q)
    n_cov = sum(n_hits) / len(n)
    given = _tok_sim(q[-1], n[-1])
    given_any = max(_tok_sim(q[-1], nt) for nt in n)
    compact_q, compact_n = "".join(q), "".join(n)
    char = 1 - edit_distance(compact_q, compact_n) / max(len(compact_q), len(compact_n), 1)
    if given_any < 0.8 and given < 0.8:
        return 0.25 * max(q_cov, n_cov)
    return min(1.0, 0.3 * max(q_cov, n_cov) + 0.35 * given_any + 0.2 * given + 0.15 * max(char, 0.0))


def near_name(name: str) -> str:
    """The part we still search when the full typed name is missing: last two words."""
    toks = [t for t in name.split() if t]
    return " ".join(toks[-2:]) if len(toks) >= 3 else name


# ---------- shared with the onesearch client (same rules as its web preview) ----------

# How platforms and sources are listed to clients.
SOCIAL_ORDER = ("facebook", "instagram", "linkedin", "github", "x", "tiktok", "threads", "pinterest", "reddit", "youtube")

# Networks with public, indexable profile pages.
PROFILE_SITES = (
    "(site:linkedin.com/in OR site:facebook.com OR site:instagram.com OR site:x.com OR site:tiktok.com "
    "OR site:github.com OR site:pinterest.com OR site:reddit.com/user)"
)


def name_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", fold(text)) if t]


def name_from_title(title: str, query: str) -> str | None:
    """The title segment naming the person, when it shares enough words with the query."""
    q = list(dict.fromkeys(name_tokens(query)))
    if not q:
        return None
    needed = min(2, len(q))
    best: tuple[str, int] | None = None
    for raw in re.split(r"\s+[-|–—·•/]\s+|[()]|\s+on\s+", title, flags=re.I):
        text = " ".join(raw.split())
        if not text or "@" in text:
            continue
        seg = name_tokens(text)
        if not seg or len(seg) > 6:
            continue
        score = sum(1 for t in q if any(tokens_close(t, s) for s in seg))
        if score >= needed and (best is None or score > best[1]):
            best = (text, score)
    return best[0] if best else None


def _client_host(url: str) -> str | None:
    try:
        h = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None
    return re.sub(r"^(www|m|mobile)\.", "", h) or None


def platform_of_url(url: str) -> str:
    """Platform of any URL (profile or not): a social network, "wikipedia" or "web"."""
    h = _client_host(url)
    if not h:
        return "web"
    on = lambda d: h == d or h.endswith("." + d)  # noqa: E731
    if on("linkedin.com"):
        return "linkedin"
    if on("facebook.com") or h == "fb.com":
        return "facebook"
    if on("instagram.com"):
        return "instagram"
    if on("threads.net") or on("threads.com"):
        return "threads"
    if on("x.com") or on("twitter.com"):
        return "x"
    if on("tiktok.com"):
        return "tiktok"
    if on("youtube.com") or h == "youtu.be":
        return "youtube"
    if on("github.com"):
        return "github"
    if re.search(r"(^|\.)pinterest\.[a-z.]+$", h):
        return "pinterest"
    if on("reddit.com"):
        return "reddit"
    if on("wikipedia.org"):
        return "wikipedia"
    return "web"


@dataclass(frozen=True)
class ProfileRef:
    platform: str
    handle: str  # lower-cased; ids/paths keep a prefix ("id/123", "channel/uc…")
    url: str  # canonical profile page: https, no tab, no query string


_EXCLUDED = {
    "facebook": {"groups", "events", "pages", "watch", "hashtag", "public", "search", "login", "share", "sharer", "sharer.php", "photo.php", "story.php", "permalink.php", "reel", "marketplace", "gaming", "help", "policies", "photo", "photos", "video", "videos", "posts"},
    "instagram": {"p", "reel", "reels", "explore", "stories", "accounts", "tv", "direct"},
    "x": {"search", "hashtag", "i", "home", "explore", "intent", "share", "settings", "login"},
    "github": {"topics", "orgs", "features", "marketplace", "collections", "sponsors", "about", "pricing", "search", "login", "settings", "trending", "explore", "enterprise"},
    "pinterest": {"pin", "search", "ideas", "today", "explore", "business", "login", "categories", "topics", "_"},
    "youtube": {"watch", "results", "shorts", "playlist", "feed", "embed", "live", "hashtag"},
}
_CLIENT_TABS = {
    "facebook": {"about", "about_work_and_education", "about_places"},
    "instagram": {"reels", "tagged"},
    "x": {"with_replies", "media"},
    "tiktok": set(),
    "github": set(),
    "pinterest": {"_created", "_saved", "_profile"},
    "youtube": {"videos", "featured", "about", "shorts", "streams", "playlists", "community"},
}
_YOUTUBE_SYSTEM = _EXCLUDED["youtube"] | {"channel", "c", "user", "gaming", "music", "premium", "account", "t", "about", "kids", "@"}


def profile_of(url: str) -> ProfileRef | None:
    """Person-profile URL -> platform, handle and canonical URL. Posts, videos,
    tweets, pins, repos and boards are not profiles."""
    url = _unwrap(url)
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = _client_host(url) or ""
    raw = [unquote(s) for s in parts.path.split("/") if s]
    segs = [s.lower() for s in raw]
    if not segs:
        return None
    first = segs[0]

    def at_root(platform: str, depth: int = 1) -> bool:
        return len(segs) == depth or (len(segs) == depth + 1 and segs[depth] in _CLIENT_TABS.get(platform, set()))

    def make(platform: str, handle: str, canonical: str) -> ProfileRef | None:
        return None if handle in _EXCLUDED.get(platform, set()) else ProfileRef(platform, handle, canonical)

    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        return make("linkedin", segs[1], f"https://www.linkedin.com/in/{raw[1]}/") if first == "in" and len(segs) > 1 else None
    if host in ("facebook.com", "fb.com"):
        if first == "people" and len(segs) == 3:
            return make("facebook", f"people/{segs[1]}/{segs[2]}", f"https://www.facebook.com/people/{raw[1]}/{raw[2]}/")
        if first == "groups" and len(segs) > 3 and segs[2] == "user":
            return make("facebook", f"id/{segs[3]}", f"https://www.facebook.com/profile.php?id={quote(raw[3], safe='')}")
        if first == "p" and len(segs) == 2:
            return make("facebook", f"p/{segs[1]}", f"https://www.facebook.com/p/{raw[1]}/")
        if first == "profile.php":
            pid = dict(parse_qsl(parts.query)).get("id")
            return make("facebook", f"id/{pid}", f"https://www.facebook.com/profile.php?id={quote(pid, safe='')}") if pid else None
        return make("facebook", first, f"https://www.facebook.com/{raw[0]}") if at_root("facebook") else None
    if host == "instagram.com":
        return make("instagram", first, f"https://www.instagram.com/{raw[0]}/") if at_root("instagram") else None
    if host in ("threads.net", "threads.com"):
        handle = first[1:] if first.startswith("@") else first
        return make("threads", handle, f"https://www.threads.net/@{raw[0].lstrip('@')}") if handle and at_root("threads") else None
    if host in ("x.com", "twitter.com"):
        return make("x", first, f"https://x.com/{raw[0]}") if at_root("x") else None
    if host == "github.com":
        return make("github", first, f"https://github.com/{raw[0]}") if at_root("github") else None
    if host == "tiktok.com":
        ok = first.startswith("@") and len(first) > 1 and at_root("tiktok")
        return make("tiktok", first[1:], f"https://www.tiktok.com/{raw[0]}") if ok else None
    if re.search(r"(^|\.)pinterest\.[a-z.]+$", host):
        return make("pinterest", first, f"https://www.pinterest.com/{raw[0]}/") if at_root("pinterest") else None
    if host == "reddit.com" or host.endswith(".reddit.com"):
        ok = first in ("user", "u") and len(segs) == 2
        return make("reddit", segs[1], f"https://www.reddit.com/user/{raw[1]}/") if ok else None
    if host == "youtube.com":
        if first.startswith("@") and len(first) > 1 and at_root("youtube"):
            return make("youtube", first, f"https://www.youtube.com/{raw[0]}")
        if first in ("channel", "c", "user") and len(segs) > 1 and at_root("youtube", 2):
            return make("youtube", f"{first}/{segs[1]}", f"https://www.youtube.com/{first}/{raw[1]}")
        if first not in _YOUTUBE_SYSTEM and at_root("youtube"):
            return make("youtube", first, f"https://www.youtube.com/{raw[0]}")
    return None


def _source_urls(sources: list) -> list[str]:
    out: list[str] = []
    for s in sources:
        url = s if isinstance(s, str) else getattr(s, "url", "") or ""
        if url.startswith("http"):
            out.append(url)
    return out


def _image_site_clause(url: str) -> str | None:
    """Handle + site: so Google Images looks on this profile, not every namesake."""
    account = profile_of(url)
    if account:
        handle = account.handle.lstrip("@")
        if account.platform == "facebook":
            if handle.startswith("id/"):
                return f"{handle[3:]} site:facebook.com"
            return f"{handle} site:facebook.com"
        if account.platform == "instagram":
            return f"{handle} site:instagram.com"
        if account.platform == "tiktok":
            return f"@{handle} site:tiktok.com"
        if account.platform == "linkedin":
            return f"{handle} site:linkedin.com/in"
        if account.platform == "x":
            return f"{handle} site:x.com"
        if account.platform == "threads":
            return f"@{handle} site:threads.net"
        if account.platform == "youtube":
            return f"{handle} site:youtube.com"
        if account.platform == "github":
            return f"{handle} site:github.com"
        if account.platform == "pinterest":
            return f"{handle} site:pinterest.com"
        if account.platform == "reddit":
            return f"{handle} site:reddit.com/user"
    if platform_of_url(url) == "wikipedia":
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        path = parts.path or ""
        if host.endswith(".wikipedia.org") and path.startswith("/wiki/"):
            return f"site:{host}{path}"
    if platform_of_url(url) == "web":
        host = host_of(url)
        if host and "." in host:
            return f"site:{host}"
    return None


def source_image_query(name: str, sources: list) -> str | None:
    """Serper image query pinned to this person's pages. None when there is no profile to pin to."""
    clauses: list[str] = []
    wiki: str | None = None
    for url in _source_urls(sources):
        clause = _image_site_clause(url)
        if not clause:
            continue
        if platform_of_url(url) == "wikipedia":
            wiki = wiki or clause
            continue
        clauses.append(clause)
    picked = ([wiki] if wiki else []) + list(dict.fromkeys(clauses))
    picked = [c for c in picked if c][:3]
    if not picked:
        return None
    name = " ".join((name or "").split())
    group = picked[0] if len(picked) == 1 else f"({' OR '.join(picked)})"
    return " ".join(p for p in (name, group) if p)[:200]


def image_on_sources(page_url: str, image_url: str | None, sources: list, title: str = "") -> bool:
    """The Serper hit was found on this person's own page, not a namesake's."""
    if not page_url:
        return False
    page_key = link_key(page_url)
    page_prof = profile_of(page_url)
    page_host = host_of(page_url)
    img_host = host_of(image_url or "")
    blob = fold(f"{page_url} {title}")
    have_wiki = False
    for url in _source_urls(sources):
        if page_key == link_key(url):
            return True
        prof = profile_of(url)
        if page_prof and prof and link_key(page_prof.url) == link_key(prof.url):
            return True
        if prof:
            handle = prof.handle.lstrip("@")
            token = handle.split("/")[-1]
            same_net = platform_of_url(page_url) == prof.platform
            if same_net and token and len(token) >= 4 and fold(token) in fold(page_url):
                return True
            if same_net and token and len(token) >= 6 and fold(token) in blob:
                return True
        if platform_of_url(url) == "wikipedia":
            have_wiki = True
        elif not prof and platform_of_url(url) == "web":
            host = host_of(url)
            if host and (page_host == host or page_host.endswith("." + host)):
                return True
    if have_wiki and (
        page_host.endswith("wikipedia.org")
        or page_host.endswith("wikimedia.org")
        or img_host.endswith("wikimedia.org")
    ):
        return True
    return False


_OFFICIAL_TITLE = re.compile(r"official\s+(web\s*)?site|trang\s+(web\s+)?chính\s+thức", re.I)


def official_website_of(url: str, title: str, name: str, knowledge_graph: bool = False) -> str | None:
    """A person's own homepage root: Google's knowledge-graph website, or a site
    root whose domain carries the name ("taylorswift.com") or whose title says
    "Official Website". Sub-pages never qualify (except the knowledge-graph link)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname or platform_of_url(url) != "web":
        return None
    root = f"{parts.scheme}://{parts.hostname}/"
    if knowledge_graph:
        return root
    if parts.path.rstrip("/"):
        return None
    host = re.sub(r"[^a-z0-9]", "", fold(re.sub(r"^www\.", "", parts.hostname)))
    folded = "".join(name_tokens(name))
    if len(folded) >= 6 and folded in host:
        return root
    if _OFFICIAL_TITLE.search(title) and name_from_title(title, name):
        return root
    return None


# Social CDNs refuse hotlinking (Facebook "lookaside" answers with an HTML page).
_UNSTABLE_IMAGE_HOST = re.compile(
    r"(^|\.)(fbsbx\.com|fbcdn\.net|facebook\.com|cdninstagram\.com|instagram\.com|licdn\.com|tiktokcdn\.com|tiktokcdn-us\.com|tiktok\.com)$", re.I
)


def stable_image_url(image_url: str | None, thumbnail_url: str | None = None) -> str | None:
    """The original image, or Google's thumbnail when the original sits on a social CDN."""
    host = ""
    if image_url:
        try:
            host = urlsplit(image_url).hostname or ""
        except ValueError:
            host = ""
    if image_url and host and not _UNSTABLE_IMAGE_HOST.search(host):
        return image_url
    return thumbnail_url if thumbnail_url and thumbnail_url.startswith("https://") else None


_IMAGE_FILE = re.compile(r"\.(?:jpe?g|png|webp|gif)(?:\?|$)", re.I)
_IMAGE_HOST = re.compile(
    r"(^|\.)(cdninstagram\.com|fbcdn\.net|fbsbx\.com|scontent[^.]*\.(?:cdninstagram\.com|fbcdn\.net)|licdn\.com|tiktokcdn(?:-us)?\.com)$",
    re.I,
)


def looks_like_image_url(url: str) -> bool:
    """A direct image (file suffix or a social/CDN photo host), to embed rather than list as a page."""
    if not url.startswith("http"):
        return False
    if _IMAGE_FILE.search(url):
        return True
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return bool(_IMAGE_HOST.search(host) or re.search(r"profile.pic|profile_pic|displayphoto|lookaside", url, re.I))


_IMAGE_SIZE_PARAMS = {"w", "h", "width", "height", "s", "size", "crop", "scale", "fit", "dpr", "q_auto", "quality", "format", "resize", "rt", "cb", "v", "get_thumbnail"}


def image_key(url: str) -> str:
    """Same picture at another size gets the same key (Wikimedia serves one file as
    `/thumb/<file>/330px-...` and `/1280px-...`; news CDNs add `?w=800&h=420`).
    Identifying query params are kept: Google thumbnails (`/images?q=tbn:...`) and
    Facebook media (`?media_id=...`) differ only there."""
    parts = urlsplit(url)
    path = unquote(parts.path)
    if m := re.search(r"/thumb/(.+?)/\d+px-[^/]*$", path):
        path = m.group(1)
    ident = sorted(
        (k.lower(), v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _IMAGE_SIZE_PARAMS and not k.lower().startswith("utm_")
    )
    key = (parts.netloc + path).lower()
    return key + "?" + urlencode(ident) if ident else key
