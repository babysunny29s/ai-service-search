"""Search-engine locale (gl, hl), local to the person searched for so local
sources rank first. It only shapes the search; everything returned to the
client is in OUTPUT_LANGUAGE."""

from __future__ import annotations

from app.core.urls import fold

# (keywords matched against the folded filter value) -> (gl, hl)
_LOCALES: list[tuple[tuple[str, ...], tuple[str, str]]] = [
    (("viet", "ha noi", "hanoi", "ho chi minh", "hcm", "sai gon", "saigon", "da nang"), ("vn", "vi")),
    (("united states", "usa", "u.s", "america", "new york", "california", "san francisco", "los angeles"), ("us", "en")),
    (("united kingdom", "uk", "england", "british", "london", "scotland"), ("uk", "en")),
    (("singapore",), ("sg", "en")),
    (("australia", "sydney", "melbourne"), ("au", "en")),
    (("canada", "toronto", "vancouver"), ("ca", "en")),
    (("india", "mumbai", "delhi", "bangalore"), ("in", "en")),
    (("japan", "tokyo", "osaka"), ("jp", "ja")),
    (("korea", "seoul"), ("kr", "ko")),
    (("china", "chinese", "beijing", "shanghai"), ("cn", "zh-cn")),
    (("taiwan", "taipei"), ("tw", "zh-tw")),
    (("thai", "bangkok"), ("th", "th")),
    (("indonesia", "jakarta"), ("id", "id")),
    (("malaysia", "kuala lumpur"), ("my", "en")),
    (("philippines", "filipino", "manila"), ("ph", "en")),
    (("france", "french", "paris"), ("fr", "fr")),
    (("germany", "german", "berlin", "munich"), ("de", "de")),
    (("spain", "spanish", "madrid", "barcelona"), ("es", "es")),
    (("portugal", "portuguese", "lisbon"), ("pt", "pt")),
    (("brazil", "brasil"), ("br", "pt")),
    (("argentin",), ("ar", "es")),
    (("ital", "rome", "milan"), ("it", "it")),
]

# Letters used by Vietnamese and no other Latin-script language.
_VIETNAMESE = set("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")


def locale_for(filters: dict[str, str], default: tuple[str, str], name: str = "") -> tuple[str, str]:
    """A location/nationality filter decides; otherwise a Vietnamese-spelled
    name searches Vietnam; otherwise the default."""
    for key in ("location", "nationality"):
        value = fold(filters.get(key) or "")
        if not value:
            continue
        words = f" {value} "
        for keywords, locale in _LOCALES:
            if any((f" {k} " in words) if len(k) <= 3 else (k in value) for k in keywords):
                return locale
    if _VIETNAMESE & set(name.lower()):
        return ("vn", "vi")
    return default
