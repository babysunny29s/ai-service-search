"""Fake search MCP server with canned results, used by the e2e test."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-search")

BASE = [
    {
        "title": "Nguyễn Văn A - Founder - Công ty X | LinkedIn",
        "url": "https://vn.linkedin.com/in/nguyenvana?trk=public",
        "snippet": "Nguyễn Văn A. Founder tại Công ty X. Fintech. Hồ Chí Minh.",
        "position": 1,
    },
    {
        "title": "Nguyễn Văn A | Founder Công ty X",
        "url": "https://nguyenvana.dev/about?utm_source=google",
        "snippet": "Trang cá nhân của Nguyễn Văn A, nhà sáng lập Công ty X.",
        "position": 2,
    },
    {
        "title": "Nguyễn Văn A - Giảng viên Đại học Z",
        "url": "https://daihocz.edu.vn/giang-vien/nguyen-van-a",
        "snippet": "ThS. Nguyễn Văn A, giảng viên khoa Hóa, Đại học Z.",
        "position": 3,
    },
    {"title": "Trần Thị B", "url": "https://example.com/b", "snippet": "Không liên quan", "position": 4},
]

# Only reachable once "Công ty X" has been discovered and used as a seed.
ENRICHED = {
    "title": "Nguyễn Văn A: hành trình khởi nghiệp fintech",
    "url": "https://hoidoanhnhan.org/hoi-vien/nguyen-van-a",
    "snippet": "Nguyễn Văn A, nhà sáng lập Công ty X, chia sẻ về hành trình khởi nghiệp.",
    "position": 1,
}


RONALDO = [
    {
        "title": "Cristiano Ronaldo - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
        "snippet": "Cristiano Ronaldo is a Portuguese professional footballer who plays for Al-Nassr.",
        "position": 1,
    },
    {
        "title": "Ronaldo (Brazilian footballer) - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Ronaldo_(Brazilian_footballer)",
        "snippet": "Ronaldo Luís Nazário de Lima is a Brazilian former professional footballer.",
        "position": 2,
    },
    {"title": "Ronaldinho - Wikipedia", "url": "https://en.wikipedia.org/wiki/Ronaldinho", "snippet": "Brazilian former footballer.", "position": 3},
]
RONALDO_KG = {
    "title": "Cristiano Ronaldo",
    "type": "Portuguese footballer",
    "description": "Cristiano Ronaldo is a Portuguese professional footballer who plays for Al-Nassr.",
    "descriptionLink": "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
    "imageUrl": "https://upload.wikimedia.org/cr7.jpg",
    "attributes": {"Born": "5 February 1985 (age 41), Funchal, Portugal", "Spouse": "Private Person"},
}


CR7_SOCIALS = [
    {"title": "Cristiano Ronaldo (@cristiano) • Instagram", "url": "https://www.instagram.com/cristiano/", "snippet": "650M Followers, 600 Following, 3,800 Posts - Cristiano Ronaldo official.", "position": 1},
    {"title": "Cristiano Ronaldo fans", "url": "https://facebook.com/groups/123", "snippet": "Cristiano Ronaldo fan group", "position": 2},
    {"title": "Cristiano Ronaldo goal", "url": "https://facebook.com/espn/videos/999", "snippet": "Cristiano Ronaldo scores", "position": 3},
    {"title": "Cristiano Ronaldo (@Cristiano) / X", "url": "https://x.com/Cristiano/with_replies", "snippet": "Cristiano Ronaldo", "position": 4},
    {"title": "Ronaldinho (@ronaldinho) • Instagram", "url": "https://instagram.com/ronaldinho", "snippet": "Ronaldinho", "position": 5},
]


@mcp.tool()
async def web_search(query: str, num: int = 10, gl: str = "vn", hl: str = "vi") -> dict:
    if "Chipu" in query:
        return {"provider": "fake", "query": query, "results": []}
    if query.startswith("Cristiano Ronaldo") and "site:instagram.com" in query:
        return {"provider": "fake", "query": query, "results": CR7_SOCIALS[:num]}
    if "Ronaldo" in query:
        return {"provider": "fake", "query": query, "results": RONALDO[:num], "knowledgeGraph": RONALDO_KG}
    results = ([ENRICHED] if "Công ty X" in query else []) + BASE
    return {"provider": "fake", "query": query, "results": results[:num]}


@mcp.tool()
async def image_search(query: str, num: int = 10, gl: str = "vn", hl: str = "vi") -> dict:
    return {
        "provider": "fake",
        "query": query,
        "results": [{"title": "Nguyễn Văn A - Công ty X", "imageUrl": "https://img.example/a.jpg", "url": "https://othersite.vn/a", "position": 1}],
    }


if __name__ == "__main__":
    mcp.run("stdio")
