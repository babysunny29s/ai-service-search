"""Minimal Serper.dev MCP server (stdio or streamable-http).

The API key is read from SERPER_API_KEY in this process only; it is never
returned to the MCP client.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

SERPER_BASE = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")
TIMEOUT = float(os.environ.get("SERPER_TIMEOUT_S", "15"))

mcp = FastMCP("serper-search")


class SerperError(RuntimeError):
    pass


async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise SerperError("SERPER_API_KEY is not configured on the MCP server")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{SERPER_BASE}/{path}",
            json=payload,
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
        )
    if resp.status_code == 401 or resp.status_code == 403:
        raise SerperError(f"serper auth error ({resp.status_code})")
    if resp.status_code == 429:
        raise SerperError("serper rate limited (429)")
    if resp.status_code >= 400:
        raise SerperError(f"serper http {resp.status_code}")
    return resp.json()


@mcp.tool()
async def web_search(query: str, num: int = 10, gl: str = "us", hl: str = "en", page: int = 1) -> dict[str, Any]:
    """Google web search via Serper. Returns organic results (title, link, snippet, date, position)."""
    data = await _post("search", {"q": query, "num": max(1, min(num, 20)), "gl": gl, "hl": hl, "page": page})
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
            "date": r.get("date"),
            "position": r.get("position"),
        }
        for r in data.get("organic", [])
        if r.get("link")
    ]
    kg = data.get("knowledgeGraph")
    return {
        "provider": "serper",
        "query": query,
        "results": results,
        "knowledgeGraph": kg if isinstance(kg, dict) else None,
        "credits": data.get("credits"),
    }


@mcp.tool()
async def image_search(query: str, num: int = 10, gl: str = "us", hl: str = "en") -> dict[str, Any]:
    """Google image search via Serper. Returns image URL and the page where it appears."""
    data = await _post("images", {"q": query, "num": max(1, min(num, 20)), "gl": gl, "hl": hl})
    results = [
        {
            "title": r.get("title", ""),
            "imageUrl": r.get("imageUrl", ""),
            "thumbnailUrl": r.get("thumbnailUrl"),
            "imageWidth": r.get("imageWidth"),
            "imageHeight": r.get("imageHeight"),
            "url": r.get("link", ""),
            "source": r.get("source"),
            "position": r.get("position"),
        }
        for r in data.get("images", [])
        if r.get("imageUrl")
    ]
    return {"provider": "serper", "query": query, "results": results, "credits": data.get("credits")}


@mcp.tool()
async def lens_search(url: str, gl: str = "us", hl: str = "en") -> dict[str, Any]:
    """Google Lens via Serper. `url` must be a public image URL. One call costs 3 credits."""
    data = await _post("lens", {"url": url, "gl": gl, "hl": hl})
    rows = data.get("organic") or data.get("visualMatches") or []
    results = []
    for r in rows if isinstance(rows, list) else []:
        link = r.get("link") or r.get("url") or ""
        if not link:
            continue
        results.append(
            {
                "title": r.get("title") or "",
                "url": link,
                "source": r.get("source"),
                "imageUrl": r.get("imageUrl"),
                "thumbnailUrl": r.get("thumbnailUrl"),
                "position": r.get("position"),
            }
        )
    return {"provider": "serper", "query": url, "results": results, "credits": data.get("credits", 3)}


def main() -> None:
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.settings.host = os.environ.get("SERPER_MCP_HOST", mcp.settings.host)
    mcp.settings.port = int(os.environ.get("SERPER_MCP_PORT", mcp.settings.port))
    mcp.run(transport=transport)  # "stdio" | "streamable-http"


if __name__ == "__main__":
    main()
