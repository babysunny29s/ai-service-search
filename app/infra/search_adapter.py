"""Provider-agnostic search on top of MCP tools. Business logic only sees
SearchHit objects; which MCP server/tool serves them is configuration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from app.core import tracing
from app.core.cache import TTLCache
from app.core.cost import extra_vnd, serper_cost_details
from app.infra.mcp_manager import MCPClientManager, MCPToolError
from app.core.models import ToolCallLog
from app.core.urls import stable_image_url

# Smaller images are icons or tiny thumbnails, not usable photos.
MIN_IMAGE_WIDTH = 200


@dataclass
class SearchHit:
    provider: str
    query: str
    title: str
    url: str
    snippet: str
    position: int
    date: str | None = None
    image_url: str | None = None
    # Google's cached thumbnail, for when image_url does not serve an image.
    image_fallback: str | None = None


Kind = Literal["web", "image"]


def result_image(row: dict, kind: Kind) -> tuple[str | None, str | None]:
    """Avatar/thumbnail Serper put on the hit. Web results carry LinkedIn photos
    on imageUrl (media.licdn.com); those stay, with Google's thumb as fallback."""
    raw = row.get("imageUrl") or row.get("image")
    thumb = row.get("thumbnailUrl")
    raw = raw if isinstance(raw, str) and raw.startswith("http") else None
    thumb = thumb if isinstance(thumb, str) and thumb.startswith("https://") else None
    if kind == "image":
        image_url = stable_image_url(raw, thumb)
        fallback = thumb if thumb and thumb != image_url else None
        return image_url, fallback
    if raw:
        return raw, thumb if thumb and thumb != raw else None
    return thumb, None


class SearchAdapter:
    def __init__(self, manager: MCPClientManager, cache_ttl_s: float, gl: str, hl: str, num: int):
        self.manager = manager
        self.cache: TTLCache[tuple[list[SearchHit], dict[str, Any] | None]] = TTLCache(cache_ttl_s)
        self.gl, self.hl, self.num = gl, hl, num

    async def search(
        self, query: str, kind: Kind, logs: list[ToolCallLog], locale: tuple[str, str] | None = None, page: int = 1
    ) -> tuple[list[SearchHit], list[dict]]:
        """Run a query against every server offering the capability.
        Returns hits and any knowledge-graph payloads."""
        capability = "web_search" if kind == "web" else "image_search"
        targets = self.manager.servers_for(capability)
        if not targets:
            raise RuntimeError(f"no MCP server provides {capability}")
        gl, hl = locale or (self.gl, self.hl)
        page = max(1, page)
        with tracing.span(f"search:{kind}", as_type="generation", model="serper", input={"query": query, "gl": gl, "hl": hl, "page": page}) as obs:
            local: list[ToolCallLog] = []
            outs = await asyncio.gather(*(self._one(s, t, query, kind, local, gl, hl, page) for s, t in targets))
            logs.extend(local)
            hits: list[SearchHit] = []
            kgs: list[dict] = []
            for h, kg in outs:
                hits.extend(h)
                if kg:
                    kgs.append(kg)
            billable = sum(1 for c in local if not c.cached and not c.error)
            cached = sum(1 for c in local if c.cached)
            tracing.record_search(billable=billable, cached=cached, query=query)
            usage_details, cost_details = serper_cost_details(billable)
            vnd = extra_vnd(serper_calls=billable)
            obs.update(
                output={
                    "results": [{"title": h.title, "url": h.url, **({"image": h.image_url} if h.image_url else {})} for h in hits],
                    "knowledgeGraph": [kg.get("title") for kg in kgs],
                    "billable": billable,
                    "cached": cached,
                },
                model="serper",
                usage_details=usage_details,
                cost_details=cost_details,
                metadata={"vnd": round(vnd, 2), "query": query},
                level="ERROR" if local and all(c.error for c in local) else "DEFAULT",
            )
            tracing.mark_vnd(obs, vnd)
        return hits, kgs

    def has_lens(self) -> bool:
        return bool(self.manager.servers_for("lens_search"))

    async def lens(self, image_url: str, logs: list[ToolCallLog], locale: tuple[str, str] | None = None) -> list[SearchHit]:
        """Reverse image search (Google Lens): pages showing the picture at `image_url`,
        which must be publicly reachable. Not cached: the URL is a temporary upload."""
        targets = self.manager.servers_for("lens_search")
        if not targets:
            raise RuntimeError("no MCP server provides lens_search")
        gl, hl = locale or (self.gl, self.hl)
        server, tool = targets[0]
        with tracing.span("search:lens", as_type="generation", model="serper", input={"url": image_url, "gl": gl, "hl": hl}) as obs:
            try:
                res = await self.manager.call(server, tool, {"url": image_url, "gl": gl, "hl": hl})
            except MCPToolError as e:
                logs.append(ToolCallLog(server=server, tool=tool, query="lens", latencyMs=0, resultCount=0, error=str(e)[:300]))
                obs.update(level="ERROR", output={"error": str(e)[:300]})
                return []
            provider = str(res.data.get("provider") or server)
            hits: list[SearchHit] = []
            for i, r in enumerate(res.data.get("results") or []):
                url = r.get("url") or r.get("link")
                if not url:
                    continue
                image_url_hit, fallback = result_image(r, "image")
                hits.append(
                    SearchHit(
                        provider=provider,
                        query="lens",
                        title=r.get("title") or "",
                        url=url,
                        snippet=r.get("source") or "",
                        position=int(r.get("position") or i + 1),
                        image_url=image_url_hit,
                        image_fallback=fallback,
                    )
                )
            logs.append(ToolCallLog(server=server, tool=tool, query="lens", latencyMs=res.latency_ms, resultCount=len(hits)))
            billable = int(res.data.get("credits") or 3)
            tracing.record_search(billable=billable, query="lens")
            usage_details, cost_details = serper_cost_details(billable)
            vnd = extra_vnd(serper_calls=billable)
            obs.update(
                output={"results": [{"title": h.title, "url": h.url} for h in hits], "billable": billable},
                model="serper",
                usage_details=usage_details,
                cost_details=cost_details,
                metadata={"vnd": round(vnd, 2), "query": "lens"},
            )
            tracing.mark_vnd(obs, vnd)
        return hits

    async def _one(self, server: str, tool: str, query: str, kind: Kind, logs: list[ToolCallLog], gl: str, hl: str, page: int = 1):
        key = (server, tool, query, gl, hl, self.num, page)
        cached = self.cache.get(key)
        if cached is not None:
            logs.append(ToolCallLog(server=server, tool=tool, query=query, latencyMs=0, resultCount=len(cached[0]), cached=True))
            return cached
        args: dict = {"query": query, "num": self.num, "gl": gl, "hl": hl}
        if page > 1:
            args["page"] = page
        try:
            res = await self.manager.call(server, tool, args)
        except MCPToolError as e:
            logs.append(ToolCallLog(server=server, tool=tool, query=query, latencyMs=0, resultCount=0, error=str(e)[:300]))
            return [], None
        provider = str(res.data.get("provider") or server)
        hits: list[SearchHit] = []
        for i, r in enumerate(res.data.get("results") or []):
            url = r.get("url") or r.get("link")
            if not url:
                continue
            image_url, image_fallback = result_image(r, kind)
            if kind == "image":
                width = r.get("imageWidth")
                if not image_url or (isinstance(width, (int, float)) and 0 < width < MIN_IMAGE_WIDTH):
                    continue
            hits.append(
                SearchHit(
                    provider=provider,
                    query=query,
                    title=r.get("title") or "",
                    url=url,
                    snippet=r.get("snippet") or "",
                    position=int(r.get("position") or i + 1),
                    date=r.get("date"),
                    image_url=image_url,
                    image_fallback=image_fallback,
                )
            )
        kg = res.data.get("knowledgeGraph")
        out = (hits, kg if isinstance(kg, dict) else None)
        self.cache.set(key, out)
        logs.append(ToolCallLog(server=server, tool=tool, query=query, latencyMs=res.latency_ms, resultCount=len(hits)))
        return out
