"""Shared service bootstrap for the HTTP API and the RabbitMQ worker."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from app.chat.chat import ChatService
from app.chat.history import HistoryStore
from app.core import tracing
from app.core.jobs import JobStore
from app.core.llm import LLMError, build_llm
from app.infra.fetcher import PageFetcher
from app.infra.mcp_manager import MCPClientManager, load_server_configs
from app.infra.search_adapter import SearchAdapter
from app.infra.search_store import MemorySearchStore, PostgresSearchStore
from app.infra.social_crawl import SocialCrawl
from app.media.faces import FaceVerifier
from app.media.photos import PhotoService
from app.media.source_photos import SourcePhotoFinder
from app.search.deep_search import DeepSearch
from app.search.enrich import ProfileEnricher
from app.search.identify import IdentifyService, ImagePeopleSearch
from app.search.orchestrator import ProfileSearchOrchestrator
from app.search.preview import PreviewSearch
from app.search.search_profile import SearchProfile
from app.search.sources import SourceLookup
from .config import get_settings

log = logging.getLogger("profile-search")

_STATE_ATTRS = (
    "settings",
    "mcp",
    "llm",
    "jobs",
    "orch",
    "source_photos",
    "preview",
    "profile_search",
    "deep_search",
    "photos",
    "history",
    "chat",
    "sources",
    "identify",
    "image_search",
    "enricher",
    "search_store",
)


def load_dotenv() -> None:
    """Expose .env to os.environ so MCP subprocesses can receive their keys."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Runtime:
    def bind(self, state: Any) -> None:
        for name in _STATE_ATTRS:
            setattr(state, name, getattr(self, name))

    async def close(self) -> None:
        self.history.close()
        store = getattr(self, "search_store", None)
        if store is not None:
            await store.close()
        await self.jobs.shutdown()
        await self.mcp.stop()
        tracing.flush()


async def start_runtime() -> Runtime:
    load_dotenv()
    settings = get_settings()
    manager = MCPClientManager(load_server_configs(settings.mcp_config_path))
    await manager.start()
    try:
        llm = build_llm(settings)
    except LLMError as e:
        log.warning("LLM disabled: %s", e)
        llm = None
    search = SearchAdapter(manager, settings.search_cache_ttl_s, settings.search_locale_gl, settings.search_locale_hl, settings.results_per_query)
    fetcher = PageFetcher(settings.fetch_timeout_s, settings.fetch_max_bytes, settings.search_cache_ttl_s, settings.fetch_user_agent)
    rt = Runtime()
    rt.settings = settings
    rt.mcp = manager
    rt.llm = llm
    rt.jobs = JobStore()
    faces = None
    if settings.face_verify:
        model_dir = Path(settings.face_model_dir)
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parent.parent / model_dir
        faces = FaceVerifier(str(model_dir), fetcher.user_agent, settings.fetch_timeout_s, settings.search_cache_ttl_s)
        asyncio.create_task(faces.ready())
    rt.orch = ProfileSearchOrchestrator(settings, search, fetcher, llm, faces)
    rt.source_photos = SourcePhotoFinder(settings, fetcher, faces)
    rt.preview = PreviewSearch(settings, search, faces, rt.source_photos)
    rt.profile_search = SearchProfile(settings, search, llm, rt.source_photos, fetcher)
    crawl = SocialCrawl(settings.crawlfeed_api_key, settings.crawlfeed_base_url, settings.crawlfeed_timeout_s, settings.search_cache_ttl_s)
    rt.sources = SourceLookup(settings, search)
    rt.deep_search = DeepSearch(settings, search, llm, rt.source_photos, faces, fetcher, crawl, rt.sources)
    rt.photos = PhotoService(settings, search, faces)
    db_path = Path(settings.history_db_path)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent.parent / db_path
    rt.history = HistoryStore(str(db_path))
    rt.chat = ChatService(settings, rt.history, search, llm, fetcher)
    if settings.db_host and settings.db_name:
        try:
            rt.search_store = await PostgresSearchStore.connect(settings)
        except Exception:
            log.exception("Postgres search store failed; using in-memory fallback")
            rt.search_store = MemorySearchStore()
    else:
        log.warning("DB_HOST unset; search_id / candidate_id persist in memory only")
        rt.search_store = MemorySearchStore()
    rt.identify = IdentifyService(llm, rt.sources, settings, rt.source_photos)
    rt.image_search = ImagePeopleSearch(settings, rt.identify, rt.orch, faces, search, fetcher, llm)
    rt.enricher = ProfileEnricher(llm, rt.history)
    return rt
