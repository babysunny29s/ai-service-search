"""MCP Client Manager: connects to configured search MCP servers, discovers
tools, executes tool calls and normalizes results/errors."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)\}")


class MCPToolError(RuntimeError):
    def __init__(self, server: str, tool: str, message: str):
        super().__init__(f"[{server}.{tool}] {message}")
        self.server = server
        self.tool = tool


@dataclass
class ServerConfig:
    name: str
    transport: str
    tools: dict[str, str]
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_s: float = 90


@dataclass
class ConnectedServer:
    config: ServerConfig
    session: ClientSession
    tool_schemas: dict[str, dict[str, Any]]


@dataclass
class ToolCallResult:
    server: str
    tool: str
    data: dict[str, Any]
    latency_ms: int


def _expand(value: str) -> str:
    return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_server_configs(path: str) -> list[ServerConfig]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [ServerConfig(**s) for s in raw.get("servers", []) if s.get("enabled", True)]


class MCPClientManager:
    def __init__(self, configs: list[ServerConfig]):
        self._configs = configs
        self._stack = AsyncExitStack()
        self._servers: dict[str, ConnectedServer] = {}
        self.errors: dict[str, str] = {}

    async def start(self) -> None:
        for cfg in self._configs:
            try:
                await self._connect(cfg)
            except Exception as e:  # keep the service up even if one provider fails
                log.exception("MCP server %s failed to start", cfg.name)
                self.errors[cfg.name] = f"{type(e).__name__}: {e}"

    async def stop(self) -> None:
        await self._stack.aclose()
        self._servers.clear()

    async def _connect(self, cfg: ServerConfig) -> None:
        if cfg.transport == "stdio":
            command = sys.executable if cfg.command in ("python", "python3") else cfg.command
            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
            env.update({k: os.environ.get(k, "") for k in cfg.env})
            params = StdioServerParameters(command=command, args=cfg.args, env=env, cwd=str(PROJECT_ROOT))
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif cfg.transport in ("streamable_http", "http"):
            if not cfg.url:
                raise ValueError("url required for streamable_http transport")
            headers = {k: _expand(v) for k, v in cfg.headers.items()}
            read, write, _ = await self._stack.enter_async_context(streamablehttp_client(_expand(cfg.url), headers=headers))
        else:
            raise ValueError(f"unsupported transport {cfg.transport}")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        schemas = {t.name: t.inputSchema or {} for t in listed.tools}
        missing = [t for t in cfg.tools.values() if t not in schemas]
        if missing:
            log.warning("MCP server %s missing declared tools %s", cfg.name, missing)
        self._servers[cfg.name] = ConnectedServer(cfg, session, schemas)
        log.info("MCP server %s connected, tools=%s", cfg.name, list(schemas))

    def servers_for(self, capability: str) -> list[tuple[str, str]]:
        """(server, tool) pairs that provide a capability such as 'web_search'."""
        out = []
        for name, srv in self._servers.items():
            tool = srv.config.tools.get(capability)
            if tool and tool in srv.tool_schemas:
                out.append((name, tool))
        return out

    def status(self) -> dict[str, Any]:
        return {
            "connected": {n: sorted(s.tool_schemas) for n, s in self._servers.items()},
            "errors": self.errors,
        }

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> ToolCallResult:
        srv = self._servers.get(server)
        if not srv:
            raise MCPToolError(server, tool, "server not connected")
        props = (srv.tool_schemas.get(tool) or {}).get("properties")
        if isinstance(props, dict):
            arguments = {k: v for k, v in arguments.items() if k in props}
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(srv.session.call_tool(tool, arguments), timeout=srv.config.timeout_s)
        except asyncio.TimeoutError as e:
            raise MCPToolError(server, tool, "timeout") from e
        except Exception as e:
            raise MCPToolError(server, tool, f"{type(e).__name__}: {e}") from e
        latency = int((time.perf_counter() - t0) * 1000)

        text_parts = [c.text for c in res.content if getattr(c, "type", None) == "text"]
        if res.isError:
            raise MCPToolError(server, tool, " ".join(text_parts)[:300] or "tool error")

        data: dict[str, Any] | None = None
        structured = getattr(res, "structuredContent", None)
        if isinstance(structured, dict):
            data = structured.get("result", structured) if set(structured) == {"result"} else structured
        if data is None:
            for t in text_parts:
                try:
                    parsed = json.loads(t)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    data = parsed
                    break
        if data is None:
            raise MCPToolError(server, tool, "unparseable tool result")
        return ToolCallResult(server=server, tool=tool, data=data, latency_ms=latency)
