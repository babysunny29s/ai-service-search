"""Optional Langfuse tracing. Enabled only when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are set (LANGFUSE_BASE_URL selects cloud or self-hosted);
otherwise every helper is a no-op and the plain OpenAI client is used.

Per-request counters (LLM tokens, Serper, Scrape Creators) accumulate on the
outermost `span()` even when Langfuse is off, so API `meta` can show them."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.core.cost import extra_vnd, llm_from_settings, usd_from_vnd

log = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def openai_client_class() -> type:
    if enabled():
        from langfuse.openai import AsyncOpenAI
    else:
        from openai import AsyncOpenAI
    return AsyncOpenAI


def generation_kwargs(name: str, **metadata: Any) -> dict[str, Any]:
    """Extra `chat.completions.create` arguments understood only by the Langfuse wrapper."""
    return {"name": name, "metadata": metadata} if enabled() else {}


@dataclass
class CostLine:
    model: str
    kind: str
    token_in: int
    token_out: int
    usd: float
    vnd: float
    detail: str = ""

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "model": self.model,
            "kind": self.kind,
            "tokenIn": self.token_in,
            "tokenOut": self.token_out,
            "totalTokens": self.token_in + self.token_out,
            "usd": round(self.usd, 6),
            "vnd": round(self.vnd, 2),
            "detail": self.detail,
        }


@dataclass
class TraceStats:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    serper_calls: int = 0
    serper_cached: int = 0
    scrape_calls: int = 0
    scrape_credits: int = 0
    scrape_cached: int = 0
    llm_credits: float = 0.0
    llm_input_usd: float = 0.0
    llm_output_usd: float = 0.0
    llm_vnd: float = 0.0
    extra_vnd: float = 0.0
    lines: list[CostLine] = field(default_factory=list)

    def cost_usd(self) -> float:
        return self.llm_credits + usd_from_vnd(self.extra_vnd)

    def cost_vnd(self) -> float:
        return self.llm_vnd + self.extra_vnd

    def cost_details(self) -> dict[str, float]:
        extra = usd_from_vnd(self.extra_vnd)
        return {
            "input": self.llm_input_usd,
            "output": self.llm_output_usd,
            "total": self.llm_credits + extra,
        }

    def as_dict(self) -> dict[str, int | float]:
        return {
            "llmCalls": self.llm_calls,
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "totalTokens": self.total_tokens,
            "serperCalls": self.serper_calls,
            "serperCached": self.serper_cached,
            "scrapeCreatorsCalls": self.scrape_calls,
            "scrapeCreatorsCredits": self.scrape_credits,
            "scrapeCreatorsCached": self.scrape_cached,
        }


_stats: ContextVar[TraceStats | None] = ContextVar("trace_stats", default=None)


def current_stats() -> TraceStats | None:
    return _stats.get()


# Yescale chat.completions always sends OpenAI-style prompt_tokens / completion_tokens.
# It also copies unused Claude aliases input_tokens=0, output_tokens=0 — ignore those
# unless the OpenAI keys are actually missing.
_PROMPT_KEYS = ("prompt_tokens", "promptTokenCount", "prompt_token_count")
_COMPLETION_KEYS = ("completion_tokens", "candidatesTokenCount", "candidates_token_count")
_TOTAL_KEYS = ("total_tokens", "totalTokenCount", "total_token_count")
_PROMPT_FALLBACK = ("input_tokens",)
_COMPLETION_FALLBACK = ("output_tokens",)


def extract_usage(payload: Any) -> dict[str, Any]:
    """Token object from a Yescale/OpenAI completion or a stream chunk.

    Non-stream responses put `usage` on the JSON body. Stream responses put the
    same object on the last SSE chunk (empty `choices`) when
    `stream_options.include_usage` is set.
    """
    if payload is None:
        return {}
    dumped = _as_dict(payload)
    usage = dumped.get("usage")
    if usage is None:
        usage = getattr(payload, "usage", None)
    usage_d = _as_dict(usage)
    extra = dumped.get("usage_metadata") or dumped.get("usageMetadata")
    extra_d = extra if isinstance(extra, dict) else _as_dict(extra)
    if extra_d:
        usage_d = {**extra_d, **usage_d}
    return usage_d


def parse_usage(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    usage = _flatten_usage(usage)
    prompt = _first_int(usage, *_PROMPT_KEYS)
    completion = _first_int(usage, *_COMPLETION_KEYS)
    if prompt is None:
        prompt = _first_int(usage, *_PROMPT_FALLBACK)
    if completion is None:
        completion = _first_int(usage, *_COMPLETION_FALLBACK)
    prompt = prompt or 0
    completion = completion or 0
    total = _first_int(usage, *_TOTAL_KEYS)
    if total is None:
        total = prompt + completion
    return prompt, completion, total


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in data or data[key] is None:
            continue
        try:
            return int(data[key])
        except (TypeError, ValueError):
            continue
    return None


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        try:
            dumped = obj.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    data: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
        "usage",
        "usage_metadata",
        "usageMetadata",
        "prompt_tokens_details",
        "completion_tokens_details",
    ):
        if hasattr(obj, key):
            value = getattr(obj, key)
            if value is not None:
                data[key] = value
    return data


def _flatten_usage(usage: Any) -> dict[str, Any]:
    usage = _as_dict(usage)
    nested = usage.get("usage_metadata") or usage.get("usageMetadata") or usage.get("usage")
    if isinstance(nested, dict):
        merged = dict(usage)
        for k, v in nested.items():
            merged.setdefault(k, v)
        return merged
    return usage


def record_llm(usage: dict[str, Any] | None, *, model: str | None = None) -> tuple[int, int, int]:
    prompt, completion, total = parse_usage(usage)
    billed = llm_from_settings(prompt, completion)
    st = _stats.get()
    if st is not None:
        st.llm_calls += 1
        st.prompt_tokens += prompt
        st.completion_tokens += completion
        st.total_tokens += total
        st.llm_credits += billed.credits
        st.llm_input_usd += billed.input_usd
        st.llm_output_usd += billed.output_usd
        st.llm_vnd += billed.vnd
        from app.config import get_settings

        st.lines.append(
            CostLine(
                model=model or get_settings().llm_model,
                kind="llm",
                token_in=prompt,
                token_out=completion,
                usd=billed.credits,
                vnd=billed.vnd,
            )
        )
    return prompt, completion, total


def record_search(*, billable: int, cached: int = 0, query: str = "") -> None:
    st = _stats.get()
    if st is None:
        return
    st.serper_calls += billable
    st.serper_cached += cached
    if not billable:
        return
    vnd = extra_vnd(serper_calls=billable)
    st.extra_vnd += vnd
    st.lines.append(
        CostLine(model="serper", kind="serper", token_in=billable, token_out=0, usd=usd_from_vnd(vnd), vnd=vnd, detail=query)
    )


def record_scrape(*, calls: int = 0, credits: int = 0, cached: bool = False, detail: str = "") -> None:
    st = _stats.get()
    if st is None:
        return
    if cached:
        st.scrape_cached += 1
        return
    st.scrape_calls += calls
    st.scrape_credits += credits
    if not calls:
        return
    vnd = extra_vnd(scrape_credits=credits)
    st.extra_vnd += vnd
    st.lines.append(
        CostLine(
            model="scrapecreators",
            kind="scrapecreators",
            token_in=credits,
            token_out=0,
            usd=usd_from_vnd(vnd),
            vnd=vnd,
            detail=detail,
        )
    )


def client_cost(st: TraceStats | None) -> dict[str, Any]:
    if st is None:
        return {}
    return {
        "promptTokens": st.prompt_tokens,
        "completionTokens": st.completion_tokens,
        "costUsd": round(st.cost_usd(), 6),
        "costVnd": round(st.cost_vnd(), 2),
        "costLines": [line.as_dict() for line in st.lines],
    }


def _usage_and_cost(st: TraceStats) -> dict[str, Any]:
    """Only for generation observations. Parent spans ignore cost_details."""
    return {
        "usage_details": {"input": st.prompt_tokens, "output": st.completion_tokens, "total": st.total_tokens},
        "cost_details": st.cost_details(),
    }


class _Noop:
    def update(self, **_: Any) -> None:
        pass

    def end(self, **_: Any) -> None:
        pass

    def score(self, **_: Any) -> None:
        pass

    def score_trace(self, **_: Any) -> None:
        pass


def mark_vnd(obs: Any, vnd: float, *, on_trace: bool = False) -> None:
    """Langfuse Cost is USD only. Numeric score `vnd` is the VND column on that row."""
    value = round(float(vnd or 0), 2)
    try:
        obs.update(metadata={"vnd": value})
    except Exception:
        pass
    score = getattr(obs, "score", None)
    if callable(score):
        try:
            score(name="vnd", value=value, comment="VND")
        except Exception as e:
            log.warning("langfuse vnd score failed: %s", e)
    if on_trace:
        score_trace = getattr(obs, "score_trace", None)
        if callable(score_trace):
            try:
                score_trace(name="vnd", value=value, comment="VND")
            except Exception as e:
                log.warning("langfuse vnd trace score failed: %s", e)


def _reset_stats(token: Token | None) -> None:
    if token is None:
        return
    try:
        _stats.reset(token)
    except ValueError:
        try:
            _stats.set(None)
        except Exception:
            pass


def begin(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> tuple[Any, Token | None]:
    """Start an observation that is safe across async-generator yields.

    Unlike `span()`, this does not attach OpenTelemetry current context, so
    `aclose()` / `GeneratorExit` cannot fail with 'Token created in a different Context'.
    """
    created = _stats.get() is None
    token = _stats.set(TraceStats()) if created else None
    if not enabled():
        return _Noop(), token
    from langfuse import get_client

    meta = dict(metadata or {})
    if session_id:
        meta["sessionId"] = session_id
    if tags:
        meta["tags"] = tags
    obs = get_client().start_observation(name=name, as_type=as_type, input=input, metadata=meta)  # type: ignore[arg-type]
    return obs, token


def finish(obs: Any, token: Token | None = None, metadata: dict[str, Any] | None = None) -> None:
    try:
        st = _stats.get()
        extra = dict(metadata or {})
        if token is not None and st is not None:
            extra.update(st.as_dict())
            extra["vnd"] = round(st.cost_vnd(), 2)
            obs.update(metadata=extra, **_usage_and_cost(st))
            mark_vnd(obs, st.cost_vnd(), on_trace=True)
        elif extra:
            obs.update(metadata=extra)
        obs.end()
    except Exception as e:
        log.warning("langfuse finish failed: %s", e)
    _reset_stats(token)
    flush()


@contextmanager
def span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    model: str | None = None,
) -> Iterator[Any]:
    """A Langfuse observation around a block; LLM calls made inside it become
    its children. `session_id`/`tags` set on the outermost span apply to the trace."""
    created = _stats.get() is None
    token: Token | None = None
    if created:
        token = _stats.set(TraceStats())
    try:
        if not enabled():
            yield _Noop()
            return
        from langfuse import get_client, propagate_attributes

        extra = {"model": model} if model else {}
        with get_client().start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata, **extra) as obs:  # type: ignore[arg-type, misc]
            try:
                if session_id or tags:
                    with propagate_attributes(session_id=session_id, tags=tags, trace_name=name):
                        yield obs
                else:
                    yield obs
            finally:
                st = _stats.get()
                if created and st is not None:
                    # Parent is a span: Langfuse Cost only sums generation USD.
                    obs.update(metadata={**(metadata or {}), **st.as_dict(), "vnd": round(st.cost_vnd(), 2)})
                    mark_vnd(obs, st.cost_vnd(), on_trace=True)
    finally:
        _reset_stats(token)


def flush() -> None:
    if not enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        log.warning("langfuse flush failed: %s", e)
