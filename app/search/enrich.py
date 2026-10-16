"""Full profile of one candidate, written when the user opens it: biography,
facts and career from the candidate's own sources (one LLM call, no search).
The result is stored on the history row, so reopening costs nothing."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.core.client_format import onesearch_facts
from app.chat.history import HistoryStore
from app.core.llm import EnrichOutput, LLMProvider
from app.core.models import ClientCandidate, ClientCareer, ClientKeyFact, EnrichRequest, EnrichResponse, StepMeta, utcnow

log = logging.getLogger(__name__)

# Bump when the prompt or the output shape changes: stored profiles are rewritten on next open.
ENRICHMENT_VERSION = "2026-10-14.1"
MAX_SOURCES = 12
SNIPPET_CHARS = 300
MAX_FACTS = 18
MAX_CAREER = 10
MAX_PARAGRAPHS = 6
_FACT_KEY = re.compile(r"^[a-z][a-zA-Z0-9]{1,39}$")
_PRIVATE = re.compile(r"address|birth|^dob$|religio|health|politic|spouse|children|idnumber", re.I)
# Synonyms the model uses (onesearch keys, singulars) for keys search already fills.
_KEY_ALIASES = {
    "primaryRole": "jobTitle",
    "role": "jobTitle",
    "institution": "company",
    "almaMater": "education",
    "award": "awards",
    "team": "teams",
    "publication": "notableWorks",
    "publications": "notableWorks",
    "skill": "specialization",
    "skills": "specialization",
    "currentRole": "jobTitle",
    "birthDate": "born",
    "dateOfBirth": "born",
    "birthday": "born",
    "phoneHint": "phone",
    "telephone": "phone",
    "mobile": "phone",
    "emailAddress": "email",
}
_BORN_SOURCES = ("wikipedia", "website")


class EnrichError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def needs_enrichment(c: ClientCandidate) -> bool:
    return not c.enrichedAt or c.enrichmentVersion != ENRICHMENT_VERSION


def evidence_pack(c: ClientCandidate) -> dict[str, Any]:
    known = {f.key: f.value for f in c.keyFacts}
    for key in ("occupation", "nationality", "knownFor", "location"):
        if (v := getattr(c, key)) and key not in known:
            known[key] = v
    return {
        "name": c.displayName,
        "aliases": c.aliases,
        "headline": c.headline,
        "disambiguationNote": c.disambiguationNote,
        "knownFacts": known,
        "sources": [
            {"type": s.type, "url": s.url, "title": s.title, "snippet": (s.snippet or "")[:SNIPPET_CHARS]} for s in c.sources[:MAX_SOURCES]
        ],
        "priorProfile": c.profile[:4],
    }


def _fact_key(raw: str) -> str | None:
    key = raw.strip()
    key = key[:1].lower() + key[1:]
    key = _KEY_ALIASES.get(key, key)
    return key if _FACT_KEY.match(key) and not _PRIVATE.search(key) else None


def _born_supported(c: ClientCandidate, value: str) -> bool:
    """A birth date is kept only when a Wikipedia or official-site snippet states that year."""
    years = re.findall(r"\b(1[89]\d\d|20\d\d)\b", value)
    texts = [f"{s.title} {s.snippet or ''}" for s in c.sources if s.type in _BORN_SOURCES]
    return bool(years) and any(all(y in t for y in years) for t in texts)


def merge_enrichment(c: ClientCandidate, out: EnrichOutput) -> ClientCandidate:
    """Search-verified facts stay as they are; the model only adds missing keys (unverified)."""
    facts = list(c.keyFacts)
    have = {f.key for f in facts}
    for f in out.facts:
        key, value = _fact_key(f.key), " ".join(f.value.split())[:240]
        if not key or not value or key in have or len(facts) >= MAX_FACTS:
            continue
        if key == "born" and not _born_supported(c, value):
            continue
        have.add(key)
        facts.append(ClientKeyFact(key=key, label=" ".join(f.label.split())[:60] or key, value=value, verified=False))

    career = [
        ClientCareer(period=x.period, title=x.title, organization=x.organization)
        for x in out.career
        if (x.title or "").strip() or (x.organization or "").strip()
    ][:MAX_CAREER]
    profile = [" ".join(p.split()) for p in out.profile if p.strip()][:MAX_PARAGRAPHS]
    # The Wikipedia lead section is better than anything written from snippets.
    keep_profile = c.profileSource == "wikipedia" and c.profile
    headline = " ".join((out.headline or "").split())[:160]
    return c.model_copy(
        update={
            "headline": c.headline or headline or None,
            "keyFacts": facts,
            "facts": onesearch_facts(facts, c.sources),
            "career": career or c.career,
            "profile": c.profile if keep_profile or not profile else profile,
            "profileSource": c.profileSource if keep_profile or not profile else "llm",
            "tags": c.tags or [t.strip() for t in out.tags if t.strip()][:6],
            "enrichedAt": utcnow().isoformat(),
            "enrichmentVersion": ENRICHMENT_VERSION,
        }
    )


class ProfileEnricher:
    def __init__(self, llm: LLMProvider | None, history: HistoryStore):
        self.llm = llm
        self.history = history
        # Concurrent opens of the same profile share one LLM call.
        self._inflight: dict[str, asyncio.Task[ClientCandidate]] = {}

    async def run(self, req: EnrichRequest) -> EnrichResponse:
        candidate = await self._load(req)
        if not req.force and not needs_enrichment(candidate):
            return EnrichResponse(candidate=candidate, cached=True, meta=StepMeta())
        if self.llm is None:
            raise EnrichError(503, "LLM provider is not configured")

        key = f"{req.historyId or ''}|{candidate.id}"
        task = self._inflight.get(key)
        joined = task is not None
        if task is None:
            task = asyncio.create_task(self._enrich(candidate, req.historyId))
            self._inflight[key] = task
            task.add_done_callback(lambda _: self._inflight.pop(key, None))
        try:
            out = await asyncio.shield(task)
        except EnrichError:
            raise
        except Exception as e:
            raise EnrichError(502, f"enrichment failed: {str(e)[:200]}") from e
        return EnrichResponse(candidate=out, cached=False, meta=StepMeta(llmCalls=0 if joined else 1))

    async def _load(self, req: EnrichRequest) -> ClientCandidate:
        if req.historyId:
            rec = await self.history.get(req.historyId)
            if rec is None:
                raise EnrichError(404, "history not found")
            if rec["userId"] and rec["userId"] != req.userId:
                raise EnrichError(403, "history belongs to another user")
            return ClientCandidate.model_validate(rec["candidate"])
        if req.candidate is None:
            raise EnrichError(422, "send historyId or candidate")
        return req.candidate

    async def _enrich(self, candidate: ClientCandidate, history_id: str | None) -> ClientCandidate:
        assert self.llm is not None
        out = await self.llm.enrich(evidence_pack(candidate))
        merged = merge_enrichment(candidate, out)
        if history_id:
            await self.history.update_candidate(history_id, merged.model_dump(mode="json"))
        log.info("enriched %s facts=%d career=%d paragraphs=%d", merged.id, len(merged.keyFacts), len(merged.career), len(merged.profile))
        return merged
