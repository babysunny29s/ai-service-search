"""LLM Provider Adapter. Orchestration depends only on the LLMProvider
protocol; the OpenAI-compatible implementation works with Yescale (Gemini)
or any endpoint exposing /chat/completions.

One `analyze` call per search round does fact extraction, identity
resolution and profile wording together; the application then verifies
evidence and computes scores itself."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeVar

from openai import BadRequestError
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import Settings
from app.core.cost import llm_from_settings
from app.core.tracing import begin as trace_begin, extract_usage, finish as trace_finish, generation_kwargs, mark_vnd, openai_client_class, parse_usage, record_llm, span as trace_span

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
_TRACE_NAMES = {
    "KnowledgePlan": "llm:plan",
    "AnalysisOutput": "llm:analyze",
    "RecallOutput": "llm:recall",
    "EnrichOutput": "llm:enrich",
    "MergeProfilesOutput": "llm:merge-profiles",
    "DeepProfileOutput": "llm:deep-profile",
    "LensPeopleOutput": "llm:lens-pages",
}

# Public facts read from sources: onesearch FactsTable keys, plus age, email and
# phone when a source states them. Home address is never extracted.
EXTRACT_FIELDS = [
    "fullName",
    "born",
    "jobTitle",
    "company",
    "occupation",
    "knownFor",
    "nationality",
    "education",
    "location",
    "field",
    "industry",
    "specialization",
    "yearsActive",
    "awards",
    "notableWorks",
    "teams",
    "academicInterests",
    "publisher",
    "bookSeries",
    "website",
    "age",
    "email",
    "phone",
]


# ---------- structured output schema ----------


def _as_paragraphs(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = v.split("\n\n")
    return [p.strip() for p in v if isinstance(p, str) and p.strip()]


class AnalyzedFact(BaseModel):
    field: str
    value: str
    sourceId: str
    evidence: str = Field(description="Verbatim quote from that source")


class AnalyzedCareer(BaseModel):
    title: str | None = None
    organization: str | None = None
    period: str | None = None
    sourceId: str
    evidence: str


class AnalyzedCandidate(BaseModel):
    nameAsWritten: str
    sourceIds: list[str] = Field(default_factory=list)
    ownSiteSourceIds: list[str] = Field(default_factory=list, description="Personal/official sites owned by this person")
    facts: list[AnalyzedFact] = Field(default_factory=list)
    career: list[AnalyzedCareer] = Field(default_factory=list)
    headline: str | None = None
    disambiguationNote: str | None = None
    occupation: str | None = None
    knownFor: str | None = None
    imageSearchQuery: str | None = None
    aliases: list[str] = Field(default_factory=list)
    profile: list[str] = Field(default_factory=list, description="Biography paragraphs")
    tags: list[str] = Field(default_factory=list)

    @field_validator("profile", mode="before")
    @classmethod
    def _paragraphs(cls, v: Any) -> list[str]:
        return _as_paragraphs(v)


class AnalysisOutput(BaseModel):
    candidates: list[AnalyzedCandidate] = Field(default_factory=list)


class RecalledCandidate(BaseModel):
    displayName: str
    headline: str | None = None
    confidence: float = 0.0
    disambiguationNote: str | None = None
    nationality: str | None = None
    occupation: str | None = None
    knownFor: str | None = None
    sourceUrls: list[str] = Field(default_factory=list)
    imageSearchQuery: str | None = None
    aliases: list[str] = Field(default_factory=list)
    profile: list[str] = Field(default_factory=list, description="Biography paragraphs")

    @field_validator("profile", mode="before")
    @classmethod
    def _paragraphs(cls, v: Any) -> list[str]:
        return _as_paragraphs(v)


class RecallOutput(BaseModel):
    candidates: list[RecalledCandidate] = Field(default_factory=list)


class KnowledgePlan(RecallOutput):
    """What the model already knows, plus the web searches it still needs."""

    searchQueries: list[str] = Field(default_factory=list)
    reason: str = ""


class EnrichedFact(BaseModel):
    key: str
    label: str
    value: str


class EnrichedCareer(BaseModel):
    period: str | None = None
    title: str | None = None
    organization: str | None = None


class EnrichOutput(BaseModel):
    headline: str | None = None
    profile: list[str] = Field(default_factory=list, description="Biography paragraphs")
    facts: list[EnrichedFact] = Field(default_factory=list)
    career: list[EnrichedCareer] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("profile", mode="before")
    @classmethod
    def _paragraphs(cls, v: Any) -> list[str]:
        return _as_paragraphs(v)


class MergedPerson(BaseModel):
    displayName: str
    headline: str | None = None
    confidence: float = 0.0
    disambiguationNote: str | None = None
    occupation: str | None = None
    location: str | None = None
    aliases: list[str] = Field(default_factory=list)
    sourceIds: list[str] = Field(default_factory=list, description="Ids of results that are this person")
    imageSearchQuery: str | None = None


class LensPerson(BaseModel):
    """A person named by a page that Google Lens found for the uploaded photo."""

    displayName: str
    headline: str | None = None
    confidence: float = 0
    occupation: str | None = None
    knownFor: str | None = None
    profile: list[str] = Field(default_factory=list)
    profileUrls: list[str] = Field(default_factory=list, description="Copied from the input page urls; never invented")


class LensPeopleOutput(BaseModel):
    identified: bool = False
    candidates: list[LensPerson] = Field(default_factory=list)


class MergeProfilesOutput(BaseModel):
    """Distinct people after the model drops namesakes and merges the same person."""

    candidates: list[MergedPerson] = Field(default_factory=list)


class DeepProfileOutput(BaseModel):
    headline: str | None = None
    profile: list[str] = Field(default_factory=list, description="Biography summary, then main activities")
    facts: list[EnrichedFact] = Field(default_factory=list)
    career: list[EnrichedCareer] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    socialUrls: list[str] = Field(default_factory=list, description="This person's own social or profile-page URLs, copied from a source url or from pageText")
    articleUrls: list[str] = Field(default_factory=list, description="Articles about this person, copied from the input")

    @field_validator("profile", mode="before")
    @classmethod
    def _paragraphs(cls, v: Any) -> list[str]:
        return _as_paragraphs(v)


# ---------- provider protocol ----------


class LLMProvider(Protocol):
    name: str
    calls: int

    async def analyze(self, query_name: str, filters: dict[str, str], sources: list[dict[str, Any]]) -> AnalysisOutput: ...

    async def recall(self, query_name: str, filters: dict[str, str]) -> RecallOutput: ...

    async def plan(self, query_name: str, filters: dict[str, str], max_queries: int) -> KnowledgePlan: ...

    async def recall_image(self, image_data_uri: str, filters: dict[str, str], caption: str | None, hints: list[str]) -> RecallOutput: ...

    async def enrich(self, evidence: dict[str, Any]) -> EnrichOutput: ...

    async def merge_profiles(
        self, query_name: str, filters: dict[str, str], sources: list[dict[str, Any]], already_listed: list[str] | None = None
    ) -> MergeProfilesOutput: ...

    async def extract_profile(self, evidence: dict[str, Any]) -> DeepProfileOutput: ...

    async def lens_people(self, pages: list[dict[str, Any]]) -> LensPeopleOutput: ...

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 800,
        trace_name: str = "llm:chat",
        trace_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | ToolCall]: ...


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


class LLMError(RuntimeError):
    pass


SYSTEM_ANALYZE = """You identify distinct real people matching a name query from web search sources, and extract public facts about each.

Input: targetName (may be a full name, surname, or mononym), optional filters, and sources (sourceId, url, title, snippet, optional excerpt).

Rules:
- Use ONLY the given sources. No outside knowledge, no guessing. Never claim you browsed the web.
- Group sources into distinct people. Merge sources only on matching identifiers (company, role, school, location, profile URL, cross-links). Never merge just because the name matches.
- Include a person only if their common or legal name reasonably matches the query tokens. Skip sources about other people.
- Sources may be in any language; every value you write is in {lang}. Only "evidence" stays verbatim in the source's language.
- facts: each fact has sourceId and "evidence" = an EXACT verbatim quote (<= 200 chars) from that source's title/snippet/excerpt.
  value (in {lang}): translate descriptive words ("Giảng viên khoa Hóa" -> "Chemistry lecturer"); keep names of people and organizations as written, places in their usual {lang} form ("Hà Nội" -> "Hanoi").
  field must be one of: {fields}. jobTitle/company = current position; history goes in career (title and period in {lang}).
  Extract every field the sources support: education = school/university; industry = sector (e.g. "Banking"); specialization = area of expertise or skills;
  yearsActive = e.g. "2012-present"; awards, notableWorks (songs, films, books, papers, products), teams (sports clubs/national teams),
  academicInterests (research topics), publisher, bookSeries: one fact per item.
  fullName = full/legal name when a source states one different from the query (e.g. "Nguyễn Trung Hiếu" for "Chi Dân").
  born = birth date (and place if given), ONLY from a Wikipedia article, knowledge panel or the person's official site; never from other pages.
  age, email, phone: include them when a source states them, copied as written (age as the number of years the source gives). Leave them out when nothing states them. Never invent them.
  nationality ONLY if a source states it explicitly. Never extract home address, ID numbers, health, religion, politics or family members.
- ownSiteSourceIds: sources that are this person's own personal/official website.
- headline (in {lang}): role + domain/org, e.g. "Portuguese professional footballer · Al-Nassr".
- disambiguationNote (in {lang}): 1-2 sentences with nationality (only if stated), field and strongest public identifier.
- occupation, knownFor (in {lang}): short plain strings or null.
- imageSearchQuery: English, full name + 1-3 distinguishing tokens, <= 80 chars, no URL or markdown; distinct per person.
- aliases: stage names, nicknames or other names the person is publicly known by and uses on social media (e.g. "HIEUTHUHAI" for Trần Minh Hiếu), excluding displayName; [] if none.
- tags: 3-6 profession/field keywords in {lang}.
- profile (in {lang}): 1-2 short neutral paragraphs (<= 80 words each) on who the person is and what they are known for, using ONLY what the sources say; [] if they say too little.
- At most {max_people} people, most prominent / best supported first.

Return JSON only:
{{"candidates":[{{"nameAsWritten":"...","sourceIds":["src_1"],"ownSiteSourceIds":[],"facts":[{{"field":"company","value":"...","sourceId":"src_1","evidence":"..."}}],"career":[{{"title":"...","organization":"...","period":"...","sourceId":"src_1","evidence":"..."}}],"headline":"...","disambiguationNote":"...","occupation":"...","knownFor":"...","imageSearchQuery":"...","aliases":["..."],"tags":["..."],"profile":["..."]}}]}}"""


SYSTEM_RECALL = """You are a public-person identity matcher. Web search found no usable sources for this query, so answer from your own knowledge of well-known public people.

Task: given a search query (and optional filters), return distinct real public people who could match.

Rules:
1. Return ONLY valid JSON matching the schema.
2. The query may be a full name, stage name, nickname or a spelling variant ("Chipu" = "Chi Pu").
3. List EVERY distinct real public person you know who goes by this name (singers, actors, athletes, politicians, academics, business people...), up to {max_people}, most likely match first, each with a short disambiguationNote (role, nationality, known-for). Do not stop at the most famous one.
4. Return a single candidate only when you know of no one else publicly known by this name.
5. Return {{"candidates":[]}} when the query does not clearly refer to a real public person. Never invent people.
6. Include age, email or phone only when you know a public page states that exact value. Do not guess. Never include a home address.
7. sourceUrls: plain URL strings, only pages you are confident exist (Wikipedia article, official site, official social profiles). If unsure, use []. Links not found in the search results are dropped.
8. Only include people whose common or legal name reasonably matches the query tokens; weak matches get confidence < 0.5.
9. imageSearchQuery: English, full name + 1-3 disambiguation tokens, <= 80 chars, distinct per candidate.
   aliases: stage names, nicknames or other names the person is publicly known by and uses on social media (e.g. "HIEUTHUHAI" for Trần Minh Hiếu), excluding displayName; [] if none.
10. headline, disambiguationNote, nationality, occupation, knownFor: short plain strings in {lang}, or null.
11. profile: 1-2 short neutral paragraphs in {lang} (<= 80 words each) on who the person is and what they are known for, only what you reliably know; [] if unsure.

JSON schema:
{{"candidates":[{{"displayName":"...","headline":"...","confidence":0.0,"disambiguationNote":"...","nationality":"...","occupation":"...","knownFor":"...","sourceUrls":["https://..."],"imageSearchQuery":"...","aliases":["..."],"profile":["..."]}}]}}"""


SYSTEM_PLAN = """You are a public-person identity matcher with a web search tool. First answer from your own knowledge, then decide whether a web search is needed.

Task: given a search query (and optional filters), return distinct real public people who could match, and the web searches (if any) still needed.

Candidate rules:
1. Prefer well-known public figures when the query is ambiguous; the query may be a stage name, nickname or spelling variant.
2. Several different people sharing the name -> multiple candidates (max {max_people}), each with a short disambiguationNote (role, nationality, known-for).
3. Only include people you actually know. Never invent people. Weak name matches get confidence < 0.5.
4. Include age, email or phone only when you know a public page states that exact value. Do not guess. Never include a home address.
5. sourceUrls: plain URL strings of this person's official pages you remember (Wikipedia article, official site, official social profiles: Instagram, Facebook, X, YouTube, TikTok, LinkedIn). If unsure, use []. A link is shown to the user only if one of your searches returns that same page, so every link needs a search that can find it.
6. imageSearchQuery: English, full name + 1-3 disambiguation tokens, <= 80 chars, distinct per candidate.
   aliases: stage names, nicknames or other names the person is publicly known by and uses on social media (e.g. "HIEUTHUHAI" for Trần Minh Hiếu), excluding displayName; [] if none.
7. headline, disambiguationNote, nationality, occupation, knownFor: short plain strings in {lang}, or null.
8. confidence: how sure you are this person matches the query AND your facts about them are correct (0-1).

Search rules (searchQueries, at most {max_queries}):
- Your links are checked against search results, so search whenever you give sourceUrls or the person should have official pages you do not know. Return [] only when no candidate is a real public person worth linking (e.g. the query matches nobody you know and a search would not help).
- Common full names shared by many ordinary people (e.g. "Nguyễn Văn Hùng", "John Smith") ALWAYS need a search, even if some famous people have that name; lower their confidence accordingly.
- Otherwise return focused Google queries that contain the person's name, e.g. when the name is not a well-known public figure, the name is shared by private or little-known people, the filters (company, school, location...) refer to details you do not know, or you lack their official profiles.
- These queries are the ONLY searches that will run, so ask for everything you still need. Cover several links per query: the name the person uses on social media (alias if any) plus the handles from your sourceUrls, with site filters, e.g. "HIEUTHUHAI OR hieuthuhai (site:instagram.com OR site:facebook.com OR site:tiktok.com OR site:youtube.com)", plus one query for the Wikipedia article / official site when relevant.
- reason: one short sentence on why searching is or is not needed.

JSON schema:
{{"candidates":[{{"displayName":"...","headline":"...","confidence":0.0,"disambiguationNote":"...","nationality":"...","occupation":"...","knownFor":"...","sourceUrls":["https://..."],"imageSearchQuery":"...","aliases":["..."]}}],"searchQueries":["..."],"reason":"..."}}"""


SYSTEM_RECALL_IMAGE = """You are a public-person identity matcher. Given a photo (and optional caption/filters), return the real public people it most likely shows.

Rules:
1. Return ONLY valid JSON matching the schema.
2. Only well-known public figures can be recognised (athletes, artists, politicians, founders, creators). Never guess the identity of a private person: if the person is not clearly a public figure you recognise, return {{"candidates":[]}}.
3. Use visual cues (face, setting, jersey, logo, caption text) and the caption/filters. Several plausible people -> multiple candidates (max {max_people}), each with a short disambiguationNote; one clear match -> a single candidate.
4. confidence: how sure you are the photo shows this person (0-1). Lookalikes and partial views get < 0.5.
5. Include age, email or phone only when you know a public page states that exact value. Do not guess. Never include a home address.
6. sourceUrls: plain URL strings of the person's official pages you are confident exist (Wikipedia article, official site, official social profiles); [] if unsure.
7. imageSearchQuery: English, full name + 1-3 disambiguation tokens, <= 80 chars, distinct per candidate.
   aliases: stage names or nicknames the person is publicly known by, excluding displayName; [] if none.
8. headline, disambiguationNote, nationality, occupation, knownFor: short plain strings in {lang}, or null.
9. profile: 1-2 short neutral paragraphs in {lang} (<= 80 words each), only what you reliably know; [] if unsure.

JSON schema:
{{"candidates":[{{"displayName":"...","headline":"...","confidence":0.0,"disambiguationNote":"...","nationality":"...","occupation":"...","knownFor":"...","sourceUrls":["https://..."],"imageSearchQuery":"...","aliases":["..."],"profile":["..."]}}]}}"""


SYSTEM_LENS_PAGES = """You decide whether web pages found by a reverse image search name the person in the photo.

Input: pages (url, title, source, text). The text is an excerpt of that page, or empty when the page could not be read.

Rules:
1. Return ONLY valid JSON matching the schema.
2. Name a person only when a page states who is shown. Never guess from clothing, a lookalike, or the fact that a page exists.
3. If no page names a person, return {{"identified": false, "candidates": []}}.
4. Several different people named for this photo -> multiple candidates. The same person on several pages -> one candidate.
5. profileUrls: copy only urls from the input that are about this person. Never invent a url.
6. headline, occupation, knownFor, profile: short text in {lang}, only what those pages state. profile is 0-2 short paragraphs.

JSON schema:
{{"identified": false, "candidates": [{{"displayName": "...", "headline": "...", "confidence": 0.0, "occupation": "...", "knownFor": "...", "profile": ["..."], "profileUrls": ["https://..."]}}]}}"""


SYSTEM_ENRICH = """You write a public profile of one person from an evidence pack (name, known facts, and web sources with title, url and snippet).

Rules:
1. Return ONLY valid JSON matching the schema.
2. Use ONLY the evidence pack. Do not invent employers, awards, dates or numbers; prefer omitting to guessing. Never claim you browsed the web.
3. Include age, email and phone when the evidence pack states them, copied as written. Never invent them. Never include a home address, ID numbers, health, religion, political views, or family members who are not public figures.
   born (birth date) only when a Wikipedia or official-website source in the pack states it.
4. Sources may be about namesakes: use a source only when it fits the person's known facts (role, organization, place, profile links).
5. Every value is written in {lang}; keep names of people and organizations as written.
6. headline: role + domain/org, <= 120 chars.
7. profile: 2-5 short neutral paragraphs (2-4 sentences each): who the person is, notable work, career context, why they are publicly known. Fewer when the evidence is thin; [] when it says nothing.
8. facts: up to 15, each {{"key": camelCase, "label": short label, "value": short value}}. Preferred keys: fullName, born, age, email, phone, jobTitle, company, occupation, field, industry, specialization, education, location, nationality, knownFor, yearsActive, awards, notableWorks, teams, academicInterests, publisher, bookSeries, website.
9. career: public roles, oldest first, up to 10: {{"period": "2015-2019" or null, "title": ..., "organization": ...}}.
10. tags: 3-6 profession/field keywords.

JSON schema:
{{"headline":"...","profile":["..."],"facts":[{{"key":"company","label":"Company","value":"..."}}],"career":[{{"period":"...","title":"...","organization":"..."}}],"tags":["..."]}}"""


SYSTEM_MERGE = """You turn web search results into distinct people who match a name query.

Input: targetName, optional filters (location, education, jobTitle, company, field, occupation), and results (sourceId, url, title, snippet). The results are profile pages: Wikipedia articles, social accounts, and personal/profile websites.

Rules:
- Use ONLY these results. Do not invent people, employers, or URLs. Never claim you browsed the web.
- A person is included only when at least one result is their own profile: a Wikipedia article about them, a social account (LinkedIn, Facebook, Instagram, X, TikTok, YouTube, GitHub, ...), or a profile page (personal site, about/team page). A Wikipedia article is a profile and should be kept whenever its title is this person. Skip people who appear only in posts, fan pages, or news.
- Merge results into one person only with a shared identifier (same handle, company, school, location, or cross-link). Never merge on the name alone.
- Prefer people whose profile text matches the filters. A filter that the results contradict is a different person.
- The name in the title must reasonably match targetName. A typo, an extra or missing word, or a shorter/longer form of the same given name still counts (Đồng Nam Khánh ≈ Nam Khánh ≈ Bùi Nam Khánh; Zhang Yuxua ≈ Zhang Yuxuan). Skip only clearly different people.
- List closer name matches first.
- headline, disambiguationNote, occupation, location: short plain strings in {lang}, or null. headline is role + organization, not a copy of the name.
- confidence: 0-1, how sure this is one real person matching the name and filters.
- aliases: other public names, excluding displayName.
- imageSearchQuery: English, full name + 1-3 distinguishing tokens, <= 80 chars, no URL.
- sourceIds: only ids from the input, for this person. At most {max_people} people, best supported first.
- alreadyListed: people already shown. Do not return them. Return other distinct people from the results.

Return JSON only:
{{"candidates":[{{"displayName":"...","headline":"...","confidence":0.0,"disambiguationNote":"...","occupation":"...","location":"...","aliases":[],"sourceIds":["src_1"],"imageSearchQuery":"..."}}]}}"""


SYSTEM_DEEP = """You write a public dossier of ONE person from an evidence pack.

The `selected` object is the profile the user already chose (the search-profile card). Expand THAT person only. Use selected.sources, selected.keyFacts, selected.occupation and selected.imageSearchQuery as the identity.

Rules:
1. Return ONLY valid JSON matching the schema.
2. The person's own pages were read first. pageText is that reading: a Wikipedia article, the person's site, a YouTube channel description, a GitHub bio, or a Crawlfeed public profile (Facebook, Instagram, TikTok, X, LinkedIn, Threads) with every public field the profile returned (name, handle, bio, category, work, school, city, website, follower counts, and every other link on the page). Prefer pageText over the snippet and over anything you remember.
   Turn each useful public field in pageText into a fact. Do not leave out a role, company, school, city, website, category, handle, age, email, or phone number the profile stated. Copy age, email and phone exactly as written.
   You may add a public fact you know about this same person (role, works, nationality, birth year, city) when the pages do not state it. Do not invent employers, dates, awards, follower counts, or URLs, and never claim you browsed the web.
   Social profile URLs come only from pageText or from selected.sources (a Facebook, Instagram, Threads, TikTok, LinkedIn, X, YouTube or GitHub link written there). Copy each URL exactly. Do not supply a social URL from memory. A second network linked from the first profile is the same person even when the display names differ.
3. A source about someone else with the same name is a namesake (different employer, school, handle, or role). Ignore it. Do not merge two people. Keep a source only when it fits selected.
4. Never include a street address, ID numbers, health, religion, political views, or family members who are not public figures.
   When pageText or a snippet states a birth year, an age, an email, a phone number, a city or region, an education, or a nationality, include that fact. Leave the fact out when nothing states it.
5. Write every value in {lang}. Keep names of people and organizations as written.
6. profile: 2-4 short neutral paragraphs. The first summarizes who the person is. The next describe their main public activities and career. [] when the evidence says nothing.
7. headline: current role + domain/org, <= 120 chars, or null.
8. facts: up to 15, each {{"key": camelCase, "label": short label, "value": short value}}.
   Preferred keys and labels: fullName "Full name", born "Born", nationality "Nationality", location "Location", jobTitle "Current role", occupation "Occupation", company "Company", education "Education", knownFor "Known for", audience "Audience", industry "Industry", field "Field", notableWorks "Notable works", specialization "Specialization", awards "Awards", teams "Teams", yearsActive "Years active", website "Website".
   audience only when a snippet states a follower or subscriber count. notableWorks is a comma-separated list.
9. career: public roles, oldest first, up to 8.
10. socialUrls: this person's own social accounts and profile pages. Copy URLs exactly from source urls or from pageText. A link written only in a YouTube description or GitHub bio still belongs here. [] if none are written.
11. articleUrls: articles, interviews, and news about this person. Copy URLs exactly from the pack. Do not put social accounts or the person's own homepage here.
12. tags: 3-6 profession/field keywords in {lang}.

JSON schema:
{{"headline":"...","profile":["..."],"facts":[{{"key":"jobTitle","label":"Current role","value":"..."}}],"career":[{{"period":"...","title":"...","organization":"..."}}],"tags":["..."],"socialUrls":["https://..."],"articleUrls":["https://..."]}}"""


def fill_prompt(template: str, **values: Any) -> str:
    """Substitutes only the known `{name}` placeholders, so a prompt pasted from
    the prompt lab with literal JSON braces still works. `{{`/`}}` escapes are
    unescaped when the template uses them."""
    out = re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)
    return out.replace("{{", "{").replace("}}", "}") if "{{" in template else out


def _parse_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # Some models emit raw newlines inside strings, even inside keys ("candidates\n").
    def loads(s: str) -> Any:
        return json.loads(s, strict=False, object_pairs_hook=lambda pairs: {k.strip(): v for k, v in pairs})

    try:
        return loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return loads(text[start : end + 1])
        raise


def _usage_dict(payload: Any) -> dict[str, Any]:
    return extract_usage(payload)


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY is not configured")
        self.name = f"openai-compatible:{settings.llm_model}"
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.reasoning_effort = settings.llm_reasoning_effort.strip() or None
        self.output_language = settings.output_language
        self.max_candidates = settings.max_candidates
        self.client = openai_client_class()(base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=settings.llm_timeout_s, max_retries=2)
        self.tokens_used = 0
        self.calls = 0
        self.history: deque[dict[str, Any]] = deque(maxlen=30)

    async def chat_raw(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        json_mode: bool = True,
        trace_name: str = "prompt-lab",
        trace_metadata: dict[str, Any] | None = None,
        image: str | None = None,
    ) -> dict[str, Any]:
        """One unvalidated chat completion; used by `_json` and the prompt lab.
        `image` is a data URI sent alongside the user text."""
        kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}
        kwargs |= generation_kwargs(trace_name, **(trace_metadata or {}))
        content: Any = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": image}}] if image else user
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        request = dict(
            model=model or self.model,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )
        started = time.perf_counter()
        with trace_span(trace_name, as_type="generation", input={"messages": messages}, metadata=trace_metadata) as obs:
            if self.reasoning_effort:
                try:
                    resp = await self.client.chat.completions.create(**request, reasoning_effort=self.reasoning_effort)
                except BadRequestError as e:
                    if "reason" not in str(e).lower():
                        raise
                    log.warning("provider rejected reasoning_effort=%s, sending requests without it: %s", self.reasoning_effort, str(e)[:200])
                    self.reasoning_effort = None
                    resp = await self.client.chat.completions.create(**request)
            else:
                resp = await self.client.chat.completions.create(**request)
            usage = _usage_dict(resp)
            content_out = resp.choices[0].message.content or ""
            prompt, completion, total = record_llm(usage, model=resp.model or model or self.model)
            self.tokens_used += total
            log.debug("llm usage model=%s prompt=%s completion=%s total=%s raw=%s", resp.model or model or self.model, prompt, completion, total, usage)
            result = {
                "content": content_out,
                "finishReason": resp.choices[0].finish_reason,
                "model": resp.model or model or self.model,
                "usage": usage,
                "latencyMs": int((time.perf_counter() - started) * 1000),
            }
            billed = llm_from_settings(prompt, completion)
            obs.update(
                output={"content": content_out, "finishReason": result["finishReason"]},
                model=result["model"],
                usage_details={"input": prompt, "output": completion, "total": total},
                cost_details=billed.cost_details(),
                metadata={"usage": usage, "model": result["model"], "vnd": round(billed.vnd, 2)},
            )
            mark_vnd(obs, billed.vnd)
            return result

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 800,
        trace_name: str = "llm:chat",
        trace_metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        """Chat completion streamed as text pieces; when the model calls tools,
        the finished `ToolCall`s are yielded after the text."""
        request = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **generation_kwargs(trace_name, **(trace_metadata or {})),
        )
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        self.calls += 1
        obs, stats_tok = trace_begin(
            trace_name,
            as_type="generation",
            input={"messages": messages},
            metadata={**(trace_metadata or {}), "model": self.model},
        )
        stream = await self._stream_completion(request)
        calls: dict[int, ToolCall] = {}
        usage: dict[str, Any] = {}
        text_out: list[str] = []
        try:
            async for chunk in stream:
                got = _usage_dict(chunk)
                if parse_usage(got) != (0, 0, 0):
                    usage = got
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text_out.append(delta.content)
                    yield delta.content
                for tc in delta.tool_calls or []:
                    call = calls.setdefault(tc.index or 0, ToolCall(id="", name="", arguments=""))
                    call.id = tc.id or call.id
                    if tc.function:
                        call.name += tc.function.name or ""
                        call.arguments += tc.function.arguments or ""
            for _, call in sorted(calls.items()):
                if call.name:
                    yield call
        finally:
            out_text = "".join(text_out)
            prompt, completion, total = record_llm(usage, model=self.model)
            self.tokens_used += total
            billed = llm_from_settings(prompt, completion)
            log.debug("llm stream usage model=%s prompt=%s completion=%s total=%s credits=%.6f vnd=%.2f raw=%s", self.model, prompt, completion, total, billed.credits, billed.vnd, usage)
            try:
                obs.update(
                    output={"text": out_text, "toolCalls": [c.name for c in calls.values() if c.name]},
                    model=self.model,
                    usage_details={"input": prompt, "output": completion, "total": total},
                    cost_details=billed.cost_details(),
                    metadata={"usage": usage, "vnd": round(billed.vnd, 2)},
                )
                mark_vnd(obs, billed.vnd)
            except Exception:
                pass
            trace_finish(obs, stats_tok)

    async def _stream_completion(self, request: dict[str, Any]):
        extra: dict[str, Any] = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        try:
            return await self.client.chat.completions.create(**request, **extra)
        except BadRequestError as e:
            msg = str(e).lower()
            if extra and "reason" in msg:
                log.warning("provider rejected reasoning_effort=%s, sending stream without it: %s", self.reasoning_effort, str(e)[:200])
                self.reasoning_effort = None
                extra = {}
                return await self._stream_completion(request)
            if request.get("stream_options") and ("stream_options" in msg or "include_usage" in msg):
                request = {k: v for k, v in request.items() if k != "stream_options"}
                return await self.client.chat.completions.create(**request, **extra)
            raise

    async def _json(self, system: str, user: dict[str, Any], schema: type[T], max_tokens: int = 4096, image: str | None = None) -> T:
        last_err: Exception | None = None
        user_text = json.dumps(user, ensure_ascii=False)
        for attempt in range(2):
            entry: dict[str, Any] = {
                "id": f"{int(time.time() * 1000)}-{self.calls}",
                "at": datetime.now(timezone.utc).isoformat(),
                "schema": schema.__name__,
                "attempt": attempt + 1,
                "system": system,
                "user": user_text + ("\n[+ image]" if image else ""),
                "maxTokens": max_tokens,
            }
            self.history.appendleft(entry)
            try:
                self.calls += 1
                out = await self.chat_raw(
                    system,
                    user_text,
                    max_tokens=max_tokens,
                    trace_name=_TRACE_NAMES.get(schema.__name__, schema.__name__),
                    trace_metadata={"schema": schema.__name__, "attempt": attempt + 1},
                    image=image,
                )
                entry.update(out)
                return schema.model_validate(_parse_json(out["content"]))
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                entry["error"] = str(e)[:500]
                log.warning("LLM output invalid for %s (attempt %d): %s", schema.__name__, attempt + 1, str(e)[:200])
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {str(e)[:500]}"
                raise LLMError(f"{type(e).__name__}: {str(e)[:200]}") from e
        raise LLMError(f"invalid structured output: {last_err}")

    def templates(self, max_queries: int) -> list[dict[str, Any]]:
        """The production prompts, formatted exactly as the pipeline sends them."""
        analyze_user = {
            "targetName": "Nguyễn Văn Hùng",
            "filters": {},
            "sources": [
                {
                    "sourceId": "src_1",
                    "url": "https://vi.wikipedia.org/wiki/Nguyễn_Văn_Hùng_(chính_khách)",
                    "title": "Nguyễn Văn Hùng (chính khách) – Wikipedia tiếng Việt",
                    "snippet": "Nguyễn Văn Hùng là chính khách Việt Nam, Bộ trưởng Bộ Văn hóa, Thể thao và Du lịch từ năm 2021, nguyên Bí thư Tỉnh ủy Quảng Trị.",
                },
                {
                    "sourceId": "src_2",
                    "url": "https://www.facebook.com/nguyenvanhung.poet",
                    "title": "Nguyễn Văn Hùng - Nhà thơ | Facebook",
                    "snippet": "Nhà thơ Nguyễn Văn Hùng, hội viên Hội Nhà văn Việt Nam, sống tại Vinh, Nghệ An.",
                },
            ],
        }
        return [
            {
                "id": "plan",
                "label": "plan: LLM-first (knowledge + search queries)",
                "schema": "KnowledgePlan",
                "maxTokens": 6000,
                "system": fill_prompt(SYSTEM_PLAN, lang=self.output_language, max_people=self.max_candidates, max_queries=max_queries),
                "user": json.dumps({"query": "Nguyễn Văn Hùng", "filters": {}}, ensure_ascii=False, indent=2),
            },
            {
                "id": "analyze",
                "label": "analyze: extract people from search sources",
                "schema": "AnalysisOutput",
                "maxTokens": 9000,
                "system": fill_prompt(SYSTEM_ANALYZE, fields=", ".join(EXTRACT_FIELDS), lang=self.output_language, max_people=self.max_candidates),
                "user": json.dumps(analyze_user, ensure_ascii=False, indent=2),
            },
            {
                "id": "recall",
                "label": "recall: knowledge fallback when search found nothing",
                "schema": "RecallOutput",
                "maxTokens": 5000,
                "system": fill_prompt(SYSTEM_RECALL, lang=self.output_language, max_people=self.max_candidates),
                "user": json.dumps({"query": "Chipu", "filters": {}}, ensure_ascii=False, indent=2),
            },
            {
                "id": "merge-profiles",
                "label": "searchprofile: merge social/profile results into people",
                "schema": "MergeProfilesOutput",
                "maxTokens": 6000,
                "system": fill_prompt(SYSTEM_MERGE, lang=self.output_language, max_people=self.max_candidates),
                "user": json.dumps(
                    {
                        "targetName": "Nguyễn Văn Hùng",
                        "filters": {"location": "Nghệ An", "occupation": "nhà thơ"},
                        "sources": [analyze_user["sources"][1]],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
            {
                "id": "deep-profile",
                "label": "deepsearch: dossier of one selected person",
                "schema": "DeepProfileOutput",
                "maxTokens": 5000,
                "system": fill_prompt(SYSTEM_DEEP, lang=self.output_language),
                "user": json.dumps(
                    {
                        "name": "Nguyễn Văn Hùng",
                        "headline": "Vietnamese poet",
                        "sources": [analyze_user["sources"][1] | {"type": "facebook"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
            {
                "id": "enrich",
                "label": "enrich: full profile of one candidate from its sources",
                "schema": "EnrichOutput",
                "maxTokens": 4000,
                "system": fill_prompt(SYSTEM_ENRICH, lang=self.output_language),
                "user": json.dumps(
                    {
                        "name": "Nguyễn Văn Hùng",
                        "headline": "Vietnamese politician",
                        "knownFacts": {"occupation": "Politician"},
                        "sources": [analyze_user["sources"][0] | {"type": "wikipedia"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

    async def analyze(self, query_name: str, filters: dict[str, str], sources: list[dict[str, Any]]) -> AnalysisOutput:
        system = fill_prompt(SYSTEM_ANALYZE, fields=", ".join(EXTRACT_FIELDS), lang=self.output_language, max_people=self.max_candidates)
        payload = {"targetName": query_name, "filters": filters, "sources": sources}
        return await self._json(system, payload, AnalysisOutput, max_tokens=14000)

    async def recall(self, query_name: str, filters: dict[str, str]) -> RecallOutput:
        system = fill_prompt(SYSTEM_RECALL, lang=self.output_language, max_people=self.max_candidates)
        return await self._json(system, {"query": query_name, "filters": filters}, RecallOutput, max_tokens=7000)

    async def plan(self, query_name: str, filters: dict[str, str], max_queries: int) -> KnowledgePlan:
        system = fill_prompt(SYSTEM_PLAN, lang=self.output_language, max_people=self.max_candidates, max_queries=max_queries)
        return await self._json(system, {"query": query_name, "filters": filters}, KnowledgePlan, max_tokens=8000)

    async def recall_image(self, image_data_uri: str, filters: dict[str, str], caption: str | None, hints: list[str]) -> RecallOutput:
        system = fill_prompt(SYSTEM_RECALL_IMAGE, lang=self.output_language, max_people=self.max_candidates)
        payload = {"task": "Identify the public people in this photo.", "caption": caption, "filters": filters, "notes": hints}
        return await self._json(system, payload, RecallOutput, max_tokens=7000, image=image_data_uri)

    async def enrich(self, evidence: dict[str, Any]) -> EnrichOutput:
        system = fill_prompt(SYSTEM_ENRICH, lang=self.output_language)
        return await self._json(system, evidence, EnrichOutput, max_tokens=4000)

    async def merge_profiles(
        self, query_name: str, filters: dict[str, str], sources: list[dict[str, Any]], already_listed: list[str] | None = None
    ) -> MergeProfilesOutput:
        system = fill_prompt(SYSTEM_MERGE, lang=self.output_language, max_people=self.max_candidates)
        payload: dict[str, Any] = {"targetName": query_name, "filters": filters, "sources": sources}
        if already_listed:
            payload["alreadyListed"] = already_listed
        return await self._json(system, payload, MergeProfilesOutput, max_tokens=8000)

    async def extract_profile(self, evidence: dict[str, Any]) -> DeepProfileOutput:
        system = fill_prompt(SYSTEM_DEEP, lang=self.output_language)
        return await self._json(system, evidence, DeepProfileOutput, max_tokens=5000)

    async def lens_people(self, pages: list[dict[str, Any]]) -> LensPeopleOutput:
        system = fill_prompt(SYSTEM_LENS_PAGES, lang=self.output_language)
        return await self._json(system, {"pages": pages}, LensPeopleOutput, max_tokens=3000)


OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "KnowledgePlan": KnowledgePlan,
    "AnalysisOutput": AnalysisOutput,
    "RecallOutput": RecallOutput,
    "EnrichOutput": EnrichOutput,
    "MergeProfilesOutput": MergeProfilesOutput,
    "DeepProfileOutput": DeepProfileOutput,
    "LensPeopleOutput": LensPeopleOutput,
}


def build_llm(settings: Settings) -> LLMProvider:
    return OpenAICompatibleProvider(settings)
