from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SEED_FIELDS = ("company", "jobTitle", "occupation", "education", "location", "field")
FILTER_KEYS = {"company", "jobTitle", "occupation", "education", "location", "nationality", "field"}
# Names other clients use for the same filters (onesearch sends title/school).
FILTER_ALIASES = {"title": "jobTitle", "school": "education", "industry": "field", "role": "jobTitle", "city": "location"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clean_filters(v: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, val in (v or {}).items():
        k = FILTER_ALIASES.get(k, k)
        if k in FILTER_KEYS and k not in out and isinstance(val, str) and (val := " ".join(val.split())):
            out[k] = val[:120]
    return out


class SearchInput(BaseModel):
    """The name is required; filters are optional narrowing hints. Every
    other attribute is discovered."""

    name: str = Field(min_length=2, max_length=120)
    filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("name is required")
        return v

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


SourceType = Literal[
    "social_profile",
    "official_website",
    "news",
    "wiki",
    "academic",
    "code",
    "directory",
    "other",
]
ContentStatus = Literal["fetched", "snippet_only", "blocked", "failed"]


class Source(BaseModel):
    id: str
    title: str = ""
    url: str
    snippet: str = ""
    sourceType: SourceType = "other"
    provider: str
    query: str
    retrievedAt: datetime
    contentStatus: ContentStatus = "snippet_only"
    rank: float = 0.0
    knowledgeGraph: bool = False
    image: str | None = Field(default=None, exclude=True)


class Fact(BaseModel):
    field: str
    value: str
    sourceIds: list[str]
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    observedAt: datetime
    personKey: str | None = None


FieldStatus = Literal["found", "not_found", "not_checked", "conflicting", "restricted"]


class FieldValue(BaseModel):
    value: str | None = None
    values: list[str] | None = None
    sourceIds: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: FieldStatus


class SocialProfile(BaseModel):
    platform: str
    url: str
    sourceIds: list[str]
    confidence: float


class Photo(BaseModel):
    url: str
    pageUrl: str | None = None
    title: str | None = None
    sourceIds: list[str] = Field(default_factory=list)
    faceMatch: float | None = None
    note: str = "Image appears on the source page; it does not confirm identity by itself."
    fallbackUrl: str | None = Field(default=None, description="Google's cached copy, used when url does not serve an image")


class CareerItem(BaseModel):
    title: str | None = None
    organization: str | None = None
    period: str | None = None
    sourceIds: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    nameMatch: float
    distinguishingAttributes: float
    sourceQuality: float
    crossSourceAgreement: float
    prominence: float
    filterMatch: float
    conflictPenalty: float


class FeCandidate(BaseModel):
    candidateId: str
    displayName: str
    headline: str | None = None
    confidenceScore: float
    scoreBreakdown: ScoreBreakdown
    disambiguationNote: str = ""
    matchNote: str = ""
    nationality: str | None = None
    occupation: str | None = None
    knownFor: str | None = None
    imageSearchQuery: str = ""
    aliases: list[str] = Field(default_factory=list)
    avatarHint: str | None = None
    fields: dict[str, FieldValue]
    socialProfiles: list[SocialProfile] = Field(default_factory=list)
    officialLinks: list[str] = Field(default_factory=list, description="The person's Wikipedia article and official websites")
    sources: list[Source] = Field(default_factory=list)
    photos: list[Photo] = Field(default_factory=list)
    narrative: list[str] = Field(default_factory=list, description="Lead section of the person's English Wikipedia article; otherwise written by the model")
    narrativeSource: Literal["wikipedia", "llm"] | None = None
    careerArc: list[CareerItem] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    queryFaceMatch: float | None = None


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"


class ToolCallLog(BaseModel):
    server: str
    tool: str
    query: str
    latencyMs: int
    resultCount: int
    cached: bool = False
    error: str | None = None


class QueryLog(BaseModel):
    query: str
    kind: Literal["web", "image"]
    round: Literal["discovery", "enrichment"]
    purpose: str


class SearchResult(BaseModel):
    searchId: str
    mode: Literal["deep", "image"] = "deep"
    status: Literal["resolved", "ambiguous", "no_match"]
    input: SearchInput
    candidates: list[FeCandidate]
    sources: list[Source]
    queries: list[QueryLog]
    clarificationQuestion: str | None = None
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    completedAt: datetime


class Job(BaseModel):
    searchId: str
    status: JobStatus = JobStatus.queued
    stage: str = "queued"
    input: SearchInput
    createdAt: datetime = Field(default_factory=utcnow)
    updatedAt: datetime = Field(default_factory=utcnow)
    error: str | None = None
    toolCalls: list[ToolCallLog] = Field(default_factory=list)
    result: SearchResult | None = None
    extra: dict[str, Any] = Field(default_factory=dict, exclude=True)

    def touch(self, stage: str | None = None, status: JobStatus | None = None) -> None:
        if stage:
            self.stage = stage
        if status:
            self.status = status
        self.updatedAt = utcnow()


# ---------- people search contract (what the onesearch backend consumes) ----------


class ClientPhoto(BaseModel):
    url: str
    pageUrl: str | None = None
    title: str | None = None
    fromSource: bool = Field(description="True when the image appears on one of the candidate's sources; False for unconfirmed image-search suggestions")
    faceMatch: float | None = Field(default=None, description="Cosine similarity of the face to the person's reference face (1.0 = the reference photo); None when faces were not checked")
    fallbackUrl: str | None = Field(default=None, description="Google's cached copy of the image; show it if url fails to load")


class PeopleSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    filters: dict[str, str] = Field(default_factory=dict)
    mode: Literal["deep", "preview"] = Field(
        default="deep", description="deep = search + LLM analysis; preview = search results only, no LLM"
    )
    maxSearchQueries: int | None = Field(
        default=None, ge=1, le=3, description="Preview only: 1 = social networks, 2 = + public web, 3 = + images for the first candidates"
    )

    @field_validator("query")
    @classmethod
    def _query(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("query is required")
        return v

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class ClientSource(BaseModel):
    url: str
    title: str = ""
    type: str = Field(description='A platform from SOCIAL_ORDER, or "website", "wikipedia", "academic", "web"')
    snippet: str | None = None
    verified: bool = Field(description="Found by a live search; False when it only comes from the model's knowledge")


class ClientKeyFact(BaseModel):
    key: str
    label: str
    value: str
    verified: bool = Field(description="Backed by a search result; False when it is the model's unverified knowledge")


class ClientCareer(BaseModel):
    period: str | None = None
    title: str | None = None
    organization: str | None = None


class ClientCandidate(BaseModel):
    id: str = Field(description="Stable across repeated searches for the same person")
    displayName: str
    aliases: list[str] = Field(default_factory=list)
    headline: str | None = None
    confidence: float
    disambiguationNote: str | None = None
    nationality: str | None = None
    occupation: str | None = None
    knownFor: str | None = None
    location: str | None = None
    platforms: list[str] = Field(default_factory=list, description="Social platforms in SOCIAL_ORDER")
    sources: list[ClientSource] = Field(default_factory=list)
    keyFacts: list[ClientKeyFact] = Field(default_factory=list)
    facts: dict[str, str] = Field(
        default_factory=dict, description="keyFacts keyed like the onesearch FactsTable (primaryRole, company, awards, socialProfiles...)"
    )
    photos: list[ClientPhoto] = Field(default_factory=list)
    avatarUrl: str | None = None
    imageSearchQuery: str
    profile: list[str] = Field(default_factory=list, description="Biography paragraphs (deep) or result snippets (preview)")
    profileSource: Literal["wikipedia", "llm", "snippets"] | None = None
    career: list[ClientCareer] = Field(default_factory=list, description="Public roles, oldest first")
    tags: list[str] = Field(default_factory=list)
    queryFaceMatch: float | None = Field(
        default=None, description="Photo search only: similarity of the person's photo to the uploaded face (None when not checked)"
    )
    enrichedAt: str | None = Field(default=None, description="Set by POST /v1/profiles/enrich")
    enrichmentVersion: str | None = None


class CostLine(BaseModel):
    """One billed call, same columns as the Yescale usage table."""

    model: str
    kind: str
    tokenIn: int = 0
    tokenOut: int = 0
    totalTokens: int = 0
    usd: float = 0
    vnd: float = 0
    detail: str = ""


class ClientMeta(BaseModel):
    searchCalls: int = Field(description="Billable Serper calls made for this request (0 on a cache hit)")
    llmCalls: int = 0
    promptTokens: int = Field(default=0, description="LLM input tokens for this request")
    completionTokens: int = Field(default=0, description="LLM output tokens for this request")
    costUsd: float = Field(default=0, description="Yescale credit usage plus Serper/Scrape converted at the top-up rate")
    costVnd: float = Field(default=0, description="Same spend in VND (Yescale credits × top-up + unit prices)")
    costLines: list[CostLine] = Field(default_factory=list, description="Per-call rows: model, tokens, USD, VND")
    crawlfeedCalls: int = Field(default=0, description="Scrape Creators profile reads. A cache hit on our side counts as 0")
    crawlfeedCredits: int = Field(default=0, description="Credits Scrape Creators charged for those reads")
    cacheHit: bool = False
    totalSources: int = 0


PhotoWarning = Literal["no_face", "multiple_faces", "face_too_small"]


class IdentifiedPerson(BaseModel):
    """Who the model thinks the query (or photo) is. The fields come from the model's
    knowledge; only `socialSources` and `photos` were found by a live search."""

    displayName: str
    headline: str | None = None
    confidence: float
    disambiguationNote: str | None = None
    nationality: str | None = None
    occupation: str | None = None
    knownFor: str | None = None
    aliases: list[str] = Field(default_factory=list)
    sourceUrls: list[str] = Field(default_factory=list, description="Links the model remembers; not checked against any search")
    imageSearchQuery: str
    profile: list[str] = Field(default_factory=list)
    socialSources: list[ClientSource] = Field(default_factory=list, description="One verified profile per social network (withSources only)")
    photos: list[ClientPhoto] = Field(
        default_factory=list,
        description="withSources only, at most INITIAL_PHOTOS: images from the person's source pages first "
        "(fromSource=true; the first is the profile photo), then image-search photos whose title/URL carries the name",
    )


class PeopleSearchResponse(BaseModel):
    mode: Literal["deep", "preview", "image", "searchprofile"]
    searchId: str | None = None
    status: Literal["resolved", "ambiguous", "no_match"]
    clarificationQuestion: str | None = None
    candidates: list[ClientCandidate]
    meta: ClientMeta
    identified: list[IdentifiedPerson] | None = Field(default=None, description="Photo search only: people the model recognised in the photo")
    photoWarnings: list[PhotoWarning] | None = None
    lensInput: Literal["face", "photo"] | None = Field(default=None, description="Photo search only: what was sent to Google Lens when nobody was recognised (the cropped face, or the whole photo without a face)")
    lensResults: list[ClientSource] | None = Field(default=None, description="Photo search only: every page Google Lens returned for the photo")
    lensImageUrl: str | None = Field(default=None, description="Photo search only: MinIO URL of the image sent to Google Lens")
    warnings: list[str] | None = None


class HistoryCreateRequest(BaseModel):
    """Save the candidate the user opened. Pass searchId + candidateId for a deep
    search (the server copies the candidate from its result), or the full
    `candidate` object (preview results, or after the search has expired)."""

    userId: str | None = Field(default=None, max_length=128, description="Caller's user id; history is listed and guarded per user")
    query: str = Field(min_length=1, max_length=200)
    filters: dict[str, str] = Field(default_factory=dict)
    searchId: str | None = None
    candidateId: str | None = None
    candidate: ClientCandidate | None = None

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class Citation(BaseModel):
    url: str
    title: str = ""


class ChatMessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] | None = None
    createdAt: str


class HistorySummary(BaseModel):
    id: str
    query: str
    filters: dict[str, str]
    displayName: str
    headline: str | None = None
    avatarUrl: str | None = None
    messageCount: int = 0
    createdAt: str
    updatedAt: str


class HistoryDetail(BaseModel):
    id: str
    userId: str | None = None
    query: str
    filters: dict[str, str]
    searchId: str | None = None
    candidate: ClientCandidate
    messages: list[ChatMessageOut]
    createdAt: str
    updatedAt: str


class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=128, description="Client-declared chat session; stored in chat_session")
    candidateId: str = Field(min_length=1, max_length=128, description="Person this session talks about")
    question: str = Field(min_length=1, max_length=500)
    userId: str | None = Field(default=None, max_length=128)

    @field_validator("sessionId", "candidateId")
    @classmethod
    def _id(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("sessionId and candidateId are required")
        return v[:128]

    @field_validator("question")
    @classmethod
    def _question(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question is required")
        return v


class ChatMeta(BaseModel):
    searchQueries: list[str] = Field(default_factory=list, description="Queries the model chose to search; empty when it answered from knowledge")
    searchCalls: int = Field(description="Billable searches (cache hits excluded)")
    llmCalls: int
    promptTokens: int = 0
    completionTokens: int = 0
    costUsd: float = 0
    costVnd: float = 0
    costLines: list[CostLine] = Field(default_factory=list)
    latencyMs: int


class ChatResponse(BaseModel):
    sessionId: str
    candidateId: str
    searchId: str | None = None
    answer: str
    answeredFrom: Literal["knowledge", "web"]
    citations: list[Citation] = Field(description="Search results given to the model; empty when it answered from knowledge")
    meta: ChatMeta


class MorePhotosRequest(BaseModel):
    """More photos of a candidate: an image search on its imageSearchQuery, keeping only
    images whose face matches the reference photo (the candidate's first photo)."""

    query: str = Field(min_length=2, max_length=200, description="The candidate's imageSearchQuery")
    referenceUrl: str | None = Field(
        default=None,
        description="A photo known to show the person (candidate.photos[0].url / avatarUrl). Without it, the face most results agree on is kept",
    )
    name: str | None = Field(default=None, max_length=120, description="displayName; results whose title/URL carry neither it nor an alias are skipped")
    aliases: list[str] = Field(default_factory=list, max_length=10, description="The candidate's aliases (stage names are often what image titles use)")
    profileUrls: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="The candidate's source URLs; images found on these pages are kept even without the name in their caption, and trusted as the person's face",
    )
    filters: dict[str, str] = Field(default_factory=dict, description="The search's filters (pick the search region)")
    exclude: list[str] = Field(default_factory=list, max_length=100, description="Photo URLs the client already has")
    limit: int = Field(default=6, ge=1, le=12)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class MorePhotosMeta(BaseModel):
    searchCalls: int = Field(description="Billable image searches (0 when the query was cached by the search itself)")
    faceChecked: bool = Field(
        description="False when the photos are unverified: the face model is unavailable, or there was no reference photo and no two results showed the same face"
    )


class MorePhotosResponse(BaseModel):
    photos: list[ClientPhoto]
    meta: MorePhotosMeta


# ---------- building blocks: one step of the search each ----------


def _one_line(v: str) -> str:
    v = " ".join(v.split())
    if len(v) < 2:
        raise ValueError("at least 2 characters required")
    return v


class IdentifyRequest(BaseModel):
    query: str = Field(min_length=2, max_length=120)
    filters: dict[str, str] = Field(default_factory=dict)
    withSources: bool = Field(
        default=True,
        description="Also search social profiles and photos for the top people (2-3 search calls each); false = one LLM call only",
    )

    @field_validator("query")
    @classmethod
    def _query(cls, v: str) -> str:
        return _one_line(v)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class StepMeta(BaseModel):
    searchCalls: int = Field(default=0, description="Billable search calls (cache hits excluded)")
    llmCalls: int = 0
    promptTokens: int = 0
    completionTokens: int = 0
    costUsd: float = 0
    costVnd: float = 0
    costLines: list[CostLine] = Field(default_factory=list)


class IdentifyResponse(BaseModel):
    candidates: list[IdentifiedPerson]
    photoWarnings: list[PhotoWarning] | None = None
    warnings: list[str] = Field(default_factory=list)
    meta: StepMeta


class SocialSourcesRequest(BaseModel):
    """Social profiles of one known person (e.g. a candidate from /v1/identify)."""

    name: str = Field(min_length=2, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=10)
    occupation: str | None = Field(default=None, max_length=200)
    headline: str | None = Field(default=None, max_length=200)
    knownFor: str | None = Field(default=None, max_length=200)
    filters: dict[str, str] = Field(default_factory=dict)
    withIdentity: bool = Field(
        default=True, description="Also search name + occupation on the big video/photo networks; those accounts must describe the person"
    )

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _one_line(v)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class WebSourcesRequest(BaseModel):
    """Public web pages about one person: the name (plus location filter) on Google."""

    name: str = Field(min_length=2, max_length=120)
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=8, ge=1, le=20)
    includeNews: bool = Field(default=False, description="Keep news sites and article pages")

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _one_line(v)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class SourcesResponse(BaseModel):
    sources: list[ClientSource]
    meta: StepMeta


class ImageSearchRequest(BaseModel):
    """Raw image search, no face check (cheap). For face-checked photos use /v1/photos."""

    query: str = Field(min_length=2, max_length=200)
    name: str | None = Field(default=None, max_length=120, description="When set, results whose title/URL do not carry the name are skipped")
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=3, ge=1, le=12)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class ImageSearchResponse(BaseModel):
    photos: list[ClientPhoto]
    meta: StepMeta


class SourceRef(BaseModel):
    url: str = Field(max_length=2000)
    type: str = Field(default="web", max_length=40, description='As in ClientSource: a social platform, "website", "wikipedia", "academic" or "web"')
    title: str = Field(default="", max_length=300)
    verified: bool = Field(default=False, description="Found by a live search. Unverified pages (except a Wikipedia article named after the person) need a face match")


class SourcePhotosRequest(BaseModel):
    """Photos from the person's own pages, for a candidate with sources but no photo.
    No search call, no LLM call: Wikipedia lead image, GitHub avatar, page og:image."""

    name: str = Field(min_length=2, max_length=120)
    sources: list[SourceRef] = Field(min_length=1, max_length=30)
    limit: int = Field(default=3, ge=1, le=10)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _one_line(v)


class SourcePhotosResponse(BaseModel):
    photos: list[ClientPhoto]
    faceChecked: bool = Field(description="False when the face model is unavailable or no reference face was found: photos are unverified")
    meta: StepMeta


class EnrichRequest(BaseModel):
    """Write a full profile (biography, facts, career) for one candidate from its
    sources. Pass `historyId` to use and update the saved candidate, or `candidate`."""

    historyId: str | None = None
    userId: str | None = Field(default=None, max_length=128)
    candidate: ClientCandidate | None = None
    force: bool = Field(default=False, description="Re-run even when the candidate is already enriched with the current prompt")


class EnrichResponse(BaseModel):
    candidate: ClientCandidate
    cached: bool = Field(description="True when the stored profile was returned without an LLM call")
    meta: StepMeta


class SearchProfileRequest(BaseModel):
    """Name search. Filters narrow the Serper queries; each one is optional."""

    query: str = Field(min_length=2, max_length=120)
    searchId: str | None = Field(default=None, max_length=128, description="Client-declared id; stored in Postgres with the candidate rows")
    userId: str | None = Field(default=None, max_length=128)
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="location, education, jobTitle, company, field, occupation (aliases: title, school, industry, role, city)",
    )
    excludeUrls: list[str] = Field(
        default_factory=list,
        max_length=80,
        description="Source URLs already on screen. Triggers a second round of search for other people.",
    )
    excludeNames: list[str] = Field(
        default_factory=list,
        max_length=40,
        description="displayName values already shown; those people are not returned again.",
    )
    page: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Serper result page for both queries. First search is 1; Tìm thêm is 2, then 3, …",
    )
    refresh: bool = Field(default=False, description="Search again even when a recent search (any user) has the same query and filters")

    @field_validator("query")
    @classmethod
    def _query(cls, v: str) -> str:
        return _one_line(v)

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)


class DeepSearchRequest(BaseModel):
    """One person chosen from search-profile. Queue clients send candidateId; HTTP may still send the object."""

    candidate: ClientCandidate | None = None
    candidateId: str | None = Field(default=None, max_length=128)
    filters: dict[str, str] = Field(default_factory=dict, description="The filters used in the name search, if any")
    userId: str | None = Field(default=None, max_length=128)
    refresh: bool = Field(default=False, description="Run the deep search again even when this person (any user) has a recent dossier")

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, v: dict[str, str] | None) -> dict[str, str]:
        return _clean_filters(v)

    @model_validator(mode="after")
    def _who(self):
        if self.candidate is None and not (self.candidateId or "").strip():
            raise ValueError("candidateId or candidate is required")
        return self


class DeepSearchResponse(BaseModel):
    searchId: str | None = None
    candidate: ClientCandidate
    socialSources: list[ClientSource] = Field(description="Social accounts and profile pages (personal site, Wikipedia)")
    articleSources: list[ClientSource] = Field(description="Articles and interviews about the person")
    warnings: list[str] = Field(default_factory=list)
    meta: ClientMeta
