from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    llm_base_url: str = "https://api.yescale.io/v1"
    llm_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_timeout_s: float = 180
    llm_temperature: float = 0.1
    # Thinking budget for reasoning models ("none", "low", "medium", "high"; empty = provider default).
    # Thinking tokens are billed as output and generated one by one, so they dominate latency.
    llm_reasoning_effort: str = "low"
    output_language: str = "English"

    mcp_config_path: str = "mcp_servers.json"

    max_queries: int = 2
    results_per_query: int = 10
    max_fetch_urls: int = 4
    max_llm_sources: int = 20
    excerpt_chars: int = 2000
    enrich_candidates: int = 2
    enrich_queries_per_candidate: int = 1
    enrich_fetch_urls: int = 3
    enrich_min_identifying: int = 3
    enrich_images: bool = False
    # One image search per strong candidate to fill its photo gallery (0 disables).
    photo_search_candidates: int = 3
    max_photos: int = 10
    # Photos returned per person in a search result: photos from the person's own pages
    # first (the first one is the profile photo), then image-search results by face match.
    # More come from POST /v1/photos, which runs the image query on demand.
    initial_photos: int = 3
    more_photos_max: int = 12
    # People returned per search (the LLM is asked for at most this many).
    max_candidates: int = 15
    # Hide people with no search-found source (only the model's memory) / with no face-checked photo.
    require_source: bool = True
    require_photo: bool = False
    # Keep only photos whose face matches the person's reference face (OpenCV, CPU).
    face_verify: bool = True
    face_model_dir: str = "models"
    # SFace cosine similarity; 0.363 is the model's same-person floor. A score that low
    # still mixes in other people (a 37% match), so a photo is shown only above this bar.
    face_match_threshold: float = 0.363
    face_gallery_threshold: float = 0.5
    # Two candidates whose reference faces are at least this similar share a face; only the
    # one with the most photos of it keeps it (0 = off).
    face_shared_threshold: float = 0.55
    # Preview mode (no LLM): 1 = social networks, 2 = + public web, 3 = + images.
    preview_max_queries: int = 3
    preview_max_candidates: int = 15
    preview_photo_candidates: int = 3
    preview_cache_ttl_s: float = 21600
    result_cache_ttl_s: float = 21600
    # Results saved in Postgres by any user are reused this long (0 = never): a name search
    # with the same query + filters, and a deep search of the same person (Candidate.personKey).
    reuse_search_max_age_h: float = 24
    reuse_deep_max_age_h: float = 168
    # llm_first: ask the model first and search only for what it does not know.
    # search_first: always search, the model only reads the results.
    search_strategy: str = "llm_first"
    # LLM calls per search. 1 = search, then one analysis (llm_first needs 2:
    # plan + analysis, so with 1 the search runs first; no enrichment round).
    llm_max_calls: int = 1
    # Photo search: people recognised in the photo that get a full deep search each.
    image_search_people: int = 2
    image_min_confidence: float = 0.3
    # Nobody recognised: reverse image search (Serper Lens) on the largest face, its box
    # widened lens_face_pad times (whole photo without a face), then read lens_pages_to_read pages.
    lens_face_pad: float = 1.75
    lens_pages_to_read: int = 8
    # /v1/identify withSources: people (confidence >= image_min_confidence) that get social
    # profiles + photos (source photos first; the image search only runs to fill up to initial_photos).
    identify_sources_people: int = 3
    # Photos from source pages (no search, no LLM): pages fetched per person, photos kept,
    # and preview candidates that get them.
    source_photo_pages: int = 5
    source_photo_limit: int = 3
    source_photo_candidates: int = 3
    social_lookup_candidates: int = 2
    social_min_platforms: int = 3
    fetch_timeout_s: float = 30
    fetch_user_agent: str = ""
    fetch_max_bytes: int = 1_500_000
    job_timeout_s: float = 300
    sync_timeout_s: float = 180
    # Seconds to hold a POST before 202 + poll. 0 = accept immediately; trycloudflare
    # cancels an idle POST, so the job must not sit on that connection. Poll GET instead.
    http_wait_s: float = 0
    search_cache_ttl_s: float = 900
    # Search locale when neither the filters nor the name point to a country
    # (see search_locale.py). Results are always in output_language.
    search_locale_gl: str = "us"
    search_locale_hl: str = "en"
    # Scrape Creators profile reads (Facebook, Instagram, TikTok, X, LinkedIn, Threads).
    # https://docs.scrapecreators.com — 1 credit per live profile. CRAWLFEED_* still accepted.
    crawlfeed_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("SCRAPECREATORS_API_KEY", "SCRAPE_CREATORS_API_KEY", "CRAWLFEED_API_KEY", "CRAWLFEED_KEY"),
    )
    crawlfeed_base_url: str = Field(
        default="https://api.scrapecreators.com",
        validation_alias=AliasChoices("SCRAPECREATORS_BASE_URL", "CRAWLFEED_BASE_URL"),
    )
    crawlfeed_timeout_s: float = Field(
        default=30,
        validation_alias=AliasChoices("SCRAPECREATORS_TIMEOUT_S", "CRAWLFEED_TIMEOUT_S"),
    )

    # Search history + chat about a saved candidate (SQLite file, relative to the project root).
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        validation_alias=AliasChoices("RABBITMQ_URL", "AMQP_URL"),
    )
    rabbitmq_queue: str = Field(
        default="AI_SEARCH_QUEUE_IN",
        validation_alias=AliasChoices("SEARCH_QUEUE_REQUEST", "RABBITMQ_QUEUE", "RABBITMQ_REQUEST_QUEUE"),
    )
    rabbitmq_result_queue: str = Field(
        default="AI_SEARCH_QUEUE_OUT",
        validation_alias=AliasChoices("SEARCH_RESULT_QUEUE", "RABBITMQ_RESULT_QUEUE", "RABBITMQ_REPLY_QUEUE"),
    )
    rabbitmq_prefetch: int = Field(default=1, validation_alias=AliasChoices("RABBITMQ_PREFETCH", "RABBITMQ_PREFETCH_COUNT"))
    rabbitmq_queue_durable: bool = True
    rabbitmq_consumer_timeout_s: float = Field(default=300, validation_alias=AliasChoices("RABBITMQ_CONSUMER_TIMEOUT"))

    db_host: str = Field(default="", validation_alias=AliasChoices("DB_HOST"))
    db_port: int = Field(default=5432, validation_alias=AliasChoices("DB_PORT"))
    db_name: str = Field(default="", validation_alias=AliasChoices("DB_NAME"))
    db_user: str = Field(default="", validation_alias=AliasChoices("DB_USER"))
    db_password: str = Field(default="", validation_alias=AliasChoices("DB_PASSWORD"))

    history_db_path: str = "data/history.db"
    # Previous turns (question + answer) sent back to the model with each question.
    chat_max_turns: int = 4
    chat_turn_chars: int = 280
    # Web searches the model may run for one question when it does not know (0 = never search).
    chat_max_searches: int = 2
    # Results kept per search.
    chat_search_results: int = 5
    # Top result pages read in full (passages around the name) when the model searches.
    chat_fetch_pages: int = 3
    chat_page_chars: int = 1500
    chat_fetch_timeout_s: float = 20
    chat_max_tokens: int = 800

    # Yescale: Input = tokens * LLM_PRICE_INPUT_PER_1M / 1e6
    # Output = tokens * LLM_PRICE_OUTPUT_PER_1M / 1e6
    # credits = (input + output) * LLM_GROUP_RATIO
    # VND = credits * YESCALE_VND_PER_CREDIT (top-up rate)
    llm_price_input_per_1m: float = 0.75
    llm_price_output_per_1m: float = 3.75
    llm_group_ratio: float = 1.0
    yescale_vnd_per_credit: float = 8000
    # Serper: 1 search credit ("token") = this many VND. Cache hit = 0.
    serper_price_per_search: float = Field(
        default=26,
        validation_alias=AliasChoices("SERPER_PRICE_PER_SEARCH", "SERPER_PRICE_PER_TOKEN"),
    )
    scrapecreators_price_per_credit: float = 49
    cost_currency: str = "VND"

    @field_validator(
        "llm_price_input_per_1m",
        "llm_price_output_per_1m",
        "llm_group_ratio",
        "yescale_vnd_per_credit",
        "serper_price_per_search",
        "scrapecreators_price_per_credit",
        mode="before",
    )
    @classmethod
    def _empty_float(cls, v: Any, info: ValidationInfo) -> Any:
        if v != "":
            return v
        return {
            "llm_price_input_per_1m": 0.75,
            "llm_price_output_per_1m": 3.75,
            "llm_group_ratio": 1.0,
            "yescale_vnd_per_credit": 8000.0,
            "serper_price_per_search": 26.0,
            "scrapecreators_price_per_credit": 49.0,
        }[info.field_name]

    @field_validator("cost_currency", mode="before")
    @classmethod
    def _empty_currency(cls, v: Any) -> Any:
        return "VND" if v in ("", None) else str(v).upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()
