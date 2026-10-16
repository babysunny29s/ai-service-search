"""Yescale + vendor unit prices → USD/credits and VND.

Yescale dashboard:

    Input  = tokens * price_in  / 1,000,000
    Output = tokens * price_out / 1,000,000
    Total Credit Usage = (Input + Output) * group_ratio
    VND = credits * top-up rate  (e.g. 8,000 ₫ / credit)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings


@dataclass(frozen=True)
class LlmCost:
    input_usd: float
    output_usd: float
    credits: float
    vnd: float

    def cost_details(self) -> dict[str, float]:
        return {"input": self.input_usd, "output": self.output_usd, "total": self.credits}


def yescale_llm(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    input_per_1m: float,
    output_per_1m: float,
    group_ratio: float = 1.0,
    vnd_per_credit: float = 8000,
) -> LlmCost:
    input_usd = prompt_tokens * input_per_1m / 1_000_000
    output_usd = completion_tokens * output_per_1m / 1_000_000
    credits = (input_usd + output_usd) * group_ratio
    return LlmCost(input_usd=input_usd, output_usd=output_usd, credits=credits, vnd=credits * vnd_per_credit)


def llm_from_settings(prompt_tokens: int, completion_tokens: int, settings: Settings | None = None) -> LlmCost:
    s = settings or _settings()
    return yescale_llm(
        prompt_tokens,
        completion_tokens,
        input_per_1m=s.llm_price_input_per_1m,
        output_per_1m=s.llm_price_output_per_1m,
        group_ratio=s.llm_group_ratio,
        vnd_per_credit=s.yescale_vnd_per_credit,
    )


def extra_vnd(*, serper_calls: int = 0, scrape_credits: int = 0, settings: Settings | None = None) -> float:
    """Serper: 1 billable search = 1 token. Scrape Creators: live credits_charged."""
    s = settings or _settings()
    return serper_calls * s.serper_price_per_search + scrape_credits * s.scrapecreators_price_per_credit


def usd_from_vnd(vnd: float, settings: Settings | None = None) -> float:
    s = settings or _settings()
    rate = s.yescale_vnd_per_credit or 0
    return (vnd / rate) if rate else 0.0


def serper_cost_details(billable: int, settings: Settings | None = None) -> tuple[dict[str, int], dict[str, float]]:
    usd = usd_from_vnd(extra_vnd(serper_calls=billable, settings=settings), settings)
    return {"input": billable, "output": 0, "total": billable}, {"total": usd}


def scrape_cost_details(credits: int, settings: Settings | None = None) -> tuple[dict[str, int], dict[str, float]]:
    usd = usd_from_vnd(extra_vnd(scrape_credits=credits, settings=settings), settings)
    return {"input": credits, "output": 0, "total": credits}, {"total": usd}


def _settings() -> Settings:
    from app.config import get_settings

    return get_settings()
