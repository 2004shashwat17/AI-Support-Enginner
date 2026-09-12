"""Token usage and cost estimation.

Never invents pricing. If a per-1K-token price is not configured, the
corresponding cost is reported as `None` ("unknown"), never as zero or a
guessed number.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class CostEstimate:
    token_usage: TokenUsage
    prompt_cost_usd: float | None
    completion_cost_usd: float | None

    @property
    def total_cost_usd(self) -> float | None:
        if self.prompt_cost_usd is None or self.completion_cost_usd is None:
            return None
        return self.prompt_cost_usd + self.completion_cost_usd


def estimate_cost(
    usage: TokenUsage,
    *,
    prompt_price_per_1k: float | None,
    completion_price_per_1k: float | None,
) -> CostEstimate:
    prompt_cost = (
        (usage.prompt_tokens / 1000) * prompt_price_per_1k
        if prompt_price_per_1k is not None
        else None
    )
    completion_cost = (
        (usage.completion_tokens / 1000) * completion_price_per_1k
        if completion_price_per_1k is not None
        else None
    )
    return CostEstimate(usage, prompt_cost, completion_cost)
