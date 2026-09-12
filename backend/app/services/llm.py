import logging
import time
from typing import Protocol

from openai import AsyncOpenAI

from app.models.rag import LLMGroundedAnswer
from app.observability.context import correlation_tags
from app.observability.cost import TokenUsage, estimate_cost
from app.observability.metrics import LoggingMetricsSink, MetricsSink


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base exception for language-model failures."""


class LLMProviderError(LLMError):
    """Raised when the configured language-model provider fails."""


class MalformedLLMResponseError(LLMError):
    """Raised when a provider returns no valid structured answer."""


class LLMProvider(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGroundedAnswer: ...


class OpenAILLMProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        temperature: float,
        metrics_sink: MetricsSink | None = None,
        prompt_price_per_1k: float | None = None,
        completion_price_per_1k: float | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._metrics_sink = metrics_sink or LoggingMetricsSink()
        self._prompt_price_per_1k = prompt_price_per_1k
        self._completion_price_per_1k = completion_price_per_1k

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGroundedAnswer:
        start = time.perf_counter()
        try:
            response = await self._client.responses.parse(
                model=self._model,
                temperature=self._temperature,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=LLMGroundedAnswer,
            )
        except Exception as exc:
            self._metrics_sink.increment(
                "llm_call_failed", **correlation_tags(), model=self._model
            )
            raise LLMProviderError("The language-model provider failed.") from exc
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self._metrics_sink.record_latency(
                "llm", duration_ms, **correlation_tags(), model=self._model
            )

        self._record_token_usage(response)

        if response.output_parsed is None:
            raise MalformedLLMResponseError(
                "The language-model provider returned no structured answer."
            )
        return response.output_parsed

    def _record_token_usage(self, response: object) -> None:
        """Best-effort token usage/cost logging. Never fabricates a cost:
        if pricing is not configured, cost is logged as unknown (None).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        if prompt_tokens is None or completion_tokens is None:
            return

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        cost = estimate_cost(
            token_usage,
            prompt_price_per_1k=self._prompt_price_per_1k,
            completion_price_per_1k=self._completion_price_per_1k,
        )
        tags = correlation_tags()
        self._metrics_sink.record_value(
            "llm_prompt_tokens", token_usage.prompt_tokens, **tags, model=self._model
        )
        self._metrics_sink.record_value(
            "llm_completion_tokens",
            token_usage.completion_tokens,
            **tags,
            model=self._model,
        )
        logger.info(
            "llm_token_usage prompt_tokens=%d completion_tokens=%d total_cost_usd=%s",
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
            "unknown" if cost.total_cost_usd is None else f"{cost.total_cost_usd:.6f}",
            extra={
                **tags,
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_cost_usd": cost.total_cost_usd,
            },
        )
