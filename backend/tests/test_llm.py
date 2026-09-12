import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.rag import LLMGroundedAnswer
from app.services.llm import MalformedLLMResponseError, OpenAILLMProvider


def test_openai_provider_returns_structured_answer() -> None:
    parsed = LLMGroundedAnswer(answer="Reset it from Settings.", cited_source_ids=["S1"])
    client = Mock()
    client.responses.parse = AsyncMock(return_value=Mock(output_parsed=parsed, usage=None))
    provider = OpenAILLMProvider(client, model="test-model", temperature=0.2)

    result = asyncio.run(
        provider.generate(system_prompt="system", user_prompt="question and context")
    )

    assert result == parsed
    client.responses.parse.assert_awaited_once_with(
        model="test-model",
        temperature=0.2,
        input=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question and context"},
        ],
        text_format=LLMGroundedAnswer,
    )


def test_openai_provider_rejects_missing_structured_answer() -> None:
    client = Mock()
    client.responses.parse = AsyncMock(return_value=Mock(output_parsed=None, usage=None))
    provider = OpenAILLMProvider(client, model="test-model", temperature=0.0)

    with pytest.raises(MalformedLLMResponseError, match="no structured answer"):
        asyncio.run(provider.generate(system_prompt="system", user_prompt="user"))


def test_openai_provider_records_token_usage_and_cost_via_metrics_sink() -> None:
    parsed = LLMGroundedAnswer(answer="Reset it from Settings.", cited_source_ids=["S1"])
    usage = Mock(input_tokens=100, output_tokens=50)
    client = Mock()
    client.responses.parse = AsyncMock(return_value=Mock(output_parsed=parsed, usage=usage))

    class RecordingMetricsSink:
        def __init__(self) -> None:
            self.values: list[tuple[str, float]] = []

        def record_latency(self, stage, duration_ms, **tags):
            pass

        def record_value(self, name, value, **tags):
            self.values.append((name, value))

        def increment(self, counter, **tags):
            pass

    sink = RecordingMetricsSink()
    provider = OpenAILLMProvider(
        client,
        model="test-model",
        temperature=0.0,
        metrics_sink=sink,
        prompt_price_per_1k=0.01,
        completion_price_per_1k=0.03,
    )

    asyncio.run(provider.generate(system_prompt="system", user_prompt="user"))

    assert ("llm_prompt_tokens", 100) in sink.values
    assert ("llm_completion_tokens", 50) in sink.values


def test_openai_provider_does_not_fabricate_cost_without_pricing(caplog) -> None:
    parsed = LLMGroundedAnswer(answer="Reset it from Settings.", cited_source_ids=["S1"])
    usage = Mock(input_tokens=100, output_tokens=50)
    client = Mock()
    client.responses.parse = AsyncMock(return_value=Mock(output_parsed=parsed, usage=usage))
    provider = OpenAILLMProvider(client, model="test-model", temperature=0.0)

    with caplog.at_level("INFO"):
        asyncio.run(provider.generate(system_prompt="system", user_prompt="user"))

    assert any("total_cost_usd=unknown" in message for message in caplog.messages)
