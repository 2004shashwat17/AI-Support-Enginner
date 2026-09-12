import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.rag import LLMGroundedAnswer
from app.services.llm import MalformedLLMResponseError, OpenAILLMProvider


def test_openai_provider_returns_structured_answer() -> None:
    parsed = LLMGroundedAnswer(answer="Reset it from Settings.", cited_source_ids=["S1"])
    client = Mock()
    client.responses.parse = AsyncMock(return_value=Mock(output_parsed=parsed))
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
    client.responses.parse = AsyncMock(return_value=Mock(output_parsed=None))
    provider = OpenAILLMProvider(client, model="test-model", temperature=0.0)

    with pytest.raises(MalformedLLMResponseError, match="no structured answer"):
        asyncio.run(provider.generate(system_prompt="system", user_prompt="user"))
