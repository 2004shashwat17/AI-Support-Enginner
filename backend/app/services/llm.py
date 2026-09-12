from typing import Protocol

from openai import AsyncOpenAI

from app.models.rag import LLMGroundedAnswer


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
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMGroundedAnswer:
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
            raise LLMProviderError("The language-model provider failed.") from exc

        if response.output_parsed is None:
            raise MalformedLLMResponseError(
                "The language-model provider returned no structured answer."
            )
        return response.output_parsed
