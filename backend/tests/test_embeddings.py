import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.document_chunking import DocumentChunk
from app.services.embeddings import (
    EmbeddingProviderError,
    EmbeddingService,
    EmptyEmbeddingInputError,
    MalformedEmbeddingResponseError,
    OpenAIEmbeddingProvider,
)


DOCUMENT_ID = UUID("86d7f8e8-a5f4-4b8d-9564-1f6db6b1d45a")


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


class FailingProvider:
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        raise ConnectionError("provider unavailable")


class MalformedProvider:
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        return []


def make_chunk(index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=UUID(int=index + 1),
        document_id=DOCUMENT_ID,
        content=content,
        chunk_index=index,
        metadata={"source_filename": "guide.txt"},
    )


def test_loads_embedding_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding-model")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "25")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "custom-llm-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.25")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.embedding_model == "custom-embedding-model"
    assert settings.embedding_dimensions == 1536
    assert settings.embedding_batch_size == 25
    assert settings.llm_provider == "openai"
    assert settings.llm_model == "custom-llm-model"
    assert settings.llm_temperature == 0.25


def test_rejects_dimensions_that_do_not_match_storage_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_embeds_text_successfully() -> None:
    service = EmbeddingService(RecordingProvider(), model="test-model")

    vector = asyncio.run(service.embed_text("reset password"))

    assert vector == (14.0, 0.0)


def test_rejects_empty_text() -> None:
    service = EmbeddingService(RecordingProvider(), model="test-model")

    with pytest.raises(EmptyEmbeddingInputError, match="cannot be empty"):
        asyncio.run(service.embed_text("  \n"))


def test_wraps_provider_failure() -> None:
    service = EmbeddingService(FailingProvider(), model="test-model")

    with pytest.raises(EmbeddingProviderError, match="provider failed") as error:
        asyncio.run(service.embed_text("reset password"))

    assert isinstance(error.value.__cause__, ConnectionError)


def test_rejects_malformed_provider_response() -> None:
    service = EmbeddingService(MalformedProvider(), model="test-model")

    with pytest.raises(MalformedEmbeddingResponseError, match="different number"):
        asyncio.run(service.embed_text("reset password"))


def test_batches_chunks_and_preserves_identifiers() -> None:
    provider = RecordingProvider()
    service = EmbeddingService(provider, model="test-model", batch_size=2)
    chunks = [make_chunk(index, f"text-{index}") for index in range(5)]

    embedded_chunks = asyncio.run(service.embed_chunks(chunks))

    assert [len(call) for call in provider.calls] == [2, 2, 1]
    assert [result.chunk_id for result in embedded_chunks] == [
        chunk.chunk_id for chunk in chunks
    ]
    assert all(result.document_id == DOCUMENT_ID for result in embedded_chunks)
    assert [result.chunk_index for result in embedded_chunks] == list(range(5))
    assert [result.content for result in embedded_chunks] == [
        chunk.content for chunk in chunks
    ]
    assert all(result.embedding_model == "test-model" for result in embedded_chunks)


def test_openai_provider_restores_response_order() -> None:
    client = Mock()
    client.embeddings.create = AsyncMock(
        return_value=Mock(
            data=[
                Mock(index=1, embedding=[0.3, 0.4]),
                Mock(index=0, embedding=[0.1, 0.2]),
            ]
        )
    )
    provider = OpenAIEmbeddingProvider(client)

    vectors = asyncio.run(provider.embed(["first", "second"], model="test-model"))

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    client.embeddings.create.assert_awaited_once_with(
        input=["first", "second"],
        model="test-model",
        encoding_format="float",
    )


@pytest.mark.parametrize(
    "vectors",
    [
        [[]],
        [[float("nan")]],
        [[float("inf")]],
    ],
)
def test_rejects_unusable_vectors(vectors: list[list[float]]) -> None:
    provider = AsyncMock()
    provider.embed.return_value = vectors
    service = EmbeddingService(provider, model="test-model")

    with pytest.raises(MalformedEmbeddingResponseError):
        asyncio.run(service.embed_text("content"))
