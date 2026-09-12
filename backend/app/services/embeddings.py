from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol
from uuid import UUID

from openai import AsyncOpenAI

from app.core.config import Settings
from app.services.document_chunking import DocumentChunk
from app.services.document_ingestion import MetadataValue


class EmbeddingError(RuntimeError):
    """Base exception for embedding failures."""


class EmptyEmbeddingInputError(EmbeddingError):
    """Raised when text has no content to embed."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the configured embedding provider fails."""


class MalformedEmbeddingResponseError(EmbeddingError):
    """Raised when a provider returns unusable embedding data."""


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        """Return one vector for each input text in the same order."""


class OpenAIEmbeddingProvider:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def embed(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        response = await self._client.embeddings.create(
            input=list(texts),
            model=model,
            encoding_format="float",
        )
        ordered_embeddings = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered_embeddings]


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    embedding: tuple[float, ...]
    embedding_model: str
    metadata: dict[str, MetadataValue]


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        model: str,
        batch_size: int = 100,
    ) -> None:
        if not model.strip():
            raise ValueError("Embedding model cannot be empty.")
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than zero.")

        self._provider = provider
        self._model = model
        self._batch_size = batch_size

    async def embed_text(self, text: str) -> tuple[float, ...]:
        vectors = await self._embed_texts([text])
        return vectors[0]

    async def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        vectors = await self._embed_texts([chunk.content for chunk in chunks])
        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=vector,
                embedding_model=self._model,
                metadata=dict(chunk.metadata),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    async def _embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if any(not text.strip() for text in texts):
            raise EmptyEmbeddingInputError("Text to embed cannot be empty.")

        vectors: list[tuple[float, ...]] = []
        expected_dimension: int | None = None

        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            try:
                provider_vectors = await self._provider.embed(
                    batch,
                    model=self._model,
                )
            except Exception as exc:
                raise EmbeddingProviderError("The embedding provider failed.") from exc

            validated_vectors = _validate_vectors(provider_vectors, len(batch))
            for vector in validated_vectors:
                if expected_dimension is None:
                    expected_dimension = len(vector)
                elif len(vector) != expected_dimension:
                    raise MalformedEmbeddingResponseError(
                        "Embedding dimensions must be consistent."
                    )
            vectors.extend(validated_vectors)

        return vectors


def create_openai_embedding_service(settings: Settings) -> EmbeddingService:
    provider = OpenAIEmbeddingProvider(
        AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    )
    return EmbeddingService(
        provider,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def _validate_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
) -> list[tuple[float, ...]]:
    if len(vectors) != expected_count:
        raise MalformedEmbeddingResponseError(
            "The provider returned a different number of embeddings than inputs."
        )

    validated_vectors: list[tuple[float, ...]] = []
    for vector in vectors:
        if not vector:
            raise MalformedEmbeddingResponseError("Embedding vectors cannot be empty.")
        if any(isinstance(value, bool) or not isfinite(value) for value in vector):
            raise MalformedEmbeddingResponseError(
                "Embedding vectors must contain only finite numbers."
            )
        validated_vectors.append(tuple(vector))

    return validated_vectors
