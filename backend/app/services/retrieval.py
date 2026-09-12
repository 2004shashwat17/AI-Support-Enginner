from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.db.repositories.retrieval import RetrievedChunk


DEFAULT_TOP_K = 5
MAX_TOP_K = 100


class RetrievalError(ValueError):
    """Base exception for invalid retrieval requests."""


class EmptyQueryError(RetrievalError):
    """Raised when a query has no searchable text."""


class InvalidTopKError(RetrievalError):
    """Raised when Top-K is outside the supported range."""


class QueryEmbeddingDimensionError(RetrievalError):
    """Raised when the query vector cannot be compared with stored vectors."""


class QueryEmbeddingModelError(RetrievalError):
    """Raised when query and stored vectors use different models."""


class RetrievalStore(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]: ...


class QueryEmbedder(Protocol):
    async def embed_text(self, text: str) -> tuple[float, ...]: ...


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]: ...


def validate_retrieval_request(
    query: str,
    top_k: int | None,
    *,
    default_top_k: int,
    max_top_k: int,
) -> int:
    if not query.strip():
        raise EmptyQueryError("Search query cannot be empty.")

    requested_top_k = default_top_k if top_k is None else top_k
    if not 1 <= requested_top_k <= max_top_k:
        raise InvalidTopKError(f"Top-K must be between 1 and {max_top_k}.")
    return requested_top_k


class RetrievalService:
    def __init__(
        self,
        embedding_service: QueryEmbedder,
        repository: RetrievalStore,
        *,
        embedding_model: str,
        embedding_dimensions: int,
        default_top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
    ) -> None:
        if max_top_k <= 0:
            raise ValueError("Maximum Top-K must be greater than zero.")
        if not 1 <= default_top_k <= max_top_k:
            raise ValueError("Default Top-K must be within the supported range.")

        self._embedding_service = embedding_service
        self._repository = repository
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._default_top_k = default_top_k
        self._max_top_k = max_top_k

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        requested_top_k = validate_retrieval_request(
            query,
            top_k,
            default_top_k=self._default_top_k,
            max_top_k=self._max_top_k,
        )

        requested_model = embedding_model or self._embedding_model
        if requested_model != self._embedding_model:
            raise QueryEmbeddingModelError(
                "Query embedding model does not match retrieval configuration."
            )

        query_embedding = await self._embedding_service.embed_text(query)
        if len(query_embedding) != self._embedding_dimensions:
            raise QueryEmbeddingDimensionError(
                "Query embedding dimensions do not match retrieval configuration."
            )

        return self._repository.search(
            query_embedding,
            top_k=requested_top_k,
            document_id=document_id,
            embedding_model=requested_model,
        )
