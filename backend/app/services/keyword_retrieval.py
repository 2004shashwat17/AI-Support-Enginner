from typing import Protocol
from uuid import UUID

from app.db.repositories.retrieval import RetrievedChunk
from app.services.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    QueryEmbeddingModelError,
    validate_retrieval_request,
)


class KeywordRetrievalStore(Protocol):
    def search_keyword(
        self,
        query: str,
        *,
        top_k: int,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]: ...


class KeywordRetriever:
    def __init__(
        self,
        repository: KeywordRetrievalStore,
        *,
        embedding_model: str,
        default_top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
    ) -> None:
        if max_top_k <= 0:
            raise ValueError("Maximum Top-K must be greater than zero.")
        if not 1 <= default_top_k <= max_top_k:
            raise ValueError("Default Top-K must be within the supported range.")

        self._repository = repository
        self._embedding_model = embedding_model
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
                "Requested embedding model does not match retrieval configuration."
            )

        return self._repository.search_keyword(
            query,
            top_k=requested_top_k,
            document_id=document_id,
            embedding_model=requested_model,
        )