import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.services.retrieval import (
    EmptyQueryError,
    InvalidTopKError,
    QueryEmbeddingDimensionError,
    QueryEmbeddingModelError,
    RetrievalService,
)


DOCUMENT_ID = UUID(int=100)


class FakeEmbeddingService:
    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0, 0.0)) -> None:
        self.vector = vector
        self.queries: list[str] = []

    async def embed_text(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        return self.vector


class RecordingRetrievalStore:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query_embedding": tuple(query_embedding),
                "top_k": top_k,
                "document_id": document_id,
                "embedding_model": embedding_model,
            }
        )
        return self.results[:top_k]


def make_result(index: int, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=DOCUMENT_ID,
        chunk_index=index,
        content=f"Result {index}",
        metadata={"source_filename": "guide.txt", "rank_source": index},
        embedding_model="test-model",
        cosine_distance=distance,
        cosine_similarity=1.0 - distance,
    )


def make_service(
    embedding_service: FakeEmbeddingService | None = None,
    repository: RecordingRetrievalStore | None = None,
) -> RetrievalService:
    return RetrievalService(
        embedding_service or FakeEmbeddingService(),
        repository or RecordingRetrievalStore(),
        embedding_model="test-model",
        embedding_dimensions=3,
        default_top_k=3,
        max_top_k=10,
    )


def test_generates_query_embedding_and_searches_repository() -> None:
    embedding_service = FakeEmbeddingService()
    repository = RecordingRetrievalStore([make_result(0, 0.1)])
    service = make_service(embedding_service, repository)

    results = asyncio.run(service.search("How do I reset my password?"))

    assert embedding_service.queries == ["How do I reset my password?"]
    assert results == repository.results
    assert repository.calls[0]["query_embedding"] == (1.0, 0.0, 0.0)


def test_respects_top_k_and_forwards_filters() -> None:
    repository = RecordingRetrievalStore(
        [make_result(0, 0.0), make_result(1, 0.2), make_result(2, 0.8)]
    )
    service = make_service(repository=repository)

    results = asyncio.run(
        service.search(
            "password",
            top_k=2,
            document_id=DOCUMENT_ID,
            embedding_model="test-model",
        )
    )

    assert len(results) == 2
    assert repository.calls[0]["top_k"] == 2
    assert repository.calls[0]["document_id"] == DOCUMENT_ID
    assert repository.calls[0]["embedding_model"] == "test-model"


def test_preserves_order_metadata_and_explicit_score_semantics() -> None:
    expected = [make_result(0, 0.0), make_result(1, 0.25), make_result(2, 1.0)]
    service = make_service(repository=RecordingRetrievalStore(expected))

    results = asyncio.run(service.search("password"))

    assert [result.chunk_index for result in results] == [0, 1, 2]
    assert results[0].metadata["source_filename"] == "guide.txt"
    assert [result.cosine_distance for result in results] == [0.0, 0.25, 1.0]
    assert [result.cosine_similarity for result in results] == [1.0, 0.75, 0.0]


def test_rejects_empty_query_before_embedding() -> None:
    embedding_service = FakeEmbeddingService()
    service = make_service(embedding_service=embedding_service)

    with pytest.raises(EmptyQueryError, match="cannot be empty"):
        asyncio.run(service.search("  \n"))

    assert embedding_service.queries == []


@pytest.mark.parametrize("top_k", [0, -1, 11])
def test_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(InvalidTopKError, match="between 1 and 10"):
        asyncio.run(make_service().search("password", top_k=top_k))


def test_rejects_query_vector_dimension_mismatch() -> None:
    service = make_service(FakeEmbeddingService((1.0, 0.0)))

    with pytest.raises(QueryEmbeddingDimensionError, match="dimensions"):
        asyncio.run(service.search("password"))


def test_rejects_embedding_model_mismatch_before_embedding() -> None:
    embedding_service = FakeEmbeddingService()
    service = make_service(embedding_service=embedding_service)

    with pytest.raises(QueryEmbeddingModelError, match="model"):
        asyncio.run(
            service.search("password", embedding_model="different-model")
        )

    assert embedding_service.queries == []
