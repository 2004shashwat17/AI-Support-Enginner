import asyncio
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.services.keyword_retrieval import KeywordRetriever
from app.services.retrieval import EmptyQueryError, InvalidTopKError


DOCUMENT_ID = UUID(int=100)


def make_result(index: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=DOCUMENT_ID,
        chunk_index=index,
        content=f"Result {index}",
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
        keyword_score=score,
        retrieval_strategy="keyword",
        keyword_rank=index + 1,
    )


class RecordingKeywordStore:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, object]] = []

    def search_keyword(
        self,
        query: str,
        *,
        top_k: int,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "document_id": document_id,
                "embedding_model": embedding_model,
            }
        )
        return self.results[:top_k]


def make_service(store: RecordingKeywordStore | None = None) -> KeywordRetriever:
    return KeywordRetriever(
        store or RecordingKeywordStore(),
        embedding_model="test-model",
        default_top_k=3,
        max_top_k=10,
    )


def test_forwards_query_top_k_and_filters_without_embedding() -> None:
    store = RecordingKeywordStore([make_result(0, 0.8), make_result(1, 0.5)])
    service = make_service(store)

    results = asyncio.run(
        service.search(
            "ERR-42 reset",
            top_k=1,
            document_id=DOCUMENT_ID,
            embedding_model="test-model",
        )
    )

    assert results == [store.results[0]]
    assert store.calls == [
        {
            "query": "ERR-42 reset",
            "top_k": 1,
            "document_id": DOCUMENT_ID,
            "embedding_model": "test-model",
        }
    ]


def test_preserves_keyword_ranking() -> None:
    expected = [make_result(0, 0.8), make_result(1, 0.5)]

    results = asyncio.run(make_service(RecordingKeywordStore(expected)).search("reset"))

    assert [result.chunk_index for result in results] == [0, 1]
    assert [result.keyword_score for result in results] == [0.8, 0.5]


def test_rejects_empty_query_before_repository_call() -> None:
    store = RecordingKeywordStore()

    with pytest.raises(EmptyQueryError, match="cannot be empty"):
        asyncio.run(make_service(store).search(" \n"))

    assert store.calls == []


@pytest.mark.parametrize("top_k", [0, -1, 11])
def test_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(InvalidTopKError, match="between 1 and 10"):
        asyncio.run(make_service().search("reset", top_k=top_k))