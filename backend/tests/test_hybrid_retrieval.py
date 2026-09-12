import asyncio
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.services.hybrid_retrieval import HybridRetriever, reciprocal_rank_fusion


def make_result(index: int, *, strategy: str = "vector") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100),
        chunk_index=index,
        content=f"Result {index}",
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
        retrieval_strategy=strategy,
    )


class RecordingRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
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


def test_rrf_uses_formula_merges_duplicates_and_keeps_single_source_chunks() -> None:
    first, second, third = make_result(0), make_result(1), make_result(2)

    results = reciprocal_rank_fusion(
        [first, second],
        [second, third],
        rrf_constant=60,
        top_k=3,
    )

    assert [result.chunk_id for result in results] == [second.chunk_id, first.chunk_id, third.chunk_id]
    assert results[0].rrf_score == pytest.approx((1 / 62) + (1 / 61))
    assert results[0].dense_rank == 2
    assert results[0].keyword_rank == 1
    assert results[1].rrf_score == pytest.approx(1 / 61)
    assert results[1].keyword_rank is None
    assert results[2].dense_rank is None


def test_rrf_has_deterministic_chunk_id_tie_breaker() -> None:
    later_id = make_result(9)
    earlier_id = make_result(1)

    results = reciprocal_rank_fusion(
        [later_id],
        [earlier_id],
        rrf_constant=60,
        top_k=2,
    )

    assert [result.chunk_id for result in results] == [earlier_id.chunk_id, later_id.chunk_id]


@pytest.mark.parametrize(
    ("dense", "keyword"),
    [([], [make_result(0)]), ([make_result(0)], []), ([], [])],
)
def test_rrf_handles_empty_result_sets(
    dense: list[RetrievedChunk],
    keyword: list[RetrievedChunk],
) -> None:
    results = reciprocal_rank_fusion(dense, keyword, top_k=5)

    assert len(results) == len(dense) + len(keyword)


def test_hybrid_calls_both_retrievers_and_respects_final_top_k() -> None:
    dense = RecordingRetriever([make_result(0), make_result(1)])
    keyword = RecordingRetriever([make_result(1), make_result(2)])
    service = HybridRetriever(
        dense,
        keyword,
        candidate_top_k=4,
        default_top_k=2,
        max_top_k=10,
    )

    results = asyncio.run(
        service.search("reset", top_k=2, document_id=UUID(int=100))
    )

    assert len(results) == 2
    assert [result.chunk_id for result in results] == [UUID(int=2), UUID(int=1)]
    assert dense.calls[0]["top_k"] == 4
    assert keyword.calls[0]["top_k"] == 4
    assert dense.calls[0]["document_id"] == UUID(int=100)
    assert keyword.calls[0]["document_id"] == UUID(int=100)


def test_hybrid_propagates_strategy_failure_without_partial_results() -> None:
    class FailingRetriever(RecordingRetriever):
        async def search(self, *args: object, **kwargs: object) -> list[RetrievedChunk]:
            raise RuntimeError("database unavailable")

    service = HybridRetriever(
        RecordingRetriever([make_result(0)]),
        FailingRetriever([]),
        candidate_top_k=3,
        max_top_k=10,
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        asyncio.run(service.search("reset"))