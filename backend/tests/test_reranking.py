import asyncio
from uuid import UUID

import pytest

from app.db.repositories.retrieval import RetrievedChunk
from app.services.reranking import (
    InvalidRerankRequestError,
    LexicalOverlapRerankProvider,
    Reranker,
    RerankingRetriever,
)


def make_chunk(index: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=UUID(int=100),
        chunk_index=index,
        content=content,
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
    )


class StubProvider:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return self.scores


class FailingProvider:
    async def score(self, query: str, documents: list[str]) -> list[float]:
        raise RuntimeError("provider timeout")


class WrongCountProvider:
    async def score(self, query: str, documents: list[str]) -> list[float]:
        return [1.0]


class NonFiniteProvider:
    async def score(self, query: str, documents: list[str]) -> list[float]:
        return [float("nan") for _ in documents]


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
        self.calls.append({"query": query, "top_k": top_k})
        return self.results[:top_k]


class FailingRetriever:
    async def search(self, *args: object, **kwargs: object) -> list[RetrievedChunk]:
        raise ConnectionError("database unavailable")


# 1. Reranker interface / basic reordering.
def test_reranker_reorders_by_score() -> None:
    chunks = [make_chunk(0, "irrelevant"), make_chunk(1, "very relevant")]
    reranker = Reranker(StubProvider([0.1, 0.9]))

    results = asyncio.run(reranker.rerank("query", chunks, top_k=2))

    assert [chunk.chunk_id for chunk in results] == [chunks[1].chunk_id, chunks[0].chunk_id]
    assert results[0].rerank_score == 0.9
    assert results[0].original_rank == 2


# 2 & 3. Candidate selection / candidate_k validation via RerankingRetriever.
def test_reranking_retriever_requests_candidate_top_k_from_base() -> None:
    base = RecordingRetriever([make_chunk(i, f"content {i}") for i in range(5)])
    reranker = Reranker(StubProvider([float(i) for i in range(5)]))
    retriever = RerankingRetriever(base, reranker, candidate_top_k=5, default_top_k=2)

    results = asyncio.run(retriever.search("query"))

    assert base.calls[0]["top_k"] == 5
    assert len(results) == 2


def test_reranking_retriever_rejects_candidate_k_less_than_default_top_k() -> None:
    with pytest.raises(InvalidRerankRequestError):
        RerankingRetriever(
            RecordingRetriever([]),
            Reranker(StubProvider([])),
            candidate_top_k=2,
            default_top_k=5,
        )


def test_reranking_retriever_rejects_requested_top_k_above_candidate_k() -> None:
    retriever = RerankingRetriever(
        RecordingRetriever([make_chunk(0, "a")]),
        Reranker(StubProvider([1.0])),
        candidate_top_k=3,
        default_top_k=2,
    )

    with pytest.raises(InvalidRerankRequestError):
        asyncio.run(retriever.search("query", top_k=10))


# 4. final top_k validation.
def test_reranker_rejects_non_positive_top_k() -> None:
    reranker = Reranker(StubProvider([1.0]))

    with pytest.raises(InvalidRerankRequestError):
        asyncio.run(reranker.rerank("query", [make_chunk(0, "a")], top_k=0))


# 5. Correct reranking order already covered above; add multi-item case.
def test_reranker_truncates_to_top_k_after_sorting() -> None:
    chunks = [make_chunk(i, f"content {i}") for i in range(4)]
    reranker = Reranker(StubProvider([0.2, 0.9, 0.4, 0.1]))

    results = asyncio.run(reranker.rerank("query", chunks, top_k=2))

    assert [chunk.chunk_id for chunk in results] == [chunks[1].chunk_id, chunks[2].chunk_id]


# 6. Tie-breaking: equal scores fall back to original_rank then chunk_id.
def test_reranker_breaks_ties_by_original_rank_then_chunk_id() -> None:
    chunks = [make_chunk(2, "a"), make_chunk(0, "b"), make_chunk(1, "c")]
    reranker = Reranker(StubProvider([0.5, 0.5, 0.5]))

    results = asyncio.run(reranker.rerank("query", chunks, top_k=3))

    # All scores equal -> preserve original candidate order (original_rank ascending).
    assert [chunk.chunk_id for chunk in results] == [chunk.chunk_id for chunk in chunks]


# 7. Duplicate candidate handling.
def test_reranker_deduplicates_candidates_by_chunk_id() -> None:
    chunk = make_chunk(0, "a")
    duplicate = make_chunk(0, "a")
    reranker = Reranker(StubProvider([0.5]))

    results = asyncio.run(reranker.rerank("query", [chunk, duplicate], top_k=5))

    assert len(results) == 1


# 8. Empty candidates.
def test_reranker_returns_empty_list_for_no_candidates() -> None:
    reranker = Reranker(StubProvider([]))

    results = asyncio.run(reranker.rerank("query", [], top_k=5))

    assert results == []


# 9 & 10. Reranker failure -> fallback to original order, observable via chunk state.
def test_reranker_falls_back_to_original_order_on_provider_failure() -> None:
    chunks = [make_chunk(0, "a"), make_chunk(1, "b")]
    reranker = Reranker(FailingProvider())

    results = asyncio.run(reranker.rerank("query", chunks, top_k=2))

    assert [chunk.chunk_id for chunk in results] == [chunk.chunk_id for chunk in chunks]
    assert all(chunk.rerank_score is None for chunk in results)


def test_reranker_falls_back_when_score_count_mismatches() -> None:
    chunks = [make_chunk(0, "a"), make_chunk(1, "b")]
    reranker = Reranker(WrongCountProvider())

    results = asyncio.run(reranker.rerank("query", chunks, top_k=2))

    assert [chunk.chunk_id for chunk in results] == [chunk.chunk_id for chunk in chunks]


def test_reranker_falls_back_on_non_finite_scores() -> None:
    chunks = [make_chunk(0, "a"), make_chunk(1, "b")]
    reranker = Reranker(NonFiniteProvider())

    results = asyncio.run(reranker.rerank("query", chunks, top_k=2))

    assert [chunk.chunk_id for chunk in results] == [chunk.chunk_id for chunk in chunks]


# Fewer candidates than requested top_k.
def test_reranker_returns_fewer_results_when_candidates_are_scarce() -> None:
    chunks = [make_chunk(0, "a")]
    reranker = Reranker(StubProvider([0.5]))

    results = asyncio.run(reranker.rerank("query", chunks, top_k=5))

    assert len(results) == 1


# 11 & 13. Integration with a retrieval strategy (generic base retriever) + failure propagation.
def test_reranking_retriever_propagates_base_retriever_failure() -> None:
    retriever = RerankingRetriever(
        FailingRetriever(),
        Reranker(StubProvider([])),
        candidate_top_k=5,
        default_top_k=2,
    )

    with pytest.raises(ConnectionError, match="database unavailable"):
        asyncio.run(retriever.search("query"))


def test_reranking_retriever_end_to_end_reorders_candidates() -> None:
    chunks = [make_chunk(0, "unrelated topic"), make_chunk(1, "password reset settings")]
    base = RecordingRetriever(chunks)
    retriever = RerankingRetriever(
        base,
        Reranker(LexicalOverlapRerankProvider()),
        candidate_top_k=2,
        default_top_k=2,
    )

    results = asyncio.run(retriever.search("password reset"))

    assert results[0].chunk_id == chunks[1].chunk_id
