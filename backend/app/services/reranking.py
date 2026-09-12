"""Reranking stage for the retrieval pipeline.

Initial retrieval (dense, keyword, or hybrid) is optimized for recall: it
casts a wide net over a `candidate_top_k` set that may contain the relevant
chunk somewhere in the list. Reranking is optimized for precision: it looks
at the query together with each candidate chunk and reorders them so the
most directly relevant chunk moves toward the top before a smaller
`final top_k` is handed to generation.

Reranking does not always improve quality over the underlying retrieval
ranking; it must be evaluated (see evaluation/runner.py and
scripts/evaluate_rag.py) rather than assumed.

This module keeps the reranker abstraction independent of RAGService so the
implementation (currently a local lexical-overlap heuristic) can be replaced
later with a cross-encoder model or a hosted reranking API without changing
any calling code.
"""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import replace
from math import isfinite
from typing import Protocol
from uuid import UUID

from app.db.repositories.retrieval import RetrievedChunk
from app.services.retrieval import MAX_TOP_K, Retriever, validate_retrieval_request


logger = logging.getLogger(__name__)

DEFAULT_RERANK_CANDIDATE_TOP_K = 20
DEFAULT_RERANK_TOP_K = 5

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class RerankingError(RuntimeError):
    """Base exception for reranking failures."""


class InvalidRerankRequestError(RerankingError, ValueError):
    """Raised when candidate_k/top_k configuration is invalid."""


class RerankProvider(Protocol):
    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, in the same order.

        Implementations may call an external model or API. Callers must not
        assume this call is free of latency or failure.
        """


class LexicalOverlapRerankProvider:
    """Deterministic, local, dependency-free reranking heuristic.

    Scores each candidate by the fraction of query tokens that also appear
    in the candidate content (token-set overlap). This is intentionally
    simple: it has no semantic understanding of synonyms, paraphrasing, or
    word order, and will not outperform a real cross-encoder. It exists so
    the reranking stage can be exercised and evaluated without requiring an
    external model or API key. It is expected to be replaced by a production
    cross-encoder or hosted reranking API later; only the `RerankProvider`
    implementation needs to change.
    """

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query_terms = _tokenize(query)
        if not query_terms:
            return [0.0 for _ in documents]

        scores: list[float] = []
        for document in documents:
            document_terms = _tokenize(document)
            if not document_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & document_terms)
            scores.append(overlap / len(query_terms))
        return scores


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


class Reranker:
    """Reranks candidate chunks for a query using a pluggable provider.

    Ordering rule for ties: candidates are sorted by descending
    `rerank_score`; ties are broken first by ascending `original_rank`
    (the candidate's position in the retrieval result it came from), and
    finally by ascending `chunk_id` string form. This makes ordering
    deterministic across identical runs.

    If the provider fails, times out, or returns scores that cannot be
    trusted (wrong count or non-finite values), reranking falls back to the
    original candidate order (truncated to top_k) and logs a warning. The
    request itself is not failed.
    """

    def __init__(self, provider: RerankProvider) -> None:
        self._provider = provider

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise InvalidRerankRequestError("Top-K must be greater than zero.")
        if not candidates:
            return []

        deduplicated = _deduplicate(candidates)
        ranked_candidates = [
            replace(chunk, original_rank=rank)
            for rank, chunk in enumerate(deduplicated, start=1)
        ]

        start = time.perf_counter()
        try:
            scores = await self._provider.score(
                query, [chunk.content for chunk in ranked_candidates]
            )
            if len(scores) != len(ranked_candidates):
                raise RerankingError(
                    "Reranker returned a different number of scores than candidates."
                )
            if any(not isfinite(score) for score in scores):
                raise RerankingError("Reranker returned a non-finite score.")
        except Exception:
            logger.warning(
                "Reranking failed; falling back to original retrieval order.",
                exc_info=True,
            )
            return ranked_candidates[:top_k]
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("Reranking latency: %.2f ms", elapsed_ms)

        scored_candidates = [
            replace(chunk, rerank_score=score)
            for chunk, score in zip(ranked_candidates, scores, strict=True)
        ]
        scored_candidates.sort(
            key=lambda chunk: (
                -(chunk.rerank_score or 0.0),
                chunk.original_rank or 0,
                str(chunk.chunk_id),
            )
        )
        return scored_candidates[:top_k]


def _deduplicate(candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[UUID] = set()
    deduplicated: list[RetrievedChunk] = []
    for candidate in candidates:
        if candidate.chunk_id in seen:
            continue
        seen.add(candidate.chunk_id)
        deduplicated.append(candidate)
    return deduplicated


class RerankingRetriever:
    """Wraps any `Retriever` with a candidate-retrieval + reranking stage.

    Pipeline: `base_retriever.search(candidate_top_k)` -> `Reranker.rerank(top_k)`.
    Because this class implements the same `search` interface as every other
    retriever in the project, it can wrap vector, keyword, or hybrid
    retrieval interchangeably, and `RAGService` requires no changes to use it.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_top_k: int = DEFAULT_RERANK_CANDIDATE_TOP_K,
        default_top_k: int = DEFAULT_RERANK_TOP_K,
        max_top_k: int = MAX_TOP_K,
    ) -> None:
        if candidate_top_k <= 0:
            raise InvalidRerankRequestError(
                "Candidate Top-K must be greater than zero."
            )
        if not 1 <= default_top_k <= max_top_k:
            raise InvalidRerankRequestError(
                "Default Top-K must be within the supported range."
            )
        if candidate_top_k < default_top_k:
            raise InvalidRerankRequestError(
                "Candidate Top-K must be greater than or equal to the default Top-K."
            )

        self._base_retriever = base_retriever
        self._reranker = reranker
        self._candidate_top_k = candidate_top_k
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
        if self._candidate_top_k < requested_top_k:
            raise InvalidRerankRequestError(
                "Candidate Top-K must be greater than or equal to the requested Top-K."
            )

        start = time.perf_counter()
        candidates = await self._base_retriever.search(
            query,
            top_k=self._candidate_top_k,
            document_id=document_id,
            embedding_model=embedding_model,
        )
        retrieval_latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Candidate retrieval latency: %.2f ms", retrieval_latency_ms)

        return await self._reranker.rerank(query, candidates, top_k=requested_top_k)
