from dataclasses import replace
from uuid import UUID

from app.db.repositories.retrieval import RetrievedChunk
from app.services.retrieval import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    Retriever,
    validate_retrieval_request,
)


DEFAULT_RRF_CONSTANT = 60
DEFAULT_CANDIDATE_TOP_K = 20


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    *,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
    top_k: int,
) -> list[RetrievedChunk]:
    if rrf_constant <= 0:
        raise ValueError("RRF constant must be greater than zero.")
    if top_k <= 0:
        raise ValueError("Top-K must be greater than zero.")

    chunks: dict[UUID, RetrievedChunk] = {}
    dense_ranks: dict[UUID, int] = {}
    keyword_ranks: dict[UUID, int] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        chunks.setdefault(chunk.chunk_id, chunk)
        dense_ranks.setdefault(chunk.chunk_id, rank)
    for rank, chunk in enumerate(keyword_results, start=1):
        chunks.setdefault(chunk.chunk_id, chunk)
        keyword_ranks.setdefault(chunk.chunk_id, rank)

    fused: list[RetrievedChunk] = []
    for chunk_id, chunk in chunks.items():
        dense_rank = dense_ranks.get(chunk_id)
        keyword_rank = keyword_ranks.get(chunk_id)
        score = sum(
            1.0 / (rrf_constant + rank)
            for rank in (dense_rank, keyword_rank)
            if rank is not None
        )
        fused.append(
            replace(
                chunk,
                retrieval_strategy="hybrid",
                dense_rank=dense_rank,
                keyword_rank=keyword_rank,
                rrf_score=score,
            )
        )

    return sorted(
        fused,
        key=lambda chunk: (
            -(chunk.rrf_score or 0.0),
            min(
                rank
                for rank in (chunk.dense_rank, chunk.keyword_rank)
                if rank is not None
            ),
            str(chunk.chunk_id),
        ),
    )[:top_k]


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        keyword_retriever: Retriever,
        *,
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        default_top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
    ) -> None:
        if rrf_constant <= 0:
            raise ValueError("RRF constant must be greater than zero.")
        if not 1 <= candidate_top_k <= max_top_k:
            raise ValueError("Candidate Top-K must be within the supported range.")
        if not 1 <= default_top_k <= max_top_k:
            raise ValueError("Default Top-K must be within the supported range.")

        self._dense_retriever = dense_retriever
        self._keyword_retriever = keyword_retriever
        self._rrf_constant = rrf_constant
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
        candidate_top_k = max(requested_top_k, self._candidate_top_k)
        dense_results = await self._dense_retriever.search(
            query,
            top_k=candidate_top_k,
            document_id=document_id,
            embedding_model=embedding_model,
        )
        keyword_results = await self._keyword_retriever.search(
            query,
            top_k=candidate_top_k,
            document_id=document_id,
            embedding_model=embedding_model,
        )
        return reciprocal_rank_fusion(
            dense_results,
            keyword_results,
            rrf_constant=self._rrf_constant,
            top_k=requested_top_k,
        )