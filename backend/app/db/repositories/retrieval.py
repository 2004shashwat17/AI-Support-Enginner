from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunkRecord
from app.services.document_ingestion import MetadataValue


class RetrievalRepositoryError(RuntimeError):
    """Base exception for vector retrieval failures."""


class InvalidQueryVectorError(RetrievalRepositoryError):
    """Raised when a query vector is incompatible with stored vectors."""


class IncompatibleEmbeddingModelError(RetrievalRepositoryError):
    """Raised when a query uses a different embedding model."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    metadata: dict[str, MetadataValue]
    embedding_model: str
    cosine_distance: float | None = None
    cosine_similarity: float | None = None
    keyword_score: float | None = None
    retrieval_strategy: str = "vector"
    dense_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    original_rank: int | None = None


class RetrievalRepository:
    def __init__(
        self,
        session: Session,
        *,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        self._session = session
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        model_filter = embedding_model or self._embedding_model
        if model_filter != self._embedding_model:
            raise IncompatibleEmbeddingModelError(
                "Query embedding model does not match repository configuration."
            )
        if len(query_embedding) != self._embedding_dimensions:
            raise InvalidQueryVectorError(
                "Query embedding dimensions do not match repository configuration."
            )
        if any(not isfinite(value) for value in query_embedding):
            raise InvalidQueryVectorError(
                "Query embedding must contain only finite numbers."
            )
        if top_k <= 0:
            raise ValueError("Top-K must be greater than zero.")

        distance = KnowledgeChunkRecord.embedding.cosine_distance(
            list(query_embedding)
        ).label("cosine_distance")
        statement = (
            select(KnowledgeChunkRecord, distance)
            .where(KnowledgeChunkRecord.embedding_model == model_filter)
            .order_by(distance.asc())
            .limit(top_k)
        )
        if document_id is not None:
            statement = statement.where(
                KnowledgeChunkRecord.document_id == document_id
            )

        rows = self._session.execute(statement).all()
        return [
            RetrievedChunk(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                chunk_index=record.chunk_index,
                content=record.content,
                metadata=dict(record.chunk_metadata),
                embedding_model=record.embedding_model,
                cosine_distance=float(cosine_distance),
                cosine_similarity=1.0 - float(cosine_distance),
            )
            for record, cosine_distance in rows
        ]
