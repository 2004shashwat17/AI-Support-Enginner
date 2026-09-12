from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunkRecord
from app.db.repositories.retrieval import (
    IncompatibleEmbeddingModelError,
    RetrievedChunk,
)


class KeywordRetrievalRepository:
    def __init__(self, session: Session, *, embedding_model: str) -> None:
        self._session = session
        self._embedding_model = embedding_model

    def search_keyword(
        self,
        query: str,
        *,
        top_k: int,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k <= 0:
            raise ValueError("Top-K must be greater than zero.")

        model_filter = embedding_model or self._embedding_model
        if model_filter != self._embedding_model:
            raise IncompatibleEmbeddingModelError(
                "Requested embedding model does not match repository configuration."
            )

        search_query = func.websearch_to_tsquery("english", query)
        keyword_score = func.ts_rank_cd(
            KnowledgeChunkRecord.search_vector,
            search_query,
        ).label("keyword_score")
        statement = (
            select(KnowledgeChunkRecord, keyword_score)
            .where(KnowledgeChunkRecord.search_vector.op("@@")(search_query))
            .where(KnowledgeChunkRecord.embedding_model == model_filter)
            .order_by(keyword_score.desc(), KnowledgeChunkRecord.chunk_id.asc())
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
                keyword_score=float(score),
                retrieval_strategy="keyword",
                keyword_rank=rank,
            )
            for rank, (record, score) in enumerate(rows, start=1)
        ]