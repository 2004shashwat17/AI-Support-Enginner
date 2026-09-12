from collections.abc import Sequence
from uuid import UUID

from psycopg.errors import UniqueViolation
from sqlalchemy import delete, exists, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import KnowledgeChunkRecord
from app.services.embeddings import EmbeddedChunk


class ChunkRepositoryError(RuntimeError):
    """Base exception for chunk persistence failures."""


class InvalidStoredEmbeddingError(ChunkRepositoryError):
    """Raised when an embedded chunk does not match storage configuration."""


class DuplicateChunkError(ChunkRepositoryError):
    """Raised when a chunk identifier or document position already exists."""


class InvalidReplacementScopeError(ChunkRepositoryError):
    """Raised when replacement chunks do not exactly match the deletion scope."""


class ChunkRepository:
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

    def save(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if not chunks:
            return

        records = [self._to_record(chunk) for chunk in chunks]
        self._session.add_all(records)
        try:
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            if isinstance(exc.orig, UniqueViolation):
                raise DuplicateChunkError(
                    "A chunk ID or document chunk index already exists."
                ) from exc
            raise ChunkRepositoryError("The database rejected the chunks.") from exc

    def get_by_id(self, chunk_id: UUID) -> KnowledgeChunkRecord | None:
        return self._session.get(KnowledgeChunkRecord, chunk_id)

    def exists(self, chunk_id: UUID) -> bool:
        statement = select(exists().where(KnowledgeChunkRecord.chunk_id == chunk_id))
        return bool(self._session.scalar(statement))

    def delete_by_document_id(self, document_id: UUID) -> int:
        result = self._session.execute(
            delete(KnowledgeChunkRecord).where(
                KnowledgeChunkRecord.document_id == document_id
            )
        )
        self._session.commit()
        return result.rowcount or 0

    def replace_documents(
        self,
        document_ids: set[UUID],
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        if not document_ids:
            raise InvalidReplacementScopeError(
                "Document replacement scope cannot be empty."
            )

        chunk_document_ids = {chunk.document_id for chunk in chunks}
        if chunk_document_ids != document_ids:
            raise InvalidReplacementScopeError(
                "Replacement chunks must exactly match the document scope."
            )

        records = [self._to_record(chunk) for chunk in chunks]
        try:
            self._session.execute(
                delete(KnowledgeChunkRecord).where(
                    KnowledgeChunkRecord.document_id.in_(document_ids)
                )
            )
            self._session.add_all(records)
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            if isinstance(exc.orig, UniqueViolation):
                raise DuplicateChunkError(
                    "A chunk ID or document chunk index already exists."
                ) from exc
            raise ChunkRepositoryError("The database rejected the chunks.") from exc

    def _to_record(self, chunk: EmbeddedChunk) -> KnowledgeChunkRecord:
        if chunk.embedding_model != self._embedding_model:
            raise InvalidStoredEmbeddingError(
                "Chunk embedding model does not match repository configuration."
            )
        if len(chunk.embedding) != self._embedding_dimensions:
            raise InvalidStoredEmbeddingError(
                "Chunk embedding dimensions do not match repository configuration."
            )
        if not chunk.content.strip():
            raise InvalidStoredEmbeddingError("Stored chunk content cannot be empty.")

        return KnowledgeChunkRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            embedding=list(chunk.embedding),
            embedding_model=chunk.embedding_model,
            chunk_metadata=dict(chunk.metadata),
        )
