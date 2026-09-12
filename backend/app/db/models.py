from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import DEFAULT_EMBEDDING_DIMENSIONS


class Base(DeclarativeBase):
    pass


class KnowledgeChunkRecord(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        CheckConstraint("chunk_index >= 0", name="ck_knowledge_chunks_index_nonnegative"),
        CheckConstraint(
            "char_length(btrim(content)) > 0",
            name="ck_knowledge_chunks_content_nonempty",
        ),
        CheckConstraint(
            "char_length(btrim(embedding_model)) > 0",
            name="ck_knowledge_chunks_model_nonempty",
        ),
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_knowledge_chunks_document_index",
        ),
        Index("ix_knowledge_chunks_document_id", "document_id"),
        Index("ix_knowledge_chunks_embedding_model", "embedding_model"),
        Index(
            "ix_knowledge_chunks_search_vector_gin",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    chunk_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(DEFAULT_EMBEDDING_DIMENSIONS),
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
