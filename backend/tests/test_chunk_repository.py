from unittest.mock import Mock
from uuid import UUID

import pytest

from app.db.repositories.chunks import (
    ChunkRepository,
    InvalidReplacementScopeError,
    InvalidStoredEmbeddingError,
)
from app.services.embeddings import EmbeddedChunk


def make_embedded_chunk(
    *,
    embedding: tuple[float, ...] = (0.1, 0.2, 0.3),
    embedding_model: str = "test-model",
    content: str = "Reset your password from account settings.",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=2),
        chunk_index=0,
        content=content,
        embedding=embedding,
        embedding_model=embedding_model,
        metadata={"source_filename": "guide.txt"},
    )


def make_repository() -> ChunkRepository:
    return ChunkRepository(
        Mock(),
        embedding_model="test-model",
        embedding_dimensions=3,
    )


def test_rejects_invalid_vector_dimensions_before_database_write() -> None:
    repository = make_repository()

    with pytest.raises(InvalidStoredEmbeddingError, match="dimensions"):
        repository.save([make_embedded_chunk(embedding=(0.1, 0.2))])


def test_rejects_wrong_embedding_model_before_database_write() -> None:
    repository = make_repository()

    with pytest.raises(InvalidStoredEmbeddingError, match="model"):
        repository.save([make_embedded_chunk(embedding_model="other-model")])


def test_rejects_empty_content_before_database_write() -> None:
    repository = make_repository()

    with pytest.raises(InvalidStoredEmbeddingError, match="content"):
        repository.save([make_embedded_chunk(content="  ")])


def test_replacement_rejects_chunks_outside_explicit_document_scope() -> None:
    session = Mock()
    repository = ChunkRepository(
        session,
        embedding_model="test-model",
        embedding_dimensions=3,
    )

    with pytest.raises(InvalidReplacementScopeError, match="exactly match"):
        repository.replace_documents(
            {UUID(int=999)},
            [make_embedded_chunk()],
        )

    session.execute.assert_not_called()
    session.commit.assert_not_called()
