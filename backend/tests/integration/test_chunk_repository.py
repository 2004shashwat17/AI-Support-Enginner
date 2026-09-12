import os
from uuid import UUID

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_EMBEDDING_DIMENSIONS
from app.db.models import KnowledgeChunkRecord
from app.db.repositories.chunks import ChunkRepository, DuplicateChunkError
from app.services.embeddings import EmbeddedChunk


pytestmark = pytest.mark.integration
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


@pytest.fixture
def session() -> Session:
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_engine(TEST_DATABASE_URL)
    with Session(engine) as database_session:
        database_session.execute(delete(KnowledgeChunkRecord))
        database_session.commit()
        yield database_session
    engine.dispose()


def make_chunk(index: int, document_id: UUID = UUID(int=100)) -> EmbeddedChunk:
    embedding = (float(index),) + (0.0,) * (DEFAULT_EMBEDDING_DIMENSIONS - 1)
    return EmbeddedChunk(
        chunk_id=UUID(int=index + 1),
        document_id=document_id,
        chunk_index=index,
        content=f"Knowledge chunk {index}",
        embedding=embedding,
        embedding_model="test-model",
        metadata={"source_filename": "guide.txt", "start_character": index * 10},
    )


def make_repository(session: Session) -> ChunkRepository:
    return ChunkRepository(
        session,
        embedding_model="test-model",
        embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
    )


def test_stores_and_retrieves_one_chunk(session: Session) -> None:
    repository = make_repository(session)
    chunk = make_chunk(0)

    repository.save([chunk])
    stored = repository.get_by_id(chunk.chunk_id)

    assert stored is not None
    assert stored.chunk_id == chunk.chunk_id
    assert stored.document_id == chunk.document_id
    assert stored.content == chunk.content
    assert stored.chunk_metadata == chunk.metadata
    assert stored.embedding_model == "test-model"
    assert list(stored.embedding) == pytest.approx(chunk.embedding)
    assert stored.created_at is not None
    assert repository.exists(chunk.chunk_id)


def test_stores_multiple_chunks(session: Session) -> None:
    repository = make_repository(session)
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]

    repository.save(chunks)

    assert all(repository.exists(chunk.chunk_id) for chunk in chunks)


def test_deletes_chunks_for_document(session: Session) -> None:
    repository = make_repository(session)
    target_document_id = UUID(int=100)
    other_document_id = UUID(int=200)
    repository.save([make_chunk(0, target_document_id)])
    other_chunk = make_chunk(1, other_document_id)
    repository.save([other_chunk])

    deleted_count = repository.delete_by_document_id(target_document_id)

    assert deleted_count == 1
    assert repository.exists(other_chunk.chunk_id)


def test_replaces_document_scope_idempotently_and_preserves_other_rows(
    session: Session,
) -> None:
    repository = make_repository(session)
    target_document_id = UUID(int=100)
    other_chunk = make_chunk(1, UUID(int=200))
    repository.save([make_chunk(0, target_document_id), other_chunk])
    replacement = make_chunk(0, target_document_id)

    repository.replace_documents({target_document_id}, [replacement])
    repository.replace_documents({target_document_id}, [replacement])

    assert repository.get_by_id(replacement.chunk_id) is not None
    assert repository.exists(other_chunk.chunk_id)


def test_rejects_duplicate_chunk_id(session: Session) -> None:
    repository = make_repository(session)
    chunk = make_chunk(0)
    repository.save([chunk])

    with pytest.raises(DuplicateChunkError, match="already exists"):
        repository.save([chunk])
