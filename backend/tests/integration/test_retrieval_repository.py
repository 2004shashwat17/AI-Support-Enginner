import os
from uuid import UUID

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_EMBEDDING_DIMENSIONS
from app.db.models import KnowledgeChunkRecord
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.keyword_retrieval import KeywordRetrievalRepository
from app.db.repositories.retrieval import RetrievalRepository
from app.services.embeddings import EmbeddedChunk


pytestmark = pytest.mark.integration
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
MODEL = "test-model"


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


def vector(first: float, second: float) -> tuple[float, ...]:
    return (first, second) + (0.0,) * (DEFAULT_EMBEDDING_DIMENSIONS - 2)


def make_chunk(
    index: int,
    embedding: tuple[float, ...],
    *,
    document_id: UUID = UUID(int=100),
    content: str | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=UUID(int=document_id.int + index + 1),
        document_id=document_id,
        chunk_index=index,
        content=content or f"Result {index}",
        embedding=embedding,
        embedding_model=MODEL,
        metadata={"source_filename": "guide.txt", "rank_source": index},
    )


def seed_chunks(session: Session) -> None:
    repository = ChunkRepository(
        session,
        embedding_model=MODEL,
        embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    repository.save(
        [
            make_chunk(0, vector(1.0, 0.0)),
            make_chunk(1, vector(1.0, 1.0)),
            make_chunk(2, vector(0.0, 1.0)),
            make_chunk(0, vector(0.9, 0.1), document_id=UUID(int=200)),
        ]
    )


def make_repository(session: Session) -> RetrievalRepository:
    return RetrievalRepository(
        session,
        embedding_model=MODEL,
        embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
    )


def test_orders_results_by_cosine_distance_and_respects_top_k(
    session: Session,
) -> None:
    seed_chunks(session)

    results = make_repository(session).search(
        vector(1.0, 0.0),
        top_k=3,
        document_id=UUID(int=100),
    )

    assert [result.chunk_index for result in results] == [0, 1, 2]
    assert [result.cosine_distance for result in results] == pytest.approx(
        [0.0, 1.0 - (1.0 / (2.0**0.5)), 1.0]
    )
    assert [result.cosine_similarity for result in results] == pytest.approx(
        [1.0, 1.0 / (2.0**0.5), 0.0]
    )
    assert results[0].metadata["source_filename"] == "guide.txt"


def test_filters_by_document_id(session: Session) -> None:
    seed_chunks(session)

    results = make_repository(session).search(
        vector(1.0, 0.0),
        top_k=3,
        document_id=UUID(int=200),
        embedding_model=MODEL,
    )

    assert len(results) == 1
    assert results[0].document_id == UUID(int=200)


def test_keyword_search_ranks_matches_and_applies_filters(session: Session) -> None:
    repository = ChunkRepository(
        session,
        embedding_model=MODEL,
        embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    repository.save(
        [
            make_chunk(
                0,
                vector(1.0, 0.0),
                content="Reset password password from account settings",
            ),
            make_chunk(
                1,
                vector(0.0, 1.0),
                content="Reset password using email recovery",
            ),
            make_chunk(
                0,
                vector(0.9, 0.1),
                document_id=UUID(int=200),
                content="Reset password for another product",
            ),
        ]
    )
    keyword_repository = KeywordRetrievalRepository(session, embedding_model=MODEL)

    results = keyword_repository.search_keyword(
        'password reset OR "email recovery"',
        top_k=2,
        document_id=UUID(int=100),
        embedding_model=MODEL,
    )

    assert len(results) == 2
    assert all(result.document_id == UUID(int=100) for result in results)
    assert all(result.keyword_score is not None for result in results)
    assert [result.keyword_rank for result in results] == [1, 2]
