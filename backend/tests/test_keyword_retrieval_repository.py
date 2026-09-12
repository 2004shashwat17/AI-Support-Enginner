from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from app.db.repositories.keyword_retrieval import KeywordRetrievalRepository
from app.db.repositories.retrieval import IncompatibleEmbeddingModelError


def make_repository(session: Mock | None = None) -> KeywordRetrievalRepository:
    return KeywordRetrievalRepository(session or Mock(), embedding_model="test-model")


def test_rejects_empty_query_and_invalid_top_k() -> None:
    repository = make_repository()

    with pytest.raises(ValueError, match="cannot be empty"):
        repository.search_keyword("  ", top_k=3)
    with pytest.raises(ValueError, match="greater than zero"):
        repository.search_keyword("reset", top_k=0)


def test_rejects_incompatible_embedding_model() -> None:
    with pytest.raises(IncompatibleEmbeddingModelError, match="model"):
        make_repository().search_keyword(
            "reset",
            top_k=3,
            embedding_model="different-model",
        )


def test_builds_safe_ranked_filtered_full_text_query() -> None:
    session = Mock()
    session.execute.return_value.all.return_value = []
    repository = make_repository(session)
    document_id = UUID(int=100)

    repository.search_keyword(
        'ERR-42 "password reset" OR recovery',
        top_k=7,
        document_id=document_id,
        embedding_model="test-model",
    )

    statement = session.execute.call_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "websearch_to_tsquery" in sql
    assert "ts_rank_cd" in sql
    assert "knowledge_chunks.search_vector @@" in sql
    assert "knowledge_chunks.document_id" in sql
    assert "knowledge_chunks.embedding_model" in sql
    assert "keyword_score DESC, knowledge_chunks.chunk_id ASC" in sql
    assert statement._limit_clause.value == 7


def test_maps_database_ranking_to_compatible_results() -> None:
    session = Mock()
    first = SimpleNamespace(
        chunk_id=UUID(int=1),
        document_id=UUID(int=100),
        chunk_index=0,
        content="Reset password",
        chunk_metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
    )
    second = SimpleNamespace(
        chunk_id=UUID(int=2),
        document_id=UUID(int=100),
        chunk_index=1,
        content="Email recovery",
        chunk_metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
    )
    session.execute.return_value.all.return_value = [(first, 0.8), (second, 0.4)]

    results = make_repository(session).search_keyword("reset", top_k=2)

    assert [result.chunk_id for result in results] == [UUID(int=1), UUID(int=2)]
    assert [result.keyword_score for result in results] == [0.8, 0.4]
    assert [result.keyword_rank for result in results] == [1, 2]
    assert all(result.cosine_distance is None for result in results)