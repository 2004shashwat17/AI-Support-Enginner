from unittest.mock import Mock

import pytest

from app.db.repositories.retrieval import (
    IncompatibleEmbeddingModelError,
    InvalidQueryVectorError,
    RetrievalRepository,
)


def make_repository() -> RetrievalRepository:
    return RetrievalRepository(
        Mock(),
        embedding_model="test-model",
        embedding_dimensions=3,
    )


def test_rejects_query_vector_dimension_mismatch() -> None:
    with pytest.raises(InvalidQueryVectorError, match="dimensions"):
        make_repository().search((1.0, 0.0), top_k=3)


def test_rejects_non_finite_query_vector() -> None:
    with pytest.raises(InvalidQueryVectorError, match="finite"):
        make_repository().search((1.0, float("nan"), 0.0), top_k=3)


def test_rejects_incompatible_embedding_model() -> None:
    with pytest.raises(IncompatibleEmbeddingModelError, match="model"):
        make_repository().search(
            (1.0, 0.0, 0.0),
            top_k=3,
            embedding_model="different-model",
        )
