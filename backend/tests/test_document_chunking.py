from uuid import UUID

import pytest

from app.services.document_chunking import (
    EmptyDocumentContentError,
    InvalidChunkingConfigurationError,
    chunk_document,
)
from app.services.document_ingestion import IngestedDocument


DOCUMENT_ID = UUID("86d7f8e8-a5f4-4b8d-9564-1f6db6b1d45a")


def make_document(content: str) -> IngestedDocument:
    return IngestedDocument(
        document_id=DOCUMENT_ID,
        filename="guide.txt",
        content=content,
        metadata={"source_type": "text", "character_count": len(content)},
    )


def test_chunks_normal_document() -> None:
    chunks = chunk_document(make_document("abcdefghij"), chunk_size=4)

    assert [chunk.content for chunk in chunks] == ["abcd", "efgh", "ij"]


def test_returns_one_chunk_for_small_document() -> None:
    chunks = chunk_document(make_document("short"), chunk_size=10)

    assert len(chunks) == 1
    assert chunks[0].content == "short"


def test_returns_one_chunk_when_content_equals_chunk_size() -> None:
    chunks = chunk_document(make_document("exact"), chunk_size=5)

    assert len(chunks) == 1
    assert chunks[0].content == "exact"


def test_rejects_empty_document_content() -> None:
    with pytest.raises(EmptyDocumentContentError, match="no text"):
        chunk_document(make_document("   \n"), chunk_size=10)


def test_applies_chunk_overlap() -> None:
    chunks = chunk_document(
        make_document("abcdefghij"),
        chunk_size=4,
        chunk_overlap=1,
    )

    assert [chunk.content for chunk in chunks] == ["abcd", "defg", "ghij"]


def test_preserves_chunk_order_and_document_id() -> None:
    chunks = chunk_document(make_document("abcdefgh"), chunk_size=3)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.document_id == DOCUMENT_ID for chunk in chunks)
    assert [chunk.metadata["start_character"] for chunk in chunks] == [0, 3, 6]


def test_preserves_source_metadata_and_adds_chunk_metadata() -> None:
    chunk = chunk_document(make_document("hello"), chunk_size=10)[0]

    assert chunk.metadata == {
        "source_type": "text",
        "character_count": 5,
        "source_filename": "guide.txt",
        "start_character": 0,
        "end_character": 5,
        "chunk_character_count": 5,
    }


def test_creates_deterministic_chunk_ids() -> None:
    document = make_document("abcdefgh")

    first_result = chunk_document(document, chunk_size=3)
    second_result = chunk_document(document, chunk_size=3)

    assert [chunk.chunk_id for chunk in first_result] == [
        chunk.chunk_id for chunk in second_result
    ]


def test_supports_chunk_size_of_one() -> None:
    chunks = chunk_document(make_document("abc"), chunk_size=1)

    assert [chunk.content for chunk in chunks] == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (-1, 0, "greater than zero"),
        (4, -1, "cannot be negative"),
        (4, 4, "smaller than chunk size"),
        (4, 5, "smaller than chunk size"),
    ],
)
def test_rejects_invalid_configuration(
    chunk_size: int,
    chunk_overlap: int,
    message: str,
) -> None:
    with pytest.raises(InvalidChunkingConfigurationError, match=message):
        chunk_document(
            make_document("content"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )