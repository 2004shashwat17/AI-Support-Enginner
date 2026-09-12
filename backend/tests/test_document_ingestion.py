from uuid import UUID

import pytest

from app.services.document_ingestion import (
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentError,
    MAX_DOCUMENT_SIZE_BYTES,
    UnsupportedDocumentTypeError,
    ingest_text_document,
)


def test_ingests_valid_text_document() -> None:
    document = ingest_text_document("faq.txt", b"  Shipping details  \n")

    assert isinstance(document.document_id, UUID)
    assert document.filename == "faq.txt"
    assert document.content == "Shipping details"


def test_rejects_empty_document() -> None:
    with pytest.raises(EmptyDocumentError, match="no usable text"):
        ingest_text_document("empty.txt", b"  \n\t")


def test_rejects_unsupported_file_type() -> None:
    with pytest.raises(UnsupportedDocumentTypeError, match="Only .txt"):
        ingest_text_document("policy.pdf", b"Refund policy")


def test_extracts_basic_metadata() -> None:
    document = ingest_text_document("policy.txt", b"Refunds\r\nwithin 30 days")

    assert document.metadata == {
        "source_type": "text",
        "file_extension": ".txt",
        "size_bytes": 23,
        "character_count": 22,
        "line_count": 2,
    }


def test_rejects_non_utf8_content() -> None:
    with pytest.raises(InvalidDocumentError, match="UTF-8"):
        ingest_text_document("invalid.txt", b"\xff")


def test_rejects_documents_over_the_size_limit() -> None:
    oversized_content = b"a" * (MAX_DOCUMENT_SIZE_BYTES + 1)

    with pytest.raises(DocumentTooLargeError, match="must not exceed"):
        ingest_text_document("large.txt", oversized_content)
