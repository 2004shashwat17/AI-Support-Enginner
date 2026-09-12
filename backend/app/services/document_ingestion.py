from pathlib import Path
from typing import TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


MetadataValue: TypeAlias = str | int


class DocumentIngestionError(ValueError):
    """Base exception for rejected documents."""


class UnsupportedDocumentTypeError(DocumentIngestionError):
    """Raised when a document type is not supported."""


class EmptyDocumentError(DocumentIngestionError):
    """Raised when a document has no usable text."""


class InvalidDocumentError(DocumentIngestionError):
    """Raised when a document cannot be safely processed."""


class DocumentTooLargeError(DocumentIngestionError):
    """Raised when a document exceeds the maximum accepted size."""


MAX_DOCUMENT_SIZE_BYTES = 5 * 1024 * 1024  # 5 MiB


class IngestedDocument(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    filename: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, MetadataValue]


def ingest_text_document(filename: str | None, raw_content: bytes) -> IngestedDocument:
    safe_filename = _validate_filename(filename)

    if len(raw_content) > MAX_DOCUMENT_SIZE_BYTES:
        raise DocumentTooLargeError(
            f"Documents must not exceed {MAX_DOCUMENT_SIZE_BYTES} bytes."
        )

    try:
        extracted_text = raw_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidDocumentError("Text documents must use UTF-8 encoding.") from exc

    cleaned_content = _clean_text(extracted_text)
    if not cleaned_content:
        raise EmptyDocumentError("The document contains no usable text.")

    return IngestedDocument(
        filename=safe_filename,
        content=cleaned_content,
        metadata={
            "source_type": "text",
            "file_extension": ".txt",
            "size_bytes": len(raw_content),
            "character_count": len(cleaned_content),
            "line_count": len(cleaned_content.splitlines()),
        },
    )


def _validate_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise InvalidDocumentError("A filename is required.")

    safe_filename = Path(filename.replace("\\", "/")).name
    if Path(safe_filename).suffix.lower() != ".txt":
        raise UnsupportedDocumentTypeError("Only .txt documents are supported.")

    return safe_filename


def _clean_text(content: str) -> str:
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized_content.splitlines()).strip()
