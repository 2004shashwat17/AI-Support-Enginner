from dataclasses import dataclass
from uuid import UUID, uuid5

from app.services.document_ingestion import IngestedDocument, MetadataValue


class DocumentChunkingError(ValueError):
    """Base exception for documents that cannot be chunked."""


class EmptyDocumentContentError(DocumentChunkingError):
    """Raised when a document has no content to chunk."""


class InvalidChunkingConfigurationError(DocumentChunkingError):
    """Raised when chunk size or overlap cannot produce valid chunks."""


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    metadata: dict[str, MetadataValue]


def chunk_document(
    document: IngestedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int = 0,
) -> list[DocumentChunk]:
    _validate_configuration(chunk_size, chunk_overlap)

    if not document.content.strip():
        raise EmptyDocumentContentError("The document contains no text to chunk.")

    chunks: list[DocumentChunk] = []
    step = chunk_size - chunk_overlap

    for chunk_index, start_character in enumerate(
        range(0, len(document.content), step)
    ):
        end_character = min(start_character + chunk_size, len(document.content))
        chunk_content = document.content[start_character:end_character]
        chunk_id = uuid5(
            document.document_id,
            f"chunk:{chunk_index}:{start_character}:{end_character}",
        )
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata={
                    **document.metadata,
                    "source_filename": document.filename,
                    "start_character": start_character,
                    "end_character": end_character,
                    "chunk_character_count": len(chunk_content),
                },
            )
        )

        if end_character == len(document.content):
            break

    return chunks


def _validate_configuration(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise InvalidChunkingConfigurationError("Chunk size must be greater than zero.")
    if chunk_overlap < 0:
        raise InvalidChunkingConfigurationError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise InvalidChunkingConfigurationError(
            "Chunk overlap must be smaller than chunk size."
        )
