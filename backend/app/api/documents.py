from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.services.document_ingestion import (
    DocumentIngestionError,
    DocumentTooLargeError,
    IngestedDocument,
    MetadataValue,
    UnsupportedDocumentTypeError,
    ingest_text_document,
)


class DocumentIngestionResponse(BaseModel):
    document_id: UUID
    filename: str
    metadata: dict[str, MetadataValue]

    @classmethod
    def from_document(cls, document: IngestedDocument) -> "DocumentIngestionResponse":
        return cls(
            document_id=document.document_id,
            filename=document.filename,
            metadata=document.metadata,
        )


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    file: Annotated[UploadFile, File(description="UTF-8 plain-text document")],
) -> DocumentIngestionResponse:
    raw_content = await file.read()

    try:
        document = ingest_text_document(file.filename, raw_content)
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DocumentIngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return DocumentIngestionResponse.from_document(document)
