from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.dependencies import get_rag_service
from app.models.rag import RAGResponse
from app.services.rag import (
    InvalidRAGQuestionError,
    MalformedRAGResponseError,
    RAGGenerationError,
    RAGRetrievalError,
    RAGService,
)


class SupportQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    document_id: UUID | None = None


router = APIRouter(prefix="/api/v1/support", tags=["support"])


@router.post("/ask", response_model=RAGResponse)
async def ask_support(
    request: SupportQuestionRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> RAGResponse:
    try:
        return await rag_service.answer(
            request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except InvalidRAGQuestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except (RAGRetrievalError, RAGGenerationError, MalformedRAGResponseError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The support service is temporarily unavailable.",
        ) from exc
