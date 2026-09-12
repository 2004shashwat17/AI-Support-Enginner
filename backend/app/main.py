from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from app.api.documents import router as documents_router
from app.api.support import router as support_router


class HealthResponse(BaseModel):
    status: Literal["ok"]


app = FastAPI(
    title="AI Support Engineer API",
    version="0.1.0",
)
app.include_router(documents_router)
app.include_router(support_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
