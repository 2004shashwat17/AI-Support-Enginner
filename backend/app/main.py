import logging
import time
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.documents import router as documents_router
from app.api.support import router as support_router
from app.observability.context import new_request_id, set_request_id
from app.observability.metrics import LoggingMetricsSink


logger = logging.getLogger(__name__)
_metrics_sink = LoggingMetricsSink()


class HealthResponse(BaseModel):
    status: Literal["ok"]


app = FastAPI(
    title="AI Support Engineer API",
    version="0.1.0",
)
app.include_router(documents_router)
app.include_router(support_router)


@app.middleware("http")
async def request_correlation_middleware(request: Request, call_next):
    """Assigns a request id, attaches it to the response, and logs total
    HTTP request latency. Never logs request/response bodies (which may
    contain customer questions or PII) -- only method, path, status, and
    timing.
    """
    request_id = new_request_id()
    set_request_id(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _metrics_sink.record_latency(
            "http_request",
            duration_ms,
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        set_request_id(None)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def safe_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ensures unhandled exceptions never leak stack traces, exception
    messages, or internal details to API callers. The exception type and
    request path (never the request body, headers, or exception message,
    which could contain secrets or customer data) are logged server-side
    for debugging.
    """
    logger.exception(
        "Unhandled exception while processing request",
        extra={"path": request.url.path, "exception_type": type(exc).__name__},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."},
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")
