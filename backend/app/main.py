import logging
import time
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.documents import router as documents_router
from app.api.support import router as support_router
from app.core.config import DatabaseSettings
from app.db.session import create_database_engine
from app.observability.context import new_request_id, set_request_id
from app.observability.metrics import LoggingMetricsSink


logger = logging.getLogger(__name__)
_metrics_sink = LoggingMetricsSink()


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]


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
    """Liveness probe: the process is running. Has no dependencies (no
    database calls), so it never fails just because Postgres or OpenAI are
    unavailable -- that's what /health/ready is for."""
    return HealthResponse(status="ok")


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    tags=["system"],
    responses={503: {"model": ReadinessResponse}},
)
async def readiness_check() -> JSONResponse:
    """Readiness probe: can this instance actually serve traffic right now.

    Checks database connectivity with a lightweight query. Never exposes
    the connection string, credentials, or the underlying exception -- only
    a boolean-ish status.
    """
    try:
        database_settings = DatabaseSettings()
        engine = create_database_engine(database_settings)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except (ValidationError, SQLAlchemyError):
        logger.warning("Readiness check failed: database unavailable")
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    return JSONResponse(status_code=200, content={"status": "ready"})
