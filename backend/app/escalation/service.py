"""Human escalation service.

Escalation is triggered (see app/agent/nodes.py) when: the user explicitly
asks for a human, retrieval evidence is insufficient, a tool fails in a way
that prevents resolving the request, or the same conversation repeatedly
fails to resolve (see docs/escalation.md for the full trigger list and the
AI-vs-human-reviewed distinction).
"""

import uuid
from datetime import datetime, timezone
from typing import Protocol

from app.escalation.models import (
    CreateEscalationRequest,
    EscalationRecord,
    EscalationStatus,
)
from app.escalation.repository import InMemoryEscalationRepository
from app.observability.context import correlation_tags
from app.observability.metrics import LoggingMetricsSink, MetricsSink


class EscalationError(RuntimeError):
    """Base exception for escalation failures."""


class EscalationNotFoundError(EscalationError):
    """Raised when no escalation record matches the requested id."""


class EscalationRepository(Protocol):
    def create(self, record: EscalationRecord) -> EscalationRecord: ...

    def get(self, escalation_id: str) -> EscalationRecord | None: ...

    def list(self, *, status: EscalationStatus | None = None) -> list[EscalationRecord]: ...

    def update(self, record: EscalationRecord) -> EscalationRecord: ...


class EscalationService:
    def __init__(
        self,
        repository: EscalationRepository | None = None,
        *,
        metrics_sink: MetricsSink | None = None,
    ) -> None:
        self._repository = repository or InMemoryEscalationRepository()
        self._metrics_sink = metrics_sink or LoggingMetricsSink()

    def create_escalation(self, request: CreateEscalationRequest) -> EscalationRecord:
        now = datetime.now(timezone.utc)
        record = EscalationRecord(
            escalation_id=f"esc-{uuid.uuid4()}",
            conversation_id=request.conversation_id,
            customer_id=request.customer_id,
            reason=request.reason,
            summary=request.summary,
            relevant_citations=request.relevant_citations,
            status=EscalationStatus.PENDING,
            human_reviewed=False,
            created_at=now,
            updated_at=now,
        )
        self._metrics_sink.increment(
            "escalation_created", reason=request.reason.value, **correlation_tags()
        )
        return self._repository.create(record)

    def get_escalation(self, escalation_id: str) -> EscalationRecord:
        record = self._repository.get(escalation_id)
        if record is None:
            raise EscalationNotFoundError(
                f"No escalation found for id {escalation_id!r}."
            )
        return record

    def list_escalations(
        self, *, status: EscalationStatus | None = None
    ) -> list[EscalationRecord]:
        return self._repository.list(status=status)
