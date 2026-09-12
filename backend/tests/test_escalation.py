import pytest

from app.escalation.models import CreateEscalationRequest, EscalationReason, EscalationStatus
from app.escalation.repository import InMemoryEscalationRepository
from app.escalation.service import EscalationNotFoundError, EscalationService


def service() -> EscalationService:
    return EscalationService(InMemoryEscalationRepository())


def test_create_escalation_starts_pending_and_not_human_reviewed() -> None:
    record = service().create_escalation(
        CreateEscalationRequest(
            conversation_id="conv-1",
            customer_id="cust-1001",
            reason=EscalationReason.USER_REQUESTED_HUMAN,
            summary="User asked for a human.",
        )
    )

    assert record.status is EscalationStatus.PENDING
    assert record.human_reviewed is False
    assert record.escalation_id.startswith("esc-")


def test_get_escalation_returns_created_record() -> None:
    svc = service()
    created = svc.create_escalation(
        CreateEscalationRequest(reason=EscalationReason.INSUFFICIENT_KNOWLEDGE, summary="No evidence.")
    )

    fetched = svc.get_escalation(created.escalation_id)

    assert fetched.escalation_id == created.escalation_id


def test_get_unknown_escalation_raises_not_found() -> None:
    with pytest.raises(EscalationNotFoundError):
        service().get_escalation("esc-does-not-exist")


def test_list_escalations_filters_by_status() -> None:
    svc = service()
    svc.create_escalation(
        CreateEscalationRequest(reason=EscalationReason.TOOL_FAILURE, summary="Repository down.")
    )

    pending = svc.list_escalations(status=EscalationStatus.PENDING)
    resolved = svc.list_escalations(status=EscalationStatus.RESOLVED)

    assert len(pending) == 1
    assert resolved == []
