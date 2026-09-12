"""Demo in-memory escalation queue.

Deliberately process-local, like `app/tools/repository.py` and
`app/memory/store.py` -- no real ticketing/paging system is integrated.
"""

from app.escalation.models import EscalationRecord, EscalationStatus


class InMemoryEscalationRepository:
    def __init__(self) -> None:
        self._escalations: dict[str, EscalationRecord] = {}

    def create(self, record: EscalationRecord) -> EscalationRecord:
        self._escalations[record.escalation_id] = record
        return record

    def get(self, escalation_id: str) -> EscalationRecord | None:
        return self._escalations.get(escalation_id)

    def list(self, *, status: EscalationStatus | None = None) -> list[EscalationRecord]:
        records = list(self._escalations.values())
        if status is not None:
            records = [record for record in records if record.status == status]
        return records

    def update(self, record: EscalationRecord) -> EscalationRecord:
        self._escalations[record.escalation_id] = record
        return record
