"""Human escalation / human-in-the-loop record models.

IMPORTANT: creating an `EscalationRecord` never means a human has actually
reviewed anything. `status` starts (and, in this project, always remains)
`pending` unless something external explicitly moves it forward -- there is
no real human queue wired up. `human_reviewed` is a separate, explicit flag
so callers can never confuse "an AI decided this needs a human" with "a
human looked at this".
"""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.models.rag import Citation


class EscalationStatus(StrEnum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RESOLVED = "resolved"


class EscalationReason(StrEnum):
    USER_REQUESTED_HUMAN = "user_requested_human"
    INSUFFICIENT_KNOWLEDGE = "insufficient_knowledge"
    TOOL_FAILURE = "tool_failure"
    REPEATED_FAILURE = "repeated_failure"
    SENSITIVE_ISSUE = "sensitive_issue"


class CreateEscalationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = Field(default=None, max_length=128)
    customer_id: str | None = Field(default=None, max_length=64)
    reason: EscalationReason
    summary: str = Field(min_length=1, max_length=2000)
    relevant_citations: list[Citation] = Field(default_factory=list)


class EscalationRecord(BaseModel):
    escalation_id: str
    conversation_id: str | None
    customer_id: str | None
    reason: EscalationReason
    summary: str
    relevant_citations: list[Citation]
    status: EscalationStatus
    human_reviewed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
