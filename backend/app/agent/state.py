"""Typed state and public response model for the LangGraph support agent."""

from typing import Literal, TypedDict

from pydantic import BaseModel

from app.models.rag import Citation, RAGResponse


RouteName = Literal[
    "knowledge",
    "customer",
    "order",
    "refund",
    "ticket",
    "clarification",
    "escalation",
]


class AgentState(TypedDict, total=False):
    """Explicit typed graph state. Every field is optional (`total=False`)
    since it is populated progressively as nodes run.
    """

    question: str
    customer_id: str | None
    conversation_id: str | None
    order_id: str | None
    route: RouteName
    pending_route: str | None
    pending_slot: str | None
    pending_order_id: str | None
    rag_response: RAGResponse | None
    tool_error: str | None
    final_answer: str
    final_citations: list[Citation]
    evidence_status: str
    escalated: bool
    escalation_reason: str | None
    escalation_id: str | None
    needs_clarification: bool
    clarification_prompt: str | None


class SupportAgentResponse(BaseModel):
    """Structured, stable public output of `SupportAgent.handle(...)`."""

    route: RouteName
    answer: str
    citations: list[Citation] = []
    evidence_status: str = "sufficient_evidence"
    escalated: bool = False
    escalation_reason: str | None = None
    escalation_id: str | None = None
    needs_clarification: bool = False
