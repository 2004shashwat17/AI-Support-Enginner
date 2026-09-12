"""Bounded LangGraph orchestration for the support agent.

Why LangGraph instead of a single LLM function call:

- The routing decision fans out to genuinely different code paths (RAG
  retrieval vs. structured database tool calls vs. escalation), each with
  its own validation and error handling. A single function call would mix
  these concerns in one large branch of if/else statements.
- LangGraph gives an explicit, typed graph (`AgentState`) and explicit,
  named nodes/edges. The control flow is visible in `build_support_graph`
  rather than implicit inside a large function body, which makes it easier
  to test each node in isolation and to reason about every possible path
  (see tests/test_agent.py).
- The graph is a strict DAG here (no cycles back to the router), so
  execution is bounded by construction, not by a hard iteration cap. This
  keeps behavior deterministic and avoids "agent loops forever" failure
  modes that come from more open-ended agent frameworks.

This module does not duplicate business logic: `knowledge` calls the
existing `RAGService` (retrieval + reranking + guardrails + generation),
and `customer`/`order`/`refund`/`ticket` call the existing
`SupportToolService`.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    clarification_node,
    make_customer_node,
    make_escalation_node,
    make_knowledge_node,
    make_order_node,
    make_refund_node,
    make_ticket_node,
    route_request,
    understand_request,
    validate_result,
)
from app.agent.state import AgentState, SupportAgentResponse
from app.escalation.models import CreateEscalationRequest, EscalationReason
from app.escalation.service import EscalationService
from app.memory.service import ConversationMemoryService
from app.services.rag import RAGService
from app.tools.service import SupportToolService


_ROUTE_NODES = ("knowledge", "customer", "order", "refund", "ticket", "clarification", "escalation")
REPEATED_FAILURE_THRESHOLD = 3


def build_support_graph(
    rag_service: RAGService,
    tool_service: SupportToolService,
    escalation_service: EscalationService,
) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("understand_request", understand_request)
    graph.add_node("route_request", route_request)
    graph.add_node("knowledge", make_knowledge_node(rag_service, escalation_service))
    graph.add_node("customer", make_customer_node(tool_service, escalation_service))
    graph.add_node("order", make_order_node(tool_service, escalation_service))
    graph.add_node("refund", make_refund_node(tool_service, escalation_service))
    graph.add_node("ticket", make_ticket_node(tool_service, escalation_service))
    graph.add_node("clarification", clarification_node)
    graph.add_node("escalation", make_escalation_node(escalation_service))
    graph.add_node("validate_result", validate_result)

    graph.add_edge(START, "understand_request")
    graph.add_edge("understand_request", "route_request")
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["route"],
        {route: route for route in _ROUTE_NODES},
    )
    for route in _ROUTE_NODES:
        graph.add_edge(route, "validate_result")
    graph.add_edge("validate_result", END)

    return graph.compile()


class SupportAgent:
    """Thin, stateless wrapper that compiles the graph once and runs it per request.

    Optionally accepts a `conversation_id` per call to resume short-term
    conversation memory (see docs/memory.md): the identity a conversation is
    authenticated as, and which piece of missing information (e.g. an order
    id) a prior clarification turn was waiting on. Customer/order/refund
    *facts* are still always fetched fresh through `SupportToolService` on
    every turn -- memory never substitutes for that.
    """

    def __init__(
        self,
        rag_service: RAGService,
        tool_service: SupportToolService,
        *,
        memory: ConversationMemoryService | None = None,
        escalation_service: EscalationService | None = None,
    ) -> None:
        self._escalation_service = escalation_service or EscalationService()
        self._graph = build_support_graph(rag_service, tool_service, self._escalation_service)
        self._memory = memory or ConversationMemoryService()

    async def handle(
        self,
        question: str,
        *,
        customer_id: str | None = None,
        conversation_id: str | None = None,
    ) -> SupportAgentResponse:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        conversation_state = None
        pending_route: str | None = None
        pending_slot: str | None = None
        pending_order_id: str | None = None
        effective_customer_id = customer_id

        if conversation_id is not None:
            conversation_state = self._memory.start_or_resume(conversation_id)
            if effective_customer_id is None:
                effective_customer_id = conversation_state.customer_id
            pending_route = conversation_state.pending_route
            pending_slot = conversation_state.pending_slot
            pending_order_id = conversation_state.pending_order_id

        initial_state: AgentState = {
            "question": question,
            "customer_id": effective_customer_id,
            "pending_route": pending_route,
            "pending_slot": pending_slot,
            "pending_order_id": pending_order_id,
        }
        result: AgentState = await self._graph.ainvoke(initial_state)

        response = SupportAgentResponse(
            route=result["route"],
            answer=result.get("final_answer", ""),
            citations=result.get("final_citations") or [],
            evidence_status=str(result.get("evidence_status", "sufficient_evidence")),
            escalated=result.get("escalated", False),
            escalation_reason=result.get("escalation_reason"),
            escalation_id=result.get("escalation_id"),
            needs_clarification=result.get("needs_clarification", False),
        )

        if conversation_state is not None:
            if response.needs_clarification:
                conversation_state.consecutive_clarifications += 1
            else:
                conversation_state.consecutive_clarifications = 0

            if (
                conversation_state.consecutive_clarifications >= REPEATED_FAILURE_THRESHOLD
                and not response.escalated
            ):
                record = self._escalation_service.create_escalation(
                    CreateEscalationRequest(
                        conversation_id=conversation_id,
                        customer_id=effective_customer_id,
                        reason=EscalationReason.REPEATED_FAILURE,
                        summary=(
                            "The conversation could not resolve a request after "
                            f"{conversation_state.consecutive_clarifications} clarification attempts."
                        ),
                    )
                )
                response = response.model_copy(
                    update={
                        "escalated": True,
                        "escalation_reason": record.reason.value,
                        "escalation_id": record.escalation_id,
                        "answer": (
                            "This request has needed repeated clarification, so it has "
                            f"been escalated to a human support agent. Reference: {record.escalation_id} "
                            "(status: pending)."
                        ),
                    }
                )
                conversation_state.consecutive_clarifications = 0

            conversation_state.customer_id = effective_customer_id
            conversation_state.pending_route = (
                result.get("pending_route") if response.needs_clarification else None
            )
            conversation_state.pending_slot = (
                result.get("pending_slot") if response.needs_clarification else None
            )
            conversation_state.pending_order_id = (
                result.get("pending_order_id") if response.needs_clarification else None
            )
            self._memory.record_turn(
                conversation_state,
                user_message=question,
                assistant_message=response.answer,
            )

        return response
