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
    escalation_node,
    make_customer_node,
    make_knowledge_node,
    make_order_node,
    make_refund_node,
    make_ticket_node,
    route_request,
    understand_request,
    validate_result,
)
from app.agent.state import AgentState, SupportAgentResponse
from app.services.rag import RAGService
from app.tools.service import SupportToolService


_ROUTE_NODES = ("knowledge", "customer", "order", "refund", "ticket", "clarification", "escalation")


def build_support_graph(
    rag_service: RAGService,
    tool_service: SupportToolService,
) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("understand_request", understand_request)
    graph.add_node("route_request", route_request)
    graph.add_node("knowledge", make_knowledge_node(rag_service))
    graph.add_node("customer", make_customer_node(tool_service))
    graph.add_node("order", make_order_node(tool_service))
    graph.add_node("refund", make_refund_node(tool_service))
    graph.add_node("ticket", make_ticket_node(tool_service))
    graph.add_node("clarification", clarification_node)
    graph.add_node("escalation", escalation_node)
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
    """Thin, stateless wrapper that compiles the graph once and runs it per request."""

    def __init__(self, rag_service: RAGService, tool_service: SupportToolService) -> None:
        self._graph = build_support_graph(rag_service, tool_service)

    async def handle(
        self,
        question: str,
        *,
        customer_id: str | None = None,
    ) -> SupportAgentResponse:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        initial_state: AgentState = {"question": question, "customer_id": customer_id}
        result: AgentState = await self._graph.ainvoke(initial_state)

        return SupportAgentResponse(
            route=result["route"],
            answer=result.get("final_answer", ""),
            citations=result.get("final_citations") or [],
            evidence_status=str(result.get("evidence_status", "sufficient_evidence")),
            escalated=result.get("escalated", False),
            escalation_reason=result.get("escalation_reason"),
            needs_clarification=result.get("needs_clarification", False),
        )
