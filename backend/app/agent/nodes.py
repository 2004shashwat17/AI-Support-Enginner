"""LangGraph node implementations for the support agent.

Each node reuses existing services (RAGService, SupportToolService) rather
than duplicating retrieval, reranking, guardrail, or tool logic inline.
"""

from app.agent.router import classify_intent, extract_order_id
from app.agent.state import AgentState
from app.services.rag import RAGError, RAGService
from app.tools.errors import (
    CustomerNotFoundError,
    OrderNotFoundError,
    SupportToolError,
    UnauthorizedToolAccessError,
)
from app.tools.models import (
    AuthContext,
    CreateTicketRequest,
    CustomerLookupRequest,
    OrderLookupRequest,
)
from app.tools.service import SupportToolService


MAX_TICKET_DESCRIPTION_LENGTH = 4000
DEFAULT_TICKET_SUBJECT = "Support request from conversation"


def safe_tool_error_message(exc: SupportToolError) -> str:
    """Maps internal tool errors to a message safe to show the end user.

    Never surfaces raw exception text, stack traces, or internal details.
    """
    if isinstance(exc, CustomerNotFoundError):
        return "I could not find a customer record matching that ID."
    if isinstance(exc, OrderNotFoundError):
        return "I could not find an order matching that ID."
    if isinstance(exc, UnauthorizedToolAccessError):
        return "I can only share details for your own account and orders."
    return "The support tool is temporarily unavailable. Please try again shortly."


async def understand_request(state: AgentState) -> AgentState:
    order_id = extract_order_id(state["question"])
    return {**state, "order_id": order_id}


async def route_request(state: AgentState) -> AgentState:
    intent = classify_intent(state["question"])
    if intent in ("order", "refund") and not state.get("order_id"):
        return {
            **state,
            "route": "clarification",
            "clarification_prompt": (
                "Please share your order ID (e.g. ORD-1234) so I can look this up."
            ),
        }
    return {**state, "route": intent}


def make_knowledge_node(rag_service: RAGService):
    async def knowledge_node(state: AgentState) -> AgentState:
        try:
            response = await rag_service.answer(state["question"])
        except RAGError:
            return {
                **state,
                "final_answer": "The support service is temporarily unavailable.",
                "final_citations": [],
                "evidence_status": "generation_failed",
            }
        return {
            **state,
            "rag_response": response,
            "final_answer": response.answer,
            "final_citations": response.citations,
            "evidence_status": str(response.evidence_status),
        }

    return knowledge_node


def make_customer_node(tool_service: SupportToolService):
    async def customer_node(state: AgentState) -> AgentState:
        customer_id = state.get("customer_id")
        if not customer_id:
            return {
                **state,
                "needs_clarification": True,
                "final_answer": "Please provide your customer ID.",
            }
        auth = AuthContext(customer_id=customer_id)
        try:
            customer = tool_service.get_customer(
                CustomerLookupRequest(customer_id=customer_id), auth
            )
        except SupportToolError as exc:
            return {
                **state,
                "final_answer": safe_tool_error_message(exc),
                "tool_error": str(exc),
            }
        return {
            **state,
            "final_answer": f"Customer {customer.full_name} ({customer.email}).",
        }

    return customer_node


def make_order_node(tool_service: SupportToolService):
    async def order_node(state: AgentState) -> AgentState:
        order_id = state.get("order_id")
        customer_id = state.get("customer_id")
        if not order_id:
            return {
                **state,
                "needs_clarification": True,
                "final_answer": "Please share your order ID (e.g. ORD-1234).",
            }
        if not customer_id:
            return {
                **state,
                "needs_clarification": True,
                "final_answer": "Please provide your customer ID to look up this order.",
            }
        auth = AuthContext(customer_id=customer_id)
        try:
            result = tool_service.get_order_status(
                OrderLookupRequest(order_id=order_id), auth
            )
        except SupportToolError as exc:
            return {
                **state,
                "final_answer": safe_tool_error_message(exc),
                "tool_error": str(exc),
            }
        return {
            **state,
            "final_answer": f"Order {result.order_id} status: {result.status.value}.",
        }

    return order_node


def make_refund_node(tool_service: SupportToolService):
    async def refund_node(state: AgentState) -> AgentState:
        order_id = state.get("order_id")
        customer_id = state.get("customer_id")
        if not order_id:
            return {
                **state,
                "needs_clarification": True,
                "final_answer": "Please share your order ID (e.g. ORD-1234).",
            }
        if not customer_id:
            return {
                **state,
                "needs_clarification": True,
                "final_answer": "Please provide your customer ID to check refund status.",
            }
        auth = AuthContext(customer_id=customer_id)
        try:
            result = tool_service.get_refund_status(
                OrderLookupRequest(order_id=order_id), auth
            )
        except SupportToolError as exc:
            return {
                **state,
                "final_answer": safe_tool_error_message(exc),
                "tool_error": str(exc),
            }
        return {
            **state,
            "final_answer": (
                f"Refund status for order {result.order_id}: {result.refund_status.value}."
            ),
        }

    return refund_node


def make_ticket_node(tool_service: SupportToolService):
    async def ticket_node(state: AgentState) -> AgentState:
        customer_id = state.get("customer_id")
        if not customer_id:
            return {
                **state,
                "needs_clarification": True,
                "final_answer": "Please provide your customer ID to file a support ticket.",
            }
        auth = AuthContext(customer_id=customer_id)
        request = CreateTicketRequest(
            order_id=state.get("order_id"),
            subject=DEFAULT_TICKET_SUBJECT,
            description=state["question"][:MAX_TICKET_DESCRIPTION_LENGTH],
        )
        try:
            ticket = tool_service.create_support_ticket(request, auth)
        except SupportToolError as exc:
            return {
                **state,
                "final_answer": safe_tool_error_message(exc),
                "tool_error": str(exc),
            }
        return {
            **state,
            "final_answer": (
                f"Support ticket {ticket.ticket_id} created (status: {ticket.status.value})."
            ),
        }

    return ticket_node


async def clarification_node(state: AgentState) -> AgentState:
    return {
        **state,
        "needs_clarification": True,
        "final_answer": state.get("clarification_prompt")
        or "Could you clarify your request?",
    }


async def escalation_node(state: AgentState) -> AgentState:
    return {
        **state,
        "escalated": True,
        "escalation_reason": "The user explicitly requested a human agent.",
        "final_answer": "This request has been escalated to a human support agent.",
    }


async def validate_result(state: AgentState) -> AgentState:
    answer = state.get("final_answer")
    if not answer or not answer.strip():
        return {
            **state,
            "final_answer": "The support service could not process this request.",
            "evidence_status": state.get("evidence_status") or "generation_failed",
        }
    return state
