import asyncio
from uuid import UUID

from app.agent.graph import SupportAgent
from app.models.rag import Citation, RAGResponse
from app.services.guardrails import EvidenceStatus
from app.services.rag import INSUFFICIENT_KNOWLEDGE_ANSWER, MalformedRAGResponseError
from app.tools.repository import InMemorySupportRepository
from app.tools.service import SupportToolService


class FakeRAGService:
    def __init__(self, response: RAGResponse | None = None, *, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.questions: list[str] = []

    async def answer(self, question: str, *, top_k=None, document_id=None) -> RAGResponse:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def make_agent(rag_service: FakeRAGService | None = None) -> SupportAgent:
    rag = rag_service or FakeRAGService(
        RAGResponse(
            answer="Reset it from Settings.",
            citations=[],
            retrieved_chunks=[],
            evidence_status=EvidenceStatus.SUFFICIENT_EVIDENCE,
        )
    )
    tools = SupportToolService(InMemorySupportRepository())
    return SupportAgent(rag, tools)  # type: ignore[arg-type]


# 1. Knowledge question path.
def test_knowledge_question_routes_to_rag_service() -> None:
    rag = FakeRAGService(
        RAGResponse(
            answer="Reset it from Settings.",
            citations=[
                Citation(chunk_id=UUID(int=1), document_id=UUID(int=100), chunk_index=0, metadata={})
            ],
            retrieved_chunks=[],
            evidence_status=EvidenceStatus.SUFFICIENT_EVIDENCE,
        )
    )
    agent = make_agent(rag)

    response = asyncio.run(agent.handle("How do I reset my password?"))

    assert response.route == "knowledge"
    assert response.answer == "Reset it from Settings."
    assert len(response.citations) == 1
    assert rag.questions == ["How do I reset my password?"]


# 2. Customer lookup path.
def test_customer_question_routes_to_customer_tool() -> None:
    agent = make_agent()

    response = asyncio.run(agent.handle("Show me my account details", customer_id="cust-1001"))

    assert response.route == "customer"
    assert "Ava Thompson" in response.answer


# 3. Order status path.
def test_order_question_routes_to_order_tool() -> None:
    agent = make_agent()

    response = asyncio.run(
        agent.handle("Where is my order ORD-2001?", customer_id="cust-1001")
    )

    assert response.route == "order"
    assert "shipped" in response.answer.lower()


# 4. Refund status path.
def test_refund_question_routes_to_refund_tool() -> None:
    agent = make_agent()

    response = asyncio.run(
        agent.handle("What is my refund status for ORD-2003?", customer_id="cust-1001")
    )

    assert response.route == "refund"
    assert "refunded" in response.answer.lower()


# 5. Ticket creation path.
def test_ticket_request_routes_to_ticket_tool() -> None:
    agent = make_agent()

    response = asyncio.run(
        agent.handle("I would like to file a ticket, my item arrived broken.", customer_id="cust-1001")
    )

    assert response.route == "ticket"
    assert "created" in response.answer.lower()


# 6. Clarification path (order question missing an order ID).
def test_order_question_without_order_id_requests_clarification() -> None:
    agent = make_agent()

    response = asyncio.run(agent.handle("Where is my order?", customer_id="cust-1001"))

    assert response.route == "clarification"
    assert response.needs_clarification is True


# 6b. Clarification path (customer question missing customer_id).
def test_customer_question_without_customer_id_requests_clarification() -> None:
    agent = make_agent()

    response = asyncio.run(agent.handle("Show me my account details"))

    assert response.route == "customer"
    assert response.needs_clarification is True


# 7. Insufficient knowledge path.
def test_insufficient_knowledge_is_surfaced_without_fabricating_an_answer() -> None:
    rag = FakeRAGService(
        RAGResponse(
            answer=INSUFFICIENT_KNOWLEDGE_ANSWER,
            citations=[],
            retrieved_chunks=[],
            evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
        )
    )
    agent = make_agent(rag)

    response = asyncio.run(agent.handle("What's the meaning of life?"))

    assert response.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert response.evidence_status == "insufficient_evidence"


# 8. Tool failure path (unauthorized order access).
def test_tool_failure_returns_safe_message_not_raw_exception() -> None:
    agent = make_agent()

    response = asyncio.run(
        agent.handle("Where is my order ORD-2001?", customer_id="cust-1002")
    )

    assert response.route == "order"
    assert "own account" in response.answer.lower() or "own orders" in response.answer.lower()
    assert "unauthorized" not in response.answer.lower()


# 9. Escalation path.
def test_escalation_keyword_routes_to_human_escalation() -> None:
    agent = make_agent()

    response = asyncio.run(agent.handle("I want to speak to a human agent, please."))

    assert response.route == "escalation"
    assert response.escalated is True
    assert response.escalation_reason is not None


# 10. Malformed LLM output path.
def test_malformed_llm_output_is_handled_safely() -> None:
    rag = FakeRAGService(error=MalformedRAGResponseError("citations do not match"))
    agent = make_agent(rag)

    response = asyncio.run(agent.handle("How do I reset my password?"))

    assert response.route == "knowledge"
    assert response.answer
    assert "citations do not match" not in response.answer
    assert response.evidence_status == "generation_failed"


def test_handle_rejects_blank_question() -> None:
    agent = make_agent()

    try:
        asyncio.run(agent.handle("   "))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for blank question")
