import asyncio
from uuid import UUID

from app.agent.graph import SupportAgent
from app.escalation.models import EscalationStatus
from app.escalation.repository import InMemoryEscalationRepository
from app.escalation.service import EscalationService
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


# Multi-turn conversation memory: the master-prompt example.
#   USER: "What is my order status?"
#   AI:   "Please provide your order ID."
#   USER: "ORD-1234"
def test_conversation_resumes_order_clarification_with_a_bare_order_id() -> None:
    agent = make_agent()

    async def run() -> tuple:
        first = await agent.handle(
            "What is my order status?", customer_id="cust-1001", conversation_id="conv-1"
        )
        second = await agent.handle("ORD-2001", customer_id="cust-1001", conversation_id="conv-1")
        return first, second

    first, second = asyncio.run(run())

    assert first.route == "clarification"
    assert first.needs_clarification is True
    assert second.route == "order"
    assert "shipped" in second.answer.lower()


def test_separate_conversation_ids_do_not_share_pending_state() -> None:
    agent = make_agent()

    async def run() -> tuple:
        await agent.handle(
            "What is my order status?", customer_id="cust-1001", conversation_id="conv-a"
        )
        # A bare order id in an unrelated conversation should not resume conv-a's pending order intent.
        unrelated = await agent.handle("ORD-2001", customer_id="cust-1001", conversation_id="conv-b")
        return unrelated

    unrelated = asyncio.run(run())

    assert unrelated.route == "knowledge"


# Escalation trigger: insufficient knowledge automatically queues a pending escalation.
def test_insufficient_knowledge_automatically_creates_pending_escalation() -> None:
    rag = FakeRAGService(
        RAGResponse(
            answer=INSUFFICIENT_KNOWLEDGE_ANSWER,
            citations=[],
            retrieved_chunks=[],
            evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
        )
    )
    escalation_service = EscalationService(InMemoryEscalationRepository())
    tools = SupportToolService(InMemorySupportRepository())
    agent = SupportAgent(rag, tools, escalation_service=escalation_service)

    response = asyncio.run(agent.handle("What's the meaning of life?"))

    assert response.escalated is True
    assert response.escalation_id is not None
    record = escalation_service.get_escalation(response.escalation_id)
    assert record.status is EscalationStatus.PENDING
    assert record.human_reviewed is False


# Escalation trigger: a tool repository failure prevents resolution -> escalate.
def test_tool_repository_failure_creates_escalation() -> None:
    class FailingRepository:
        def get_customer(self, customer_id: str):
            raise RuntimeError("database unavailable")

        def get_order(self, order_id: str):
            raise RuntimeError("database unavailable")

        def get_refund_status(self, order_id: str):
            raise RuntimeError("database unavailable")

        def find_open_ticket(self, customer_id, subject, description):
            raise RuntimeError("database unavailable")

        def create_ticket(self, ticket):
            raise RuntimeError("database unavailable")

    escalation_service = EscalationService(InMemoryEscalationRepository())
    tools = SupportToolService(FailingRepository())
    rag = FakeRAGService(RAGResponse(answer="unused", citations=[], retrieved_chunks=[]))
    agent = SupportAgent(rag, tools, escalation_service=escalation_service)

    response = asyncio.run(
        agent.handle("Show me my account details", customer_id="cust-1001")
    )

    assert response.escalated is True
    assert response.escalation_id is not None
    assert "runtimeerror" not in response.answer.lower()


# Escalation trigger: repeated clarification failures.
def test_repeated_clarification_failures_trigger_escalation() -> None:
    escalation_service = EscalationService(InMemoryEscalationRepository())
    tools = SupportToolService(InMemorySupportRepository())
    rag = FakeRAGService(
        RAGResponse(answer="unused", citations=[], retrieved_chunks=[])
    )
    agent = SupportAgent(rag, tools, escalation_service=escalation_service)

    async def run():
        responses = []
        for _ in range(3):
            responses.append(
                await agent.handle("Show me my account details", conversation_id="conv-1")
            )
        return responses

    responses = asyncio.run(run())

    assert all(r.needs_clarification for r in responses[:2])
    assert responses[-1].escalated is True
    assert responses[-1].escalation_id is not None

