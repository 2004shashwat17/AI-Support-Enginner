from app.agent.router import classify_intent, extract_order_id


def test_extracts_order_id_case_insensitively() -> None:
    assert extract_order_id("What is the status of ord-2001?") == "ord-2001"
    assert extract_order_id("Check ORD-99") == "ord-99"
    assert extract_order_id("No order id here") is None


def test_classifies_escalation_before_anything_else() -> None:
    assert classify_intent("I want to speak to a human about my refund") == "escalation"


def test_classifies_refund_intent() -> None:
    assert classify_intent("Can I get a refund for order ORD-2001?") == "refund"


def test_classifies_ticket_intent() -> None:
    assert classify_intent("I would like to file a ticket about a broken item.") == "ticket"


def test_classifies_order_intent() -> None:
    assert classify_intent("Where is my order?") == "order"


def test_classifies_customer_intent() -> None:
    assert classify_intent("Can you show me my account details?") == "customer"


def test_defaults_to_knowledge_intent() -> None:
    assert classify_intent("How do I reset my password?") == "knowledge"
