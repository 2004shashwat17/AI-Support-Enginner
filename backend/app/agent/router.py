"""Deterministic request understanding and routing.

Kept as small, pure, easily-testable functions rather than an LLM call so
routing is bounded, reproducible, and free of hallucination risk. This is a
simple keyword/pattern-based classifier appropriate for a bounded set of
support intents -- not a general-purpose NLU system.
"""

import re

from app.agent.state import RouteName


_ORDER_ID_PATTERN = re.compile(r"\bord-\d+\b", re.IGNORECASE)

# Checked in priority order: escalation requests win over everything else,
# then refund/ticket/order/customer intents, falling back to knowledge.
_ESCALATION_KEYWORDS = (
    "speak to a human",
    "talk to a human",
    "talk to an agent",
    "human agent",
    "human representative",
    "real person",
    "escalate",
)
_REFUND_KEYWORDS = ("refund",)
_TICKET_KEYWORDS = ("file a ticket", "open a ticket", "support ticket", "complaint", "report an issue")
_ORDER_KEYWORDS = ("order status", "where is my order", "track my order", "track order", "my order")
_CUSTOMER_KEYWORDS = ("my account", "my profile", "account details", "who am i", "my customer record")


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID_PATTERN.search(text)
    return match.group(0).lower() if match else None


def classify_intent(question: str) -> RouteName:
    lowered = question.lower()
    if _matches(lowered, _ESCALATION_KEYWORDS):
        return "escalation"
    if _matches(lowered, _REFUND_KEYWORDS):
        return "refund"
    if _matches(lowered, _TICKET_KEYWORDS):
        return "ticket"
    if _matches(lowered, _ORDER_KEYWORDS):
        return "order"
    if _matches(lowered, _CUSTOMER_KEYWORDS):
        return "customer"
    return "knowledge"


def _matches(lowered_text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in lowered_text for keyword in keywords)
