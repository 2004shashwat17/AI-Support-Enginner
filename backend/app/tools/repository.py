"""Demo customer/order/ticket data store.

This is a portfolio/demo implementation: fixed in-memory sample records, no
connection to any real company database. It is deliberately kept separate
from the pgvector knowledge-base repositories (`app/db/repositories/`).
"""

from datetime import datetime, timezone

from app.tools.models import (
    CustomerRecord,
    OrderRecord,
    OrderStatus,
    RefundStatus,
    SupportTicket,
    TicketStatus,
)


def _demo_customers() -> dict[str, CustomerRecord]:
    return {
        "cust-1001": CustomerRecord(
            customer_id="cust-1001",
            full_name="Ava Thompson",
            email="ava.thompson@example.com",
            created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        ),
        "cust-1002": CustomerRecord(
            customer_id="cust-1002",
            full_name="Noah Patel",
            email="noah.patel@example.com",
            created_at=datetime(2024, 3, 2, tzinfo=timezone.utc),
        ),
    }


def _demo_orders() -> dict[str, OrderRecord]:
    return {
        "ord-2001": OrderRecord(
            order_id="ord-2001",
            customer_id="cust-1001",
            status=OrderStatus.SHIPPED,
            total_amount_cents=4599,
            currency="USD",
            placed_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        ),
        "ord-2002": OrderRecord(
            order_id="ord-2002",
            customer_id="cust-1002",
            status=OrderStatus.PROCESSING,
            total_amount_cents=1299,
            currency="USD",
            placed_at=datetime(2024, 7, 10, tzinfo=timezone.utc),
        ),
        "ord-2003": OrderRecord(
            order_id="ord-2003",
            customer_id="cust-1001",
            status=OrderStatus.DELIVERED,
            total_amount_cents=8999,
            currency="USD",
            placed_at=datetime(2024, 5, 20, tzinfo=timezone.utc),
        ),
    }


def _demo_refunds() -> dict[str, RefundStatus]:
    return {"ord-2003": RefundStatus.REFUNDED}


class InMemorySupportRepository:
    def __init__(self) -> None:
        self._customers = _demo_customers()
        self._orders = _demo_orders()
        self._refunds = _demo_refunds()
        self._tickets: dict[str, SupportTicket] = {}

    def get_customer(self, customer_id: str) -> CustomerRecord | None:
        return self._customers.get(customer_id)

    def get_order(self, order_id: str) -> OrderRecord | None:
        return self._orders.get(order_id)

    def get_refund_status(self, order_id: str) -> RefundStatus:
        return self._refunds.get(order_id, RefundStatus.NONE)

    def find_open_ticket(
        self,
        customer_id: str,
        subject: str,
        description: str,
    ) -> SupportTicket | None:
        normalized_subject = subject.strip().casefold()
        normalized_description = description.strip().casefold()
        for ticket in self._tickets.values():
            if (
                ticket.customer_id == customer_id
                and ticket.status != TicketStatus.RESOLVED
                and ticket.subject.strip().casefold() == normalized_subject
                and ticket.description.strip().casefold() == normalized_description
            ):
                return ticket
        return None

    def create_ticket(self, ticket: SupportTicket) -> SupportTicket:
        self._tickets[ticket.ticket_id] = ticket
        return ticket
