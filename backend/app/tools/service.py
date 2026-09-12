"""Application-owned support tools.

The LLM never executes SQL or generates arbitrary queries. Its role is
limited to choosing a named tool and supplying validated Pydantic
arguments; this service performs authorization, talks to the repository,
and returns a structured Pydantic result:

    LLM -> validated tool arguments -> SupportToolService -> repository ->
    structured result -> LLM

Authorization is enforced here, in application code, using `AuthContext`
(the authenticated actor) -- never inferred from anything the LLM says.
"""

import uuid
from datetime import datetime, timezone
from typing import Protocol

from app.tools.errors import (
    CustomerNotFoundError,
    OrderNotFoundError,
    ToolRepositoryError,
    UnauthorizedToolAccessError,
)
from app.tools.models import (
    AuthContext,
    CreateTicketRequest,
    CustomerLookupRequest,
    CustomerRecord,
    OrderLookupRequest,
    OrderRecord,
    OrderStatusResult,
    RefundStatus,
    RefundStatusResult,
    SupportTicket,
    TicketStatus,
)
from app.tools.repository import InMemorySupportRepository


class SupportRepository(Protocol):
    def get_customer(self, customer_id: str) -> CustomerRecord | None: ...

    def get_order(self, order_id: str) -> OrderRecord | None: ...

    def get_refund_status(self, order_id: str) -> RefundStatus: ...

    def find_open_ticket(
        self, customer_id: str, subject: str, description: str
    ) -> SupportTicket | None: ...

    def create_ticket(self, ticket: SupportTicket) -> SupportTicket: ...


class SupportToolService:
    def __init__(self, repository: SupportRepository | None = None) -> None:
        self._repository = repository or InMemorySupportRepository()

    def get_customer(
        self, request: CustomerLookupRequest, auth: AuthContext
    ) -> CustomerRecord:
        if request.customer_id != auth.customer_id:
            raise UnauthorizedToolAccessError(
                "Customers may only look up their own record."
            )
        try:
            customer = self._repository.get_customer(request.customer_id)
        except Exception as exc:
            raise ToolRepositoryError("Customer lookup failed.") from exc
        if customer is None:
            raise CustomerNotFoundError(
                f"No customer found for id {request.customer_id!r}."
            )
        return customer

    def get_order(self, request: OrderLookupRequest, auth: AuthContext) -> OrderRecord:
        try:
            order = self._repository.get_order(request.order_id)
        except Exception as exc:
            raise ToolRepositoryError("Order lookup failed.") from exc
        if order is None:
            raise OrderNotFoundError(f"No order found for id {request.order_id!r}.")
        self._authorize_order_access(order, auth)
        return order

    def get_order_status(
        self, request: OrderLookupRequest, auth: AuthContext
    ) -> OrderStatusResult:
        order = self.get_order(request, auth)
        return OrderStatusResult(order_id=order.order_id, status=order.status)

    def get_refund_status(
        self, request: OrderLookupRequest, auth: AuthContext
    ) -> RefundStatusResult:
        order = self.get_order(request, auth)
        try:
            refund_status = self._repository.get_refund_status(order.order_id)
        except Exception as exc:
            raise ToolRepositoryError("Refund status lookup failed.") from exc
        refund_amount = (
            order.total_amount_cents
            if refund_status is RefundStatus.REFUNDED
            else None
        )
        return RefundStatusResult(
            order_id=order.order_id,
            refund_status=refund_status,
            refund_amount_cents=refund_amount,
        )

    def create_support_ticket(
        self, request: CreateTicketRequest, auth: AuthContext
    ) -> SupportTicket:
        if request.order_id is not None:
            order_request = OrderLookupRequest(order_id=request.order_id)
            self.get_order(order_request, auth)

        try:
            existing = self._repository.find_open_ticket(
                auth.customer_id, request.subject, request.description
            )
        except Exception as exc:
            raise ToolRepositoryError("Ticket lookup failed.") from exc
        if existing is not None:
            return existing

        ticket = SupportTicket(
            ticket_id=f"tkt-{uuid.uuid4()}",
            customer_id=auth.customer_id,
            order_id=request.order_id,
            subject=request.subject,
            description=request.description,
            priority=request.priority,
            status=TicketStatus.OPEN,
            created_at=datetime.now(timezone.utc),
        )
        try:
            return self._repository.create_ticket(ticket)
        except Exception as exc:
            raise ToolRepositoryError("Ticket creation failed.") from exc

    def _authorize_order_access(self, order: OrderRecord, auth: AuthContext) -> None:
        if order.customer_id != auth.customer_id:
            raise UnauthorizedToolAccessError(
                "Customers may only access their own orders."
            )
