import pytest

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
    OrderLookupRequest,
    OrderStatus,
    RefundStatus,
    TicketStatus,
)
from app.tools.repository import InMemorySupportRepository
from app.tools.service import SupportToolService


def service() -> SupportToolService:
    return SupportToolService(InMemorySupportRepository())


def auth(customer_id: str = "cust-1001") -> AuthContext:
    return AuthContext(customer_id=customer_id)


# Valid lookups.
def test_get_customer_returns_own_record() -> None:
    result = service().get_customer(CustomerLookupRequest(customer_id="cust-1001"), auth())

    assert result.customer_id == "cust-1001"
    assert result.full_name == "Ava Thompson"


def test_get_order_returns_owned_order() -> None:
    result = service().get_order(OrderLookupRequest(order_id="ord-2001"), auth())

    assert result.order_id == "ord-2001"
    assert result.status is OrderStatus.SHIPPED


def test_get_order_status_returns_status_only() -> None:
    result = service().get_order_status(OrderLookupRequest(order_id="ord-2002"), auth("cust-1002"))

    assert result.status is OrderStatus.PROCESSING


def test_get_refund_status_returns_refund_details() -> None:
    result = service().get_refund_status(OrderLookupRequest(order_id="ord-2003"), auth())

    assert result.refund_status is RefundStatus.REFUNDED
    assert result.refund_amount_cents == 8999


def test_get_refund_status_defaults_to_none_when_no_refund_recorded() -> None:
    result = service().get_refund_status(OrderLookupRequest(order_id="ord-2001"), auth())

    assert result.refund_status is RefundStatus.NONE
    assert result.refund_amount_cents is None


# Missing customer.
def test_missing_customer_raises_not_found() -> None:
    with pytest.raises(CustomerNotFoundError):
        service().get_customer(CustomerLookupRequest(customer_id="cust-9999"), auth("cust-9999"))


# Missing order.
def test_missing_order_raises_not_found() -> None:
    with pytest.raises(OrderNotFoundError):
        service().get_order(OrderLookupRequest(order_id="ord-9999"), auth())


# Invalid IDs are rejected by Pydantic validation before reaching the service.
def test_empty_customer_id_is_rejected_by_schema() -> None:
    with pytest.raises(ValueError):
        CustomerLookupRequest(customer_id="")


def test_empty_order_id_is_rejected_by_schema() -> None:
    with pytest.raises(ValueError):
        OrderLookupRequest(order_id="")


def test_unknown_request_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        CustomerLookupRequest.model_validate({"customer_id": "cust-1001", "extra": "x"})


# Unauthorized access.
def test_customer_cannot_look_up_another_customers_record() -> None:
    with pytest.raises(UnauthorizedToolAccessError):
        service().get_customer(CustomerLookupRequest(customer_id="cust-1002"), auth("cust-1001"))


def test_customer_cannot_access_another_customers_order() -> None:
    with pytest.raises(UnauthorizedToolAccessError):
        service().get_order(OrderLookupRequest(order_id="ord-2001"), auth("cust-1002"))


def test_customer_cannot_get_refund_status_for_another_customers_order() -> None:
    with pytest.raises(UnauthorizedToolAccessError):
        service().get_refund_status(OrderLookupRequest(order_id="ord-2001"), auth("cust-1002"))


# Ticket creation.
def test_create_support_ticket_succeeds() -> None:
    ticket = service().create_support_ticket(
        CreateTicketRequest(subject="Late delivery", description="My order has not arrived."),
        auth(),
    )

    assert ticket.customer_id == "cust-1001"
    assert ticket.status is TicketStatus.OPEN
    assert ticket.ticket_id.startswith("tkt-")


def test_create_support_ticket_validates_order_ownership() -> None:
    with pytest.raises(UnauthorizedToolAccessError):
        service().create_support_ticket(
            CreateTicketRequest(
                order_id="ord-2001", subject="Issue", description="Something is wrong."
            ),
            auth("cust-1002"),
        )


def test_create_support_ticket_rejects_unknown_order() -> None:
    with pytest.raises(OrderNotFoundError):
        service().create_support_ticket(
            CreateTicketRequest(
                order_id="ord-9999", subject="Issue", description="Something is wrong."
            ),
            auth(),
        )


# Duplicate ticket handling.
def test_duplicate_ticket_returns_existing_ticket_instead_of_creating_a_new_one() -> None:
    svc = service()
    request = CreateTicketRequest(subject="Late delivery", description="My order has not arrived.")

    first = svc.create_support_ticket(request, auth())
    second = svc.create_support_ticket(request, auth())

    assert first.ticket_id == second.ticket_id


def test_different_customers_do_not_share_duplicate_ticket_detection() -> None:
    svc = service()
    request = CreateTicketRequest(subject="Late delivery", description="My order has not arrived.")

    first = svc.create_support_ticket(request, auth("cust-1001"))
    second = svc.create_support_ticket(request, auth("cust-1002"))

    assert first.ticket_id != second.ticket_id


# Database/tool failure.
class FailingRepository:
    def get_customer(self, customer_id: str) -> None:
        raise RuntimeError("database unavailable")

    def get_order(self, order_id: str) -> None:
        raise RuntimeError("database unavailable")

    def get_refund_status(self, order_id: str) -> RefundStatus:
        raise RuntimeError("database unavailable")

    def find_open_ticket(self, customer_id: str, subject: str, description: str) -> None:
        raise RuntimeError("database unavailable")

    def create_ticket(self, ticket: object) -> object:
        raise RuntimeError("database unavailable")


def test_repository_failure_is_wrapped_as_tool_repository_error() -> None:
    svc = SupportToolService(FailingRepository())

    with pytest.raises(ToolRepositoryError):
        svc.get_customer(CustomerLookupRequest(customer_id="cust-1001"), auth())

    with pytest.raises(ToolRepositoryError):
        svc.get_order(OrderLookupRequest(order_id="ord-2001"), auth())
