"""Pydantic schemas for application-owned support tools.

These tools are the only way the LLM can touch customer/order/ticket data.
The LLM supplies validated arguments (e.g. `order_id`); it never sees or
generates SQL and never supplies the authenticated actor identity -- that
comes from `AuthContext`, which the application controls.

This data is intentionally separate from the knowledge-base documents used
by RAG (see app/db/models.py) -- support/customer/order records never enter
the vector store, and knowledge-base chunks never contain customer data.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class RefundStatus(StrEnum):
    NONE = "none"
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    REFUNDED = "refunded"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class AuthContext(BaseModel):
    """Identifies the authenticated actor a tool call is performed on behalf of.

    This must be supplied by the application's session/auth layer, never by
    the LLM: the LLM chooses *which* tool to call and *which arguments* to
    pass, but never *who* is calling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str = Field(min_length=1, max_length=64)


class CustomerLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=64)


class CustomerRecord(BaseModel):
    customer_id: str
    full_name: str
    email: str
    created_at: datetime


class OrderLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=1, max_length=64)


class OrderRecord(BaseModel):
    order_id: str
    customer_id: str
    status: OrderStatus
    total_amount_cents: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    placed_at: datetime


class OrderStatusResult(BaseModel):
    order_id: str
    status: OrderStatus


class RefundStatusResult(BaseModel):
    order_id: str
    refund_status: RefundStatus
    refund_amount_cents: int | None = Field(default=None, ge=0)


class CreateTicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str | None = Field(default=None, max_length=64)
    subject: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    priority: TicketPriority = TicketPriority.NORMAL


class SupportTicket(BaseModel):
    ticket_id: str
    customer_id: str
    order_id: str | None
    subject: str
    description: str
    priority: TicketPriority
    status: TicketStatus
    created_at: datetime
