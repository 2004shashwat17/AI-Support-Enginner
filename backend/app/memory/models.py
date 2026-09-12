"""Short-term conversation memory models.

IMPORTANT: this stores conversational *context* only (message history, the
identity a conversation is authenticated as, and which slot of information
is still pending). It never stores authoritative customer/order/refund
facts -- those are always fetched fresh from `SupportToolService` on every
turn (see docs/support-tools.md and docs/memory.md).
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


MessageRole = Literal["user", "assistant"]


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1, max_length=8000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, max_length=128)
    customer_id: str | None = None
    pending_route: str | None = None
    pending_slot: str | None = None
    pending_order_id: str | None = None
    consecutive_clarifications: int = 0
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
