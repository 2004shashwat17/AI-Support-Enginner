"""In-memory conversation store with bounded history and session isolation.

This is a demo-appropriate, process-local store (a dict keyed by
conversation id). It is not shared across processes and is lost on
restart -- adequate for a portfolio project, not for production multi-instance
deployments (see docs/memory.md for the tradeoff discussion).
"""

from datetime import datetime, timezone
from typing import Protocol

from app.memory.models import ConversationState


DEFAULT_MAX_HISTORY_MESSAGES = 20


class ConversationMemoryError(RuntimeError):
    """Base exception for conversation memory failures."""


class ConversationStore(Protocol):
    def get(self, conversation_id: str) -> ConversationState | None: ...

    def save(self, state: ConversationState) -> None: ...

    def delete(self, conversation_id: str) -> None: ...


class InMemoryConversationStore:
    def __init__(self, *, max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES) -> None:
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be greater than zero.")
        self._max_history_messages = max_history_messages
        self._conversations: dict[str, ConversationState] = {}

    def get(self, conversation_id: str) -> ConversationState | None:
        return self._conversations.get(conversation_id)

    def save(self, state: ConversationState) -> None:
        if len(state.messages) > self._max_history_messages:
            state.messages = state.messages[-self._max_history_messages :]
        state.updated_at = datetime.now(timezone.utc)
        self._conversations[state.conversation_id] = state

    def delete(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    @property
    def max_history_messages(self) -> int:
        return self._max_history_messages
