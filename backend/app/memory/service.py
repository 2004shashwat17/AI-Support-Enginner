"""Conversation memory service used by the support agent.

Separates SHORT-TERM CONVERSATION MEMORY (this module: message history,
which identity/slot a conversation is currently resolving) from
CUSTOMER/ORDER DATA (`app/tools/`: always fetched fresh from the
authoritative tool/repository layer, never read out of conversation memory).
"""

import logging

from app.memory.models import ConversationMessage, ConversationState
from app.memory.store import ConversationStore, InMemoryConversationStore


logger = logging.getLogger(__name__)


class ConversationMemoryService:
    def __init__(self, store: ConversationStore | None = None) -> None:
        self._store = store or InMemoryConversationStore()

    def start_or_resume(self, conversation_id: str) -> ConversationState:
        try:
            state = self._store.get(conversation_id)
        except Exception:
            logger.warning(
                "Conversation memory read failed; starting a fresh conversation.",
                exc_info=True,
            )
            state = None

        if state is None:
            return ConversationState(conversation_id=conversation_id)

        if state.conversation_id != conversation_id:
            # Defensive check against a corrupted/mismatched store entry.
            logger.warning(
                "Conversation state id mismatch; starting a fresh conversation."
            )
            return ConversationState(conversation_id=conversation_id)

        return state

    def record_turn(
        self,
        state: ConversationState,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        state.messages.append(ConversationMessage(role="user", content=user_message))
        state.messages.append(
            ConversationMessage(role="assistant", content=assistant_message)
        )
        try:
            self._store.save(state)
        except Exception:
            logger.warning("Conversation memory write failed.", exc_info=True)

    def history(
        self, conversation_id: str, *, limit: int | None = None
    ) -> list[ConversationMessage]:
        state = self.start_or_resume(conversation_id)
        messages = state.messages
        if limit is not None:
            return messages[-limit:]
        return list(messages)

    def delete(self, conversation_id: str) -> None:
        self._store.delete(conversation_id)
