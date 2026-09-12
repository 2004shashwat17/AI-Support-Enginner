from app.memory.models import ConversationState
from app.memory.service import ConversationMemoryService
from app.memory.store import InMemoryConversationStore


# New conversation.
def test_start_or_resume_creates_fresh_state_for_unseen_conversation() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())

    state = service.start_or_resume("conv-1")

    assert state.conversation_id == "conv-1"
    assert state.messages == []


# Continuing conversation / multiple turns.
def test_record_turn_appends_user_and_assistant_messages() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())
    state = service.start_or_resume("conv-1")

    service.record_turn(state, user_message="Hi", assistant_message="Hello!")
    resumed = service.start_or_resume("conv-1")

    assert [m.role for m in resumed.messages] == ["user", "assistant"]
    assert [m.content for m in resumed.messages] == ["Hi", "Hello!"]


def test_multiple_turns_accumulate_in_order() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())
    state = service.start_or_resume("conv-1")

    service.record_turn(state, user_message="Q1", assistant_message="A1")
    state = service.start_or_resume("conv-1")
    service.record_turn(state, user_message="Q2", assistant_message="A2")

    resumed = service.start_or_resume("conv-1")
    assert [m.content for m in resumed.messages] == ["Q1", "A1", "Q2", "A2"]


# Separate conversations / session isolation.
def test_separate_conversations_do_not_share_history() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())
    state_a = service.start_or_resume("conv-a")
    service.record_turn(state_a, user_message="Hello from A", assistant_message="Hi A")

    state_b = service.start_or_resume("conv-b")

    assert state_b.messages == []


# Missing conversation lookup.
def test_history_for_unknown_conversation_returns_empty_list() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())

    assert service.history("never-seen") == []


# History truncation / bounded history.
def test_history_is_truncated_to_max_messages() -> None:
    store = InMemoryConversationStore(max_history_messages=4)
    service = ConversationMemoryService(store)
    state = service.start_or_resume("conv-1")

    for turn in range(5):
        state = service.start_or_resume("conv-1")
        service.record_turn(state, user_message=f"Q{turn}", assistant_message=f"A{turn}")

    resumed = service.start_or_resume("conv-1")
    assert len(resumed.messages) == 4
    # Oldest turns are dropped; the most recent messages are kept.
    assert resumed.messages[-1].content == "A4"


def test_history_limit_parameter_returns_only_the_requested_tail() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())
    state = service.start_or_resume("conv-1")
    for turn in range(3):
        state = service.start_or_resume("conv-1")
        service.record_turn(state, user_message=f"Q{turn}", assistant_message=f"A{turn}")

    limited = service.history("conv-1", limit=2)

    assert [m.content for m in limited] == ["Q2", "A2"]


# State persistence.
def test_state_persists_pending_route_and_slot() -> None:
    service = ConversationMemoryService(InMemoryConversationStore())
    state = service.start_or_resume("conv-1")
    state.pending_route = "order"
    state.pending_slot = "order_id"
    state.customer_id = "cust-1001"
    service.record_turn(state, user_message="Where is my order?", assistant_message="Please share your order ID.")

    resumed = service.start_or_resume("conv-1")

    assert resumed.pending_route == "order"
    assert resumed.pending_slot == "order_id"
    assert resumed.customer_id == "cust-1001"


# Invalid/corrupted state handling.
class MismatchedStore:
    def get(self, conversation_id: str) -> ConversationState | None:
        return ConversationState(conversation_id="different-id")

    def save(self, state: ConversationState) -> None:
        pass

    def delete(self, conversation_id: str) -> None:
        pass


def test_mismatched_stored_state_falls_back_to_a_fresh_conversation() -> None:
    service = ConversationMemoryService(MismatchedStore())

    state = service.start_or_resume("conv-1")

    assert state.conversation_id == "conv-1"
    assert state.messages == []


class FailingStore:
    def get(self, conversation_id: str) -> None:
        raise RuntimeError("store unavailable")

    def save(self, state: ConversationState) -> None:
        raise RuntimeError("store unavailable")

    def delete(self, conversation_id: str) -> None:
        raise RuntimeError("store unavailable")


def test_store_read_failure_falls_back_to_a_fresh_conversation() -> None:
    service = ConversationMemoryService(FailingStore())

    state = service.start_or_resume("conv-1")

    assert state.conversation_id == "conv-1"


def test_store_write_failure_does_not_raise() -> None:
    service = ConversationMemoryService(FailingStore())
    state = service.start_or_resume("conv-1")

    service.record_turn(state, user_message="hi", assistant_message="hello")  # no raise


def test_rejects_non_positive_max_history() -> None:
    import pytest

    with pytest.raises(ValueError):
        InMemoryConversationStore(max_history_messages=0)
