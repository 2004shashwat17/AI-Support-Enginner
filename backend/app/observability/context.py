"""Request correlation identifiers.

A `request_id` is generated per HTTP request (see `app/main.py` middleware)
and a `conversation_id` is set when the support agent handles a
conversation turn. Both are stored in `contextvars` so any code in the call
stack (retrieval, reranking, LLM calls, tools) can attach them to log lines
without threading extra parameters through every function signature.
"""

import contextvars
import uuid


_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_conversation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "conversation_id", default=None
)


def new_request_id() -> str:
    return f"req-{uuid.uuid4()}"


def set_request_id(request_id: str | None) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    return _request_id_var.get()


def set_conversation_id(conversation_id: str | None) -> None:
    _conversation_id_var.set(conversation_id)


def get_conversation_id() -> str | None:
    return _conversation_id_var.get()


def correlation_tags() -> dict[str, str]:
    """Returns the current request/conversation ids as string tags,
    omitting any that are unset. Never includes secrets or PII."""
    tags: dict[str, str] = {}
    request_id = get_request_id()
    if request_id is not None:
        tags["request_id"] = request_id
    conversation_id = get_conversation_id()
    if conversation_id is not None:
        tags["conversation_id"] = conversation_id
    return tags
