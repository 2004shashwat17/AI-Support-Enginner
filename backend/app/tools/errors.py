"""Predictable, typed errors for application-owned support tools.

The LLM never receives raw stack traces or exception internals; callers
(the agent orchestration layer, added in a later step) translate these into
safe, user-facing messages.
"""


class SupportToolError(RuntimeError):
    """Base exception for all support tool failures."""


class CustomerNotFoundError(SupportToolError):
    """Raised when no customer record matches the requested id."""


class OrderNotFoundError(SupportToolError):
    """Raised when no order record matches the requested id."""


class UnauthorizedToolAccessError(SupportToolError):
    """Raised when the authenticated actor does not own the requested resource."""


class ToolRepositoryError(SupportToolError):
    """Raised when the underlying data store fails unexpectedly."""
