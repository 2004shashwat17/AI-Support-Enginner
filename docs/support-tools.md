# Support Tools Architecture (Step 13)

## Purpose

Turns the RAG system into an actual support-engineer system by adding
application-owned tools for customer, order, refund, and support-ticket
operations, callable by name with validated arguments. This is a
portfolio/demo system: **no real company database or PII is used.**

## Tool flow

```
LLM
 |  (chooses a tool name + supplies arguments)
 v
Validated Pydantic request (app/tools/models.py)
 |
 v
SupportToolService (app/tools/service.py)  <- AuthContext supplied by the app, never the LLM
 |
 v
SupportRepository (in-memory demo data, app/tools/repository.py)
 |
 v
Structured Pydantic result
 |
 v
LLM
```

**The LLM never executes SQL or generates arbitrary queries.** It can only
select one of the following named operations and supply arguments matching
a strict Pydantic schema (`extra="forbid"`, length-bounded strings):

| Tool | Request | Result |
|---|---|---|
| `get_customer` | `CustomerLookupRequest(customer_id)` | `CustomerRecord` |
| `get_order` | `OrderLookupRequest(order_id)` | `OrderRecord` |
| `get_order_status` | `OrderLookupRequest(order_id)` | `OrderStatusResult` |
| `get_refund_status` | `OrderLookupRequest(order_id)` | `RefundStatusResult` |
| `create_support_ticket` | `CreateTicketRequest(order_id?, subject, description, priority)` | `SupportTicket` |

## Authorization

Every tool call also takes an `AuthContext(customer_id=...)` — the
authenticated actor on whose behalf the tool runs. This **must** come from
the application's session/auth layer (in the current single-turn API, from
the request context established outside the LLM), never from anything the
LLM says or from tool arguments themselves. `SupportToolService` enforces:

- `get_customer`: the requested `customer_id` must equal `auth.customer_id`.
- `get_order` / `get_order_status` / `get_refund_status`: the order's
  `customer_id` must equal `auth.customer_id`, or `UnauthorizedToolAccessError`
  is raised.
- `create_support_ticket`: if an `order_id` is supplied, the same order
  ownership check applies before a ticket can be linked to it. New tickets
  are always created for `auth.customer_id`, never for an id supplied in the
  request body.

This mirrors the security principle used throughout the project (see
[docs/guardrails.md](guardrails.md) and the upcoming security hardening
step): **authorization is enforced in application code and is never
delegated to the model.**

## Data separation

Customer, order, and ticket data live in `app/tools/` (in-memory demo
repository seeded with a handful of fixed sample records) and are entirely
separate from the pgvector knowledge-base repositories in
`app/db/repositories/`. Support/customer data never enters the vector
store, and knowledge-base chunks never contain customer data.

## Errors

`app/tools/errors.py` defines predictable exception types:

- `CustomerNotFoundError`, `OrderNotFoundError` — resource does not exist.
- `UnauthorizedToolAccessError` — the authenticated actor does not own the requested resource.
- `ToolRepositoryError` — the underlying data store failed unexpectedly (wraps any unexpected exception from the repository so callers never see raw internals or stack traces).

Malformed or missing arguments are rejected before the service is even
called, by Pydantic validation on the request models (`extra="forbid"`,
`min_length`/`max_length` constraints).

## Duplicate ticket handling

`create_support_ticket` checks for an existing, not-yet-resolved ticket from
the same customer with the same (case-insensitive, trimmed) subject and
description before creating a new one. If found, the existing ticket is
returned instead of creating a duplicate. This is a simple, demo-appropriate
heuristic — not a general-purpose deduplication system.

## Testing

`tests/test_support_tools.py` covers: valid lookups for each tool, missing
customer, missing order, invalid IDs (schema validation), unauthorized
customer/order/refund access, ticket creation, order-linked ticket
authorization, duplicate ticket handling (same customer vs. different
customers), and repository/tool failure wrapping.

## What's intentionally not built yet

- No REST endpoints expose these tools directly yet. They are designed to
  be invoked by the LangGraph agent introduced in the next step
  (`docs/agent.md`), which already has request-scoped context to build an
  `AuthContext`. Exposing them as raw HTTP endpoints without a real
  authentication system would create a false sense of security.
- No real authentication system exists yet; `AuthContext` is currently
  constructed directly by trusted application code (tests, and soon the
  agent layer) rather than derived from a verified session token. This is
  a known limitation, not a production authentication mechanism.
