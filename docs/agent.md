# LangGraph Support Agent (Step 14)

## Why LangGraph instead of a single LLM function call

The support agent must route a request to genuinely different subsystems:
grounded RAG retrieval, four distinct database tools, a clarification path,
and human escalation — each with its own validation and error handling. A
single large function (or a single "let the LLM decide and call tools in a
loop") would either become an unreadable branch tree or an open-ended agent
loop with unclear termination guarantees.

LangGraph (`app/agent/graph.py`) gives:

- **An explicit typed state** (`AgentState`, a `TypedDict`) — every field
  the graph can produce is declared up front.
- **Explicit named nodes and edges** — the control flow is visible in
  `build_support_graph`, not implicit inside conditionals.
- **A bounded, acyclic graph** — there is no edge back to `route_request`,
  so every request runs a fixed, small number of nodes. Execution is
  bounded by construction, not by a hard iteration/recursion cap.
- **Reuse of existing services** — nodes call `RAGService` and
  `SupportToolService` rather than reimplementing retrieval, reranking,
  guardrails, or tool authorization.

## Graph

```
START
  |
UNDERSTAND REQUEST        (extract an order id like ORD-1234, if present)
  |
ROUTER                     (deterministic keyword-based intent classifier)
  |
  +-- knowledge      -> RAGService.answer(...)               (Steps 6-12)
  +-- customer        -> SupportToolService.get_customer(...) (Step 13)
  +-- order            -> SupportToolService.get_order_status(...)
  +-- refund           -> SupportToolService.get_refund_status(...)
  +-- ticket           -> SupportToolService.create_support_ticket(...)
  +-- clarification    -> asks for the missing piece of information
  +-- escalation       -> marks the request for human handoff
  |
VALIDATE RESULT            (guarantees a non-empty final answer)
  |
END
```

## Routing

`app/agent/router.py` is a small, pure, deterministic classifier —
**not an LLM call**. Keyword matching is checked in a fixed priority order
(escalation > refund > ticket > order > customer > knowledge) so behavior
is reproducible and free of hallucination risk. `extract_order_id` pulls an
`ord-\d+` pattern out of the question text with a regex.

If the classified intent is `order` or `refund` but no order id was found in
the question, the router itself redirects to `clarification` with a prompt
asking for the missing order id — this keeps the "ask for the missing piece"
behavior centralized rather than duplicated across the `order` and `refund`
nodes.

## State

`AgentState` (`app/agent/state.py`) is a `TypedDict` with every field
optional, since it's populated progressively:

```python
class AgentState(TypedDict, total=False):
    question: str
    customer_id: str | None
    order_id: str | None
    route: RouteName
    rag_response: RAGResponse | None
    tool_error: str | None
    final_answer: str
    final_citations: list[Citation]
    evidence_status: str
    escalated: bool
    escalation_reason: str | None
    needs_clarification: bool
    clarification_prompt: str | None
```

The public, stable output type is `SupportAgentResponse` (a Pydantic model),
returned by `SupportAgent.handle(question, customer_id=...)`. Internal graph
state is never returned directly to callers.

## Authorization within the graph

`customer_id` is supplied to `SupportAgent.handle(...)` by the caller (the
application, from an authenticated session — see `docs/support-tools.md`),
never inferred from the question text. Each tool node builds an
`AuthContext(customer_id=...)` from this value and passes it to
`SupportToolService`, which enforces ownership checks exactly as described in
`docs/support-tools.md`. If a node has no `customer_id` at all, it routes to
`needs_clarification` rather than guessing an identity.

## Error handling

- `knowledge` catches `RAGError` (covers `RAGRetrievalError`,
  `RAGGenerationError`, and `MalformedRAGResponseError`) and returns a safe
  fallback answer with `evidence_status = "generation_failed"` instead of
  propagating an exception or a raw error message.
- `customer` / `order` / `refund` / `ticket` catch `SupportToolError`
  subclasses and translate them into safe, user-facing text via
  `safe_tool_error_message` (e.g. `UnauthorizedToolAccessError` never leaks
  the word "unauthorized" or any internal detail — it becomes "I can only
  share details for your own account and orders.").
- `validate_result` is a final safety net: if any node somehow leaves
  `final_answer` empty, it is replaced with a safe fallback message.

## What's intentionally NOT implemented at this step

- The router does not call an LLM; adding an LLM-based intent classifier is
  possible later without changing the graph shape (`route_request` would
  just call a different classifier function).
- Conversation memory / multi-turn state (Step 15) and durable human
  escalation records (Step 16) are separate, later concerns; the
  `escalation` node here only marks `escalated=True` with a reason string.
- No API endpoint yet exposes the agent; it is exercised directly in
  `tests/test_agent.py`. Wiring `POST /api/v1/support/ask` (or a new
  endpoint) to `SupportAgent` is a follow-up integration task once
  conversation/session identity exists (Step 15).

## Testing

- `tests/test_agent_router.py`: pure router/classifier unit tests (order id
  extraction, each intent, priority ordering).
- `tests/test_agent.py`: full graph execution for all ten required paths —
  knowledge question, customer lookup, order status, refund status, ticket
  creation, clarification (missing order id and missing customer id),
  insufficient knowledge, tool failure (unauthorized order access, safe
  message), escalation, and malformed LLM output.
