# Conversation Memory (Step 15)

## What memory is (and is not) for

**Short-term conversation memory** (`app/memory/`) tracks: message history for
a `conversation_id`, which identity a conversation is currently authenticated
as, and which single piece of missing information a prior clarification turn
was waiting on (e.g. an order id).

**It never stores authoritative customer/order/refund facts.** Every turn
that needs a customer or order fact calls `SupportToolService` again (see
[docs/support-tools.md](support-tools.md)) — memory is not a cache or
substitute for that lookup. This is a deliberate separation:

```
SHORT-TERM CONVERSATION MEMORY          CUSTOMER / ORDER DATA
(app/memory/)                           (app/tools/)
- message history (bounded)             - always fetched fresh, per call
- customer_id the session is acting as  - authorization re-checked every time
- pending clarification slot
```

## Example flow

```
USER: "What is my order status?"
AI:   "Please provide your order ID."     <- route=clarification, pending_slot="order_id"
USER: "ORD-1234"
AI:   "Order ORD-1234 status: shipped."   <- resumes the pending "order" route
```

`app/agent/graph.py::SupportAgent.handle(question, customer_id=None,
conversation_id=None)` implements this:

1. If `conversation_id` is given, resume `ConversationState` from
   `ConversationMemoryService`.
2. If the caller didn't pass `customer_id` this turn, fall back to the one
   already associated with the conversation (still just an identity pointer —
   the actual customer record is re-fetched by the tool node).
3. If the previous turn left a `pending_route`/`pending_slot` (e.g. "order" /
   "order_id"), and this turn's text doesn't match a stronger intent keyword
   but does contain an order id, the router resumes the pending route instead
   of misclassifying a bare "ORD-1234" reply as a knowledge question.
4. After the graph runs, the new pending state (or its absence, once
   resolved) and the two new messages (user + assistant) are persisted.

## Session isolation

Each `conversation_id` maps to an independent `ConversationState` in
`InMemoryConversationStore` (a `dict` keyed by id). A bare order id sent under
a different `conversation_id` does **not** resume another conversation's
pending clarification — see
`tests/test_agent.py::test_separate_conversation_ids_do_not_share_pending_state`.

## Bounded history

`InMemoryConversationStore(max_history_messages=20)` (default) keeps only the
most recent N messages per conversation, dropping the oldest first
(`tests/test_memory.py::test_history_is_truncated_to_max_messages`). **No
unlimited history is ever sent anywhere.** In the current implementation this
bound is conservative to the point of being moot for LLM calls: `RAGService`
is still single-turn / stateless (it takes one `question` string per call,
see [docs/rag-evaluation.md](rag-evaluation.md) and
[docs/guardrails.md](guardrails.md)) — conversation history is not yet
forwarded into the RAG prompt at all. The bound exists so conversation state
itself doesn't grow unbounded in memory, and so that when history *is* wired
into an LLM prompt in a future step, the token/latency tradeoff is already
enforced structurally rather than added as an afterthought.

## Corrupted / failing state handling

`ConversationMemoryService.start_or_resume`:

- If the underlying store raises on read, or returns a state whose
  `conversation_id` does not match the one requested (a corrupted/mismatched
  entry), the service logs a warning and starts a **fresh** conversation
  rather than propagating the error or silently using the wrong conversation.
- If the store raises on write (`record_turn`), the failure is logged and
  swallowed rather than crashing the request — losing conversation history
  is far less harmful than failing an otherwise-successful support answer.

## Storage

`InMemoryConversationStore` is a process-local, in-memory `dict`. This is
appropriate for a single-process demo/portfolio deployment. It is **not**
shared across multiple application instances or processes and does not
survive a restart — a real production deployment would back
`ConversationStore` with Redis/PostgreSQL, but no such backend is added here
to avoid unnecessary infrastructure for a portfolio project. The `Protocol`
(`app/memory/store.py::ConversationStore`) is defined so a persistent backend
can be swapped in later without touching `ConversationMemoryService` or the
agent.

## Testing

- `tests/test_memory.py`: new conversation, multiple turns accumulating in
  order, session isolation, missing conversation, history truncation
  (bounded history), a `limit` parameter on `history()`, state persistence of
  `pending_route`/`pending_slot`/`customer_id`, a mismatched/corrupted stored
  state falling back to a fresh conversation, and store read/write failures
  not crashing the caller.
- `tests/test_agent.py`: the full "ask for order id, then supply it"
  multi-turn flow, and confirmation that two different `conversation_id`s do
  not share pending state.
