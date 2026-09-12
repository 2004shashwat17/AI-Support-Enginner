# Human Escalation / Human-in-the-Loop (Step 16)

## AI decision vs. human review — the critical distinction

Creating an `EscalationRecord` (`app/escalation/models.py`) means **the AI
decided this request needs a human**. It never means a human has actually
looked at it. `status` starts at `pending` and there is no real human queue
or ticketing system wired up in this project — `human_reviewed` is a
separate, explicit boolean that always stays `False` here. Nothing in this
codebase ever sets it to `True`, because no human workflow exists to do so
honestly. This is a known limitation, stated plainly rather than faked.

```json
{
  "escalation_id": "esc-...",
  "conversation_id": "conv-1",
  "customer_id": "cust-1001",
  "reason": "insufficient_knowledge",
  "summary": "...",
  "relevant_citations": [],
  "status": "pending",
  "human_reviewed": false,
  "created_at": "...",
  "updated_at": "..."
}
```

`EscalationStatus`: `pending` -> `assigned` -> `resolved`. Only `pending` is
ever produced automatically by this project; `assigned`/`resolved` are
modeled for a future real human-queue integration.

## Trigger conditions

| Trigger | Where | `EscalationReason` |
|---|---|---|
| User explicitly asks for a human | Router keyword match ("speak to a human", "escalate", etc.) -> `escalation` node | `user_requested_human` |
| Knowledge is insufficient | `knowledge` node, when `RAGResponse.evidence_status == insufficient_evidence` (see [docs/guardrails.md](guardrails.md)) | `insufficient_knowledge` |
| Tool failure prevents resolution | `customer`/`order`/`refund`/`ticket` nodes, only for `ToolRepositoryError` (an actual backend failure — **not** `CustomerNotFoundError`/`OrderNotFoundError`/`UnauthorizedToolAccessError`, which are expected, already-handled outcomes) | `tool_failure` |
| Repeated failure in the same conversation | `SupportAgent.handle`, when the same `conversation_id` needs clarification `REPEATED_FAILURE_THRESHOLD` (3) turns in a row | `repeated_failure` |
| Sensitive/high-risk issue | Router keyword match (fraud, unauthorized charges, legal threats, etc. are included in the escalation keyword list, so they route straight to `escalation` like an explicit human request) | `user_requested_human` (routed the same way) |

Every trigger creates a `pending` escalation record via `EscalationService`
and sets `SupportAgentResponse.escalated=True` plus an `escalation_id` the
caller can reference. **The user's answer is not blocked** by escalation
except for the explicit "human requested" and "repeated failure" paths,
where the final answer text explicitly tells the user a human will follow up.
For the "insufficient knowledge" and "tool failure" triggers, the user still
receives the existing safe fallback text (the knowledge-insufficient message,
or a safe tool-error message) — escalation just additionally queues the
request for follow-up.

## Architecture

```
app/escalation/
  models.py      EscalationStatus, EscalationReason, CreateEscalationRequest, EscalationRecord
  repository.py  InMemoryEscalationRepository (demo, process-local, like app/tools/repository.py)
  service.py     EscalationService: create_escalation / get_escalation / list_escalations
```

`SupportAgent` takes an optional `escalation_service: EscalationService`
(default: a fresh in-memory one), mirroring how `memory` is injected. Nodes
that can trigger escalation are built via factories
(`make_knowledge_node(rag_service, escalation_service)`, etc.) so the
escalation dependency is explicit and testable, not a hidden global.

## Conversation context in escalation records

`CreateEscalationRequest.conversation_id` / `customer_id` are populated from
the current `AgentState` (itself derived from `SupportAgent.handle`'s
`conversation_id`/`customer_id` parameters), and `relevant_citations` are
populated from the RAG response's citations when escalating due to
insufficient knowledge. This gives a human reviewer the same evidence the AI
saw, without duplicating retrieval logic.

## Testing

- `tests/test_escalation.py`: `EscalationService` unit tests — creation
  starts `pending`/not human-reviewed, lookup, not-found, status filtering.
- `tests/test_agent.py`: insufficient-knowledge auto-escalation (with a
  resulting `pending`, non-human-reviewed record), tool repository failure
  escalation (safe error message, no raw exception text), repeated
  clarification failure escalating after the threshold, and the existing
  explicit "speak to a human" escalation path.

## Limitations

- No real paging/ticketing/human-queue system is integrated; escalation
  records are only ever `pending` and process-local (lost on restart), like
  the other demo repositories in this project.
- The "sensitive/high-risk" trigger is keyword-based (same list used for
  explicit human requests), not a dedicated risk classifier — it will miss
  many genuinely sensitive requests and may over-trigger on others.
- `REPEATED_FAILURE_THRESHOLD = 3` is a fixed, un-tuned constant.
