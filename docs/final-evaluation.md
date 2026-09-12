# Final Evaluation Report (Step 19)

## Scope

This report separates four distinct kinds of results, per the project's
"never fabricate metrics" rule:

1. **UNIT TEST RESULTS** — deterministic, run in this environment, reported exactly.
2. **INFRASTRUCTURE-FREE EXTENDED EVALUATION** — deterministic checks against real production code (tools, agent, security heuristics), run in this environment, reported exactly.
3. **INTEGRATION TEST RESULTS** — require PostgreSQL; **not available in this environment**.
4. **RETRIEVAL / GENERATION EVALUATION RESULTS** — require PostgreSQL + a live OpenAI key; **not available in this environment**.

No numbers in categories 3 and 4 are stated below, because none were
actually produced. See [docs/rag-evaluation.md](rag-evaluation.md) for how
to run them once infrastructure is available.

## 1. Unit test results

Command: `cd backend && source .venv/bin/activate && pytest -q`

```
227 passed, 8 skipped, 1 warning
```

- **Skipped (8):** tests marked `integration`, requiring a local PostgreSQL
  database with pgvector (see `pyproject.toml` `[tool.pytest.ini_options]`
  markers). Not run here because PostgreSQL is not available in this
  sandbox (confirmed via `scripts/evaluate_rag.py`, which reports
  `PostgreSQL available: no`).
- **Warning (1):** a third-party `DeprecationWarning` from `starlette`'s
  test client (`anyio.abc.BlockingPortal`), unrelated to this project's code.
- **Failed: 0.**

Test coverage added across Steps 12-19: guardrails, support tools,
LangGraph agent (router + all graph paths), conversation memory,
escalation, security heuristics + rate limiting, observability, and
extended evaluation checks — see each step's `docs/*.md` for the specific
test list.

## 2. Infrastructure-free extended evaluation

`evaluation/extended_checks.py` (run via `python scripts/run_extended_evaluation.py`)
exercises real production code — `SupportToolService`, `SupportAgent`,
`EscalationService`, and `app.services.security` — using in-memory demo
data and fakes, requiring no PostgreSQL or live LLM call. Actual output
from this environment:

```
Extended Evaluation (infrastructure-free checks)
-------------------------------------------------
tool_correctness: 5/5 PASS
escalation_correctness: 4/4 PASS
prompt_injection_resistance: 9/9 PASS
multiturn_behavior: 3/3 PASS

Overall: PASS
```

What each category checks:

- **tool_correctness (5/5):** valid customer lookup, missing-customer
  error, cross-customer order access rejection, missing-order error, valid
  order-status lookup.
- **escalation_correctness (4/4):** insufficient-knowledge triggers
  escalation, explicit human request triggers escalation, escalation
  records start `pending`/not human-reviewed, insufficient-knowledge
  escalations carry the correct reason.
- **prompt_injection_resistance (9/9):** 4 adversarial prompts correctly
  flagged by the heuristic, 3 benign prompts correctly *not* flagged, a
  verbatim system-prompt leak correctly detected, a normal answer correctly
  not flagged as a leak.
- **multiturn_behavior (3/3):** a clarification is requested for a missing
  order id, a bare order id on the same conversation resumes the pending
  route, an unrelated conversation id does not inherit pending state.

These are real, reproducible pass/fail counts from this run — not
estimates. They complement, but do not replace, the retrieval/generation
evaluation below, which requires live infrastructure to be meaningful.

## 3. Integration test results

**Not run.** PostgreSQL is not available in this environment (`docker` is
not installed; `scripts/evaluate_rag.py` reports `PostgreSQL available:
no`). The 8 skipped tests above are exactly the integration tests that
would exercise real pgvector queries.

## 4. Retrieval / generation evaluation results

**Not measured**, for the same reason. See
[docs/rag-evaluation.md](rag-evaluation.md) for the full methodology, the
comparison across vector / keyword / hybrid / hybrid+reranking strategies,
and explicit instructions to run it once PostgreSQL + an OpenAI key are
available. As documented there, even a real run would be against a
10-case synthetic dataset (5 answerable), which is **too small to
demonstrate statistically meaningful differences between retrieval
strategies** — this limitation is stated there and repeated here so it is
not lost.

## Production hardening review (Step 19)

A pass over API / database / AI / agent / memory / security /
observability surfaced and fixed one concrete gap:

- **API — unbounded upload size.** `POST /documents/ingest` had no maximum
  document size, allowing a large upload to consume unbounded memory during
  UTF-8 decoding. Fixed: `app/services/document_ingestion.py` now enforces
  `MAX_DOCUMENT_SIZE_BYTES` (5 MiB), raising `DocumentTooLargeError`, mapped
  to HTTP 413 in `app/api/documents.py`. Covered by
  `tests/test_document_ingestion.py::test_rejects_documents_over_the_size_limit`.

Everything else reviewed (connection pooling with `pool_pre_ping=True`,
transaction-scoped seeding replacement, tool authorization, agent loop
boundedness, guardrail fallback behavior, escalation pending-state honesty,
rate limiting, safe error handling, secret handling via `SecretStr`) was
already correctly implemented in Steps 1-18 and did not need further
changes for this pass — see each step's `docs/*.md` for the original
implementation detail.

## Known weaknesses / limitations (full-project)

- No live retrieval/generation metrics exist yet in this environment (see
  above) — this is the single most important gap before any claim of
  "production-ready" retrieval quality.
- The evaluation dataset (10 cases) is synthetic and too small for
  statistically meaningful comparisons, even once infrastructure is
  available.
- No real authentication system exists; `AuthContext` is constructed by
  trusted application code, not derived from a verified session token
  (`docs/support-tools.md`).
- `InMemorySupportRepository`, `InMemoryConversationStore`, and
  `InMemoryEscalationRepository` are process-local and non-persistent —
  acceptable for a portfolio demo, not for a multi-instance production
  deployment (`docs/memory.md`, `docs/escalation.md`).
- Prompt-injection defenses are heuristic and architectural, not a
  complete solution (`docs/security.md`).
- Rate limiting is per-process/IP-keyed, not distributed.
