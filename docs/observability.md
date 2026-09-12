# Observability, Latency & Cost (Step 18)

## Pipeline instrumented

```
HTTP request
  |  (request_id assigned, X-Request-ID header set, total HTTP latency logged)
  v
RAGService.answer()
  |-- retrieval     (timed: "retrieval" stage)
  |-- guardrail      (insufficient_evidence counter, if triggered)
  |-- generation     (timed: "generation" stage; LLM latency + token usage + cost logged separately inside OpenAILLMProvider)
  |-- total          (timed: "total" stage, wraps the whole answer() call)
  v
Escalation (if triggered)  -- "escalation_created" counter, tagged by reason
```

Reranking (`app/services/reranking.py::RerankingRetriever`, built in Step 11)
already logs candidate-retrieval and reranking latency separately via
`logger.info(...)` — the same idea introduced here, just implemented
slightly earlier in the project. It was not refactored onto the new
`MetricsSink` abstraction to avoid unnecessary churn to already-tested code;
both approaches produce the same structured latency signal.

## Abstraction: `MetricsSink`

`app/observability/metrics.py` defines a single `Protocol`:

```python
class MetricsSink(Protocol):
    def record_latency(self, stage: str, duration_ms: float, **tags: str) -> None: ...
    def record_value(self, name: str, value: float, **tags: str) -> None: ...
    def increment(self, counter: str, **tags: str) -> None: ...
```

The default implementation, `LoggingMetricsSink`, only writes structured log
lines — **no paid monitoring service or extra infrastructure is required.**
`NullMetricsSink` is available for tests that don't care about metrics
output. Swapping in an OpenTelemetry-backed sink later means implementing
this Protocol once (e.g. wrapping an OTel `Meter`) and passing it into
`RAGService`, `OpenAILLMProvider`, and `EscalationService` — no call sites
need to change, since they only depend on the Protocol.

## What is tracked

| Signal | Where | How |
|---|---|---|
| `request_id` | `app/observability/context.py`, set per HTTP request in `app/main.py` middleware | `contextvars`, returned as `X-Request-ID` response header |
| `conversation_id` | `app/observability/context.py` | Available to attach to any log line inside a conversation turn (not yet wired into every log call site — see limitations) |
| `retrieval_strategy` | Selected in `app/api/dependencies.py` per Step 6/10/11 config | Not currently tagged onto every metric emission; visible in configuration/logs at startup |
| Retrieval latency | `RAGService._answer`, "retrieval" stage | `app/observability/timing.py::measure` |
| Reranking latency | `RerankingRetriever.search` (Step 11) | `logger.info("Reranking latency: ...")` |
| LLM latency | `OpenAILLMProvider.generate` | `measure`-equivalent timer around the API call, recorded via `MetricsSink.record_latency("llm", ...)` |
| Total request latency | `RAGService.answer`, "total" stage | `measure` |
| HTTP request latency | `app/main.py` middleware | `measure`-equivalent timer around `call_next` |
| Token usage | `OpenAILLMProvider._record_token_usage` | Extracted from the OpenAI response's `usage.input_tokens`/`usage.output_tokens` (best-effort: if the field is absent, nothing is recorded — never guessed) |
| Estimated cost | `app/observability/cost.py::estimate_cost` | Computed **only** if `Settings.llm_prompt_price_per_1k` / `llm_completion_price_per_1k` are configured; otherwise logged as `total_cost_usd=unknown` |
| Retrieved chunk count | `RAGResponse.retrieved_chunks` (already in every response) | N/A — already structurally present |
| Escalation events | `EscalationService.create_escalation` | `MetricsSink.increment("escalation_created", reason=...)` |
| Fallback events | Reranker fallback (Step 11, logged), guardrail insufficient-evidence (`MetricsSink.increment("insufficient_evidence", ...)`) | `logger.warning` / `MetricsSink.increment` |
| Errors | Retrieval/generation failures | `MetricsSink.increment("retrieval_failed"/"generation_failed"/"llm_call_failed", ...)` |

## Cost: never fabricated

`estimate_cost(usage, prompt_price_per_1k, completion_price_per_1k)`
(`app/observability/cost.py`) returns `None` for any cost component whose
price was not configured — **not** zero, **not** a guessed public price.
`Settings.llm_prompt_price_per_1k` / `llm_completion_price_per_1k` default to
`None` (unconfigured), so by default this project reports token counts
without a dollar figure, exactly as instructed: *"If pricing is unavailable,
report token usage without pretending to know exact cost."*

## Logging hygiene

- Metric tags are identifiers (request id, conversation id, model name,
  stage name, escalation reason) — never raw question text, customer PII, or
  secrets.
- `app/main.py`'s request-correlation middleware logs only method, path,
  status, and duration — never the request or response body.
- The global exception handler (Step 17, `docs/security.md`) logs only the
  exception type and request path.

## Testing

- `tests/test_observability.py`: request/conversation id context vars,
  `correlation_tags()`, `RequestTimings`/`measure` (including when the
  measured block raises), `NullMetricsSink`/`LoggingMetricsSink` don't
  raise, and `estimate_cost` — unconfigured pricing yields `None`,
  configured pricing computes a real number, and partial pricing still
  yields an unknown total (never a partially-fabricated total).
- `tests/test_llm.py`: token usage is extracted and passed to the metrics
  sink; without configured pricing, the logged cost is explicitly `unknown`.
- `tests/test_rag.py`: `RAGService` records `retrieval`, `generation`, and
  `total` latency stages via an injected metrics sink.
- `tests/test_health.py`: every response includes an `X-Request-ID` header.

## Limitations / what's not wired up yet

- `conversation_id` is not yet set into the observability context from
  `SupportAgent.handle` — it exists as a context var and is used by
  `correlation_tags()`, but the agent layer doesn't call `set_conversation_id`
  yet. This would be a small follow-up once the agent is exposed over HTTP.
- No OpenTelemetry exporter is wired up; `MetricsSink` is designed to make
  that a drop-in addition later, not implemented now (no new infrastructure
  dependency added without a concrete need).
- Retrieval-strategy and reranking latency are not yet tagged with the same
  `request_id` used elsewhere, since those services don't currently receive
  a `MetricsSink`/context injection — only `RAGService` and
  `OpenAILLMProvider` do. Extending this to every stage is straightforward
  given the `MetricsSink` Protocol, but was not done to avoid touching
  already-tested Step 10/11 code without a concrete driving need.
