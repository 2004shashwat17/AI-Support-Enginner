# Confidence, Answerability & Safety Guardrails

## Pipeline

```
USER QUESTION
    |
RETRIEVAL
    |
RERANKING (optional, see docs/rag-evaluation.md)
    |
EVIDENCE QUALITY GUARDRAIL
    |
ANSWER  or  INSUFFICIENT KNOWLEDGE
```

`app/services/guardrails.py` implements this stage. It is deliberately separate
from retrieval, reranking, and generation so it can be reasoned about, tested,
and tuned independently.

## Four different signals — do not conflate them

| Signal | What it measures | Where it lives | Calibrated probability? |
|---|---|---|---|
| **Retrieval relevance** | How closely a chunk's embedding/keyword match relates to the query, per the underlying model | `cosine_similarity`, `keyword_score`, `rrf_score`, `rerank_score` on `RetrievedChunk` | **No.** These are relative ranking signals from an uncalibrated model. A cosine similarity of 0.9 is not "90% confidence the answer is correct." |
| **Answer confidence** | Whether the *generated answer itself* is likely correct | Not implemented as a numeric score anywhere in this project | No calibrated confidence model exists here. This project never fabricates a "Confidence: NN%" number. |
| **Groundedness** | Whether the answer's claimed facts actually appear in the retrieved context | `evaluation/generation.py::evaluate_generation` (offline evaluation signal, key-fact overlap) | It's a deterministic lexical check, not a probability. |
| **Answerability** | Whether the knowledge base contains an answer to the question *at all* | Dataset-level ground truth in `evaluation/models.py::Answerability`, and the runtime guardrail's binary evidence-sufficiency decision | It's a discrete label (`answerable`/`unanswerable`), not a score. |

The runtime guardrail in `app/services/guardrails.py` only concerns itself with
one narrow question: **given the evidence actually retrieved for this
request, is it safe to let the LLM attempt an answer, and did the resulting
answer actually cite that evidence?** It does not attempt to estimate overall
answer confidence.

## Explicit states

`EvidenceStatus` (`app/services/guardrails.py`):

- `sufficient_evidence` — evidence passed the configured checks; the LLM was called (or its answer was grounded).
- `insufficient_evidence` — no usable evidence, evidence below a configured threshold, malformed scores, or a generated answer with zero citations.
- `retrieval_failed` — reserved status value; today retrieval failures are raised as `RAGRetrievalError` (see below) rather than returned in a response, since the API layer maps them to `503`.
- `generation_failed` — reserved status value; today generation failures are raised as `RAGGenerationError` for the same reason.
- `needs_clarification` — reserved for future multi-turn / agent use (Step 15's clarification flow); not produced by the current single-turn `RAGService`.

`RAGResponse.evidence_status` (`app/models/rag.py`) surfaces this decision to API callers without ever exposing a misleading numeric confidence value.

## Guardrail checks

`EvidenceGuardrail.assess(chunks)` runs **before** any LLM call:

1. Empty candidate list → `insufficient_evidence`.
2. Any non-finite (`NaN`/`inf`) similarity or rerank score → `insufficient_evidence` (malformed scores are never trusted).
3. Fewer than `min_chunk_count` chunks (default 1) → `insufficient_evidence`.
4. Optional `min_cosine_similarity` / `min_rerank_score` thresholds against the **top** result only, if configured (disabled by default). These are coarse, uncalibrated cutoffs, not confidence thresholds.
5. Otherwise → `sufficient_evidence`, and the LLM is called.

`assess_generation_grounding(citation_count)` runs **after** generation:

- Zero citations → `insufficient_evidence`; the response answer is overridden to the standard "knowledge base does not contain enough information" text and citations are cleared, even if the LLM produced fluent prose. This protects against confident-sounding but ungrounded answers.
- At least one citation → `sufficient_evidence`; the LLM's answer and citations are returned unchanged.

If evidence is insufficient at either stage, `RAGService.answer()` never calls the LLM unnecessarily (pre-check) or discards an ungrounded answer (post-check) rather than returning it to the user.

## Backward compatibility

- `RAGService(retriever, llm_provider)` still works with no changes — `guardrail` is an optional keyword argument defaulting to `EvidenceGuardrail()` with no thresholds configured (only the always-on empty-evidence and malformed-score checks apply).
- Existing citation, retrieval-failure, and generation-failure behavior (`RAGRetrievalError`, `RAGGenerationError`, `MalformedRAGResponseError`) is unchanged.
- `RAGResponse.evidence_status` defaults to `sufficient_evidence`, so any code constructing a `RAGResponse` directly (e.g. tests, fakes) without the field continues to work.

## Testing

- `tests/test_guardrails.py`: strong evidence, no evidence, weak evidence (similarity/rerank thresholds), minimum chunk count, malformed (NaN/inf) scores, conflicting evidence (multiple chunks, only top one gates), missing citations.
- `tests/test_rag.py`: end-to-end wiring — sufficient evidence produces a normal grounded answer; no evidence skips the LLM call entirely; weak evidence (below a configured threshold) skips the LLM call; malformed scores are treated as insufficient; a generated answer with no citations is safely overridden; retrieval/generation failures still raise exceptions rather than being silently downgraded to a guardrail state.

## Limitations

- Thresholds on cosine similarity / rerank score are optional and, when enabled, are simple cutoffs an operator must tune empirically for their embedding model — they are not calibrated against labeled correctness data in this project.
- `needs_clarification` and `retrieval_failed`/`generation_failed` as *response* states (rather than raised exceptions) are not yet produced by `RAGService`; they exist as forward-looking states for the agent/orchestration work in later steps.
- The grounding check only verifies that *some* citation exists, not that every claim in the answer is individually supported — full claim-level verification is out of scope here.
