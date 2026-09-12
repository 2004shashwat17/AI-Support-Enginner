# RAG Evaluation Baseline

## Purpose

This evaluation asks two separate questions:

1. Did retrieval find the manually labeled evidence?
2. Did generation use retrieved evidence, answer the question, and cite real chunks?

The signals identify regressions and support comparisons. They do not prove factual correctness.

## Dataset

The versioned dataset is `backend/evaluation/dataset/rag_dataset.json`.

Version 1.0 contains 10 cases: five answerable and five unanswerable. The repository does not yet contain a real company knowledge corpus, so this dataset is intentionally synthetic and limited to four facts already present in tests:

- Password reset from Settings
- Email recovery
- Refunds within 30 days
- A document containing "Shipping details"

The reviewed snapshot is `backend/evaluation/knowledge/support_knowledge.json`. Its four document UUIDs are committed values, and existing chunking derives chunk UUIDs deterministically from each document UUID and character offsets. Evaluation cases reference the resulting document and chunk UUIDs directly.

## Reproducible Seeding

Seeding uses the production ingestion, chunking, embedding, and chunk-repository components. The snapshot fixes chunk size at 500 characters and overlap at 0, producing one chunk per current document.

The operation is idempotent by scoped replacement:

1. Build and validate every snapshot document and chunk.
2. Generate all embeddings before changing the database.
3. In one transaction, delete rows whose `document_id` is one of the four snapshot IDs.
4. Insert the replacement chunks and commit.

The repository rejects replacement when embedded chunks do not exactly match the explicit document-ID scope. Rows belonging to any other document are never included in the delete predicate.

## Retrieval Metrics

- **Recall@K:** For each labeled answerable case, the fraction of expected relevant sources represented in the first K results, averaged across cases.
- **Precision@K:** For each labeled answerable case, the fraction of returned Top-K results that match a relevant-source label, averaged across cases.
- **Hit Rate@K:** The fraction of labeled answerable cases with at least one relevant source in the first K results.

Unanswerable cases have no positive relevance labels and are excluded from these three metrics. Empty retrieval produces zero recall, precision, and hit rate for answerable cases.

## Reranking

### Why reranking exists

Initial retrieval (dense vector, keyword, or hybrid) is optimized for **recall**: it returns a `candidate_top_k` set likely to contain the relevant chunk somewhere in the list, but ordering within that set can be noisy. Reranking is optimized for **precision**: it looks at the query together with each candidate chunk and reorders them so the most directly relevant chunk moves toward the top before a smaller `final top_k` is passed to generation.

Example:

```
Query: "How do I reset my password?"

Initial retrieval:
1. Account security
2. Login troubleshooting
3. Password reset          <- most relevant
4. Account profile
5. Authentication FAQ

After reranking (ideally):
1. Password reset
2. Login troubleshooting
3. Account security
...
```

Reranking does **not** always improve quality — it must be measured against the existing evaluation dataset rather than assumed. See "Actual reranking evaluation" below.

### Candidate retrieval vs. final ranking

- `candidate_top_k`: how many chunks the base retriever (vector, keyword, or hybrid) returns for the reranker to examine. Must be `> 0`.
- `top_k` (final): how many chunks the reranker returns after scoring. Must be `> 0` and `<= candidate_top_k`.

Retrieving only `top_k` before reranking defeats the purpose, since the reranker would have nothing left to reorder. `RerankingRetriever` (in `app/services/reranking.py`) enforces `candidate_top_k >= top_k` and raises `InvalidRerankRequestError` otherwise.

### How reranking differs from RRF

Reciprocal Rank Fusion (`app/services/hybrid_retrieval.py`) **combines rankings from two different retrieval systems** (dense and keyword) using only rank positions — it never looks at chunk content or the query text at fusion time. Reranking is a separate, later stage: it **evaluates the relevance of already-retrieved candidates against the query text**, using the query and each candidate's content. RRF happens once per hybrid search; reranking can be layered on top of vector, keyword, or hybrid results.

### Implementation

`app/services/reranking.py` defines:

- `RerankProvider`: a Protocol with `async def score(query, documents) -> list[float]`. This is the swappable seam — production cross-encoders or hosted reranking APIs implement this Protocol without touching any other code.
- `LexicalOverlapRerankProvider`: the current implementation. It scores each candidate by the fraction of query tokens also present in the candidate content (token-set overlap). It is deterministic, requires no network access or API key, and is intentionally simple. **Limitation:** it has no semantic understanding of synonyms, paraphrasing, or word order, so it will not match a real cross-encoder's quality. It exists to make the reranking stage usable and evaluable today; swapping in a hosted/cross-encoder provider later requires only a new `RerankProvider` implementation plus environment-variable configuration (API key, base URL) — never hardcoded secrets.
- `Reranker`: wraps a `RerankProvider`, deduplicates candidates by `chunk_id`, and produces the final ranked list.
- `RerankingRetriever`: wraps any existing `Retriever` (vector, keyword, or hybrid) with the candidate-retrieval + reranking pipeline. Because it implements the same `search(...)` interface as every other retriever, `RAGService` needs no reranking-specific code — reranking is enabled purely through composition in `app/api/dependencies.py`.

### Ordering and determinism

Results are sorted by descending `rerank_score`. Ties are broken, in order, by:

1. Ascending `original_rank` (the candidate's position in the retrieval result it came from).
2. Ascending `chunk_id` (string form).

This makes ordering deterministic across identical runs, matching the tie-break approach already used for RRF.

### Failure and fallback behavior

If the reranking provider fails, times out, returns the wrong number of scores, or returns non-finite scores, `Reranker` **falls back to the original retrieval order** (truncated to `top_k`) rather than failing the request. The fallback is logged (`logger.warning(..., exc_info=True)`) so it is observable. Duplicate candidates (same `chunk_id`) are removed before scoring, keeping the first occurrence. Empty candidate lists return an empty result. Fewer candidates than `top_k` simply return fewer results — this is not an error.

### Enabling reranking

Reranking is opt-in and does not change default behavior:

```
retrieval_strategy = hybrid       # existing setting: vector | keyword | hybrid
reranking_enabled = true          # new setting, default false
rerank_candidate_top_k = 20       # default 20
rerank_top_k = 5                  # default 5
```

Existing vector/keyword/hybrid retrieval remains available unchanged when `reranking_enabled` is `false` (the default).

### Latency and cost tradeoffs

Without reranking: one retrieval stage (embedding + vector/keyword query).

With reranking: candidate retrieval (same as above, but for `candidate_top_k` results) **plus** a reranking computation over those candidates. `RerankingRetriever` and `Reranker` log retrieval latency and reranking latency separately (`logger.info`) so the two stages can be measured independently later; total request latency is the sum plus generation. Reranking can improve precision but adds latency and — for a hosted/cross-encoder provider — API cost. The current lexical-overlap provider adds negligible compute cost since it runs locally with no network calls.

### Security

The current lexical-overlap provider runs locally; no candidate content leaves the process. If a hosted reranking API is introduced later, sending customer knowledge-base content to that provider must be an explicit, documented configuration choice, not a silent default. No API keys, secrets, or customer content are logged; only latency numbers and warnings are logged.

## Actual reranking evaluation

`scripts/evaluate_rag.py` compares four retrieval strategies against the same dataset: `vector`, `keyword`, `hybrid`, and `hybrid_reranked` (hybrid retrieval followed by the lexical-overlap reranker). `evaluation/reporting.py` prints each strategy's Hit@K/Recall@K/Precision@K and, when both are measured, an explicit hybrid -> hybrid+reranking delta in percentage points.

As of this writing, PostgreSQL is not available in this environment (see "Current Baseline" below), so **no live Hit@K/Recall@K/Precision@K numbers for `hybrid_reranked` have been produced against real data**. Results must come from actually running `scripts/evaluate_rag.py` against a seeded PostgreSQL instance; this document will not state improvement/regression figures until that run has actually happened. Unit tests (`tests/test_reranking.py`, `tests/test_evaluation_runner.py`) verify the mechanics (ordering, fallback, candidate/top_k validation, and comparison wiring) using in-memory fakes, not real retrieval data.

Given the dataset's small size (10 cases, 5 answerable), even a real run may not show statistically meaningful differences between strategies — this should be stated explicitly in any reported results rather than over-interpreted.

## Generation Signals

- **Groundedness:** A conservative key-fact overlap signal. Expected facts present in the answer must also appear in retrieved context.
- **Answer relevance:** The fraction of expected key facts found in the answer.
- **Citation correctness:** The fraction of citations that uniquely identify chunks in the response's retrieved set.
- **Answerability accuracy:** Whether answerable cases receive an answer and unanswerable cases receive the exact knowledge-base insufficiency response with no citations.

These lexical checks can miss valid paraphrases and cannot detect every unsupported claim. No LLM judge is used. An LLM judge could add another noisy signal later, but it would not establish truth.

## Current Baseline

Date: 2026-09-11

- Dataset size: 10
- PostgreSQL retrieval evaluation: **not measured** (`TEST_DATABASE_URL`/Docker unavailable)
- Generation evaluation: **not measured**
- Real LLM generation performed: **no**
- LLM judge used: **no**

No metric values are recorded because PostgreSQL is unavailable in the current environment. The reproducible corpus now exists, but it has not been seeded into a real pgvector database here.

## Run

Install Docker Desktop, start PostgreSQL, and migrate the test database from the project root:

```bash
docker compose --env-file backend/.env up -d postgres
cd backend
source .venv/bin/activate
set -a
source .env
set +a
DATABASE_URL="$TEST_DATABASE_URL" alembic upgrade head
```

Seed and evaluate from `backend/`:

```bash
python scripts/seed_evaluation_knowledge.py
python scripts/evaluate_rag.py
```

To include real generation calls after PostgreSQL is available and the evaluation corpus is seeded:

```bash
python scripts/evaluate_rag.py --run-generation
```

The runner exits without metrics if configuration, PostgreSQL, retrieval, or generation fails. It never substitutes fabricated values or reports partial results.

## Limitations And Next Use

The dataset is too small and synthetic to represent production support traffic. It lacks a real company corpus, human-reviewed paraphrases, difficult multi-document questions, and adversarial prompts. After PostgreSQL is available, the immediate next step is to seed this snapshot and record measured metrics before changing chunking, retrieval, embeddings, or prompts.