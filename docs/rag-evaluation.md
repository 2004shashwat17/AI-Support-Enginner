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