# AI Support Engineer

Production-style AI customer-support platform, built incrementally.

## Backend setup

From the project root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For OpenAI-backed embedding calls, create a local environment file:

```bash
cp ../.env.example .env
```

Replace the placeholder `OPENAI_API_KEY` in `backend/.env`. Automated tests use mock providers and do not require an API key.

## Start PostgreSQL with pgvector

Install Docker Desktop, then run from the project root:

```bash
docker compose --env-file backend/.env up -d postgres
docker compose --env-file backend/.env ps
```

The container creates local `ai_support` and `ai_support_test` databases on its first start. Database credentials come from `backend/.env`.

Apply the schema to both databases:

```bash
cd backend
source .venv/bin/activate
set -a
source .env
set +a
alembic upgrade head
DATABASE_URL="$TEST_DATABASE_URL" alembic upgrade head
```

The migration enables pgvector and creates the `knowledge_chunks` table. It intentionally does not create a vector similarity index yet.

## Run the API

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the generated API documentation, or check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Ingest a text document

Only UTF-8 plain-text files are supported in this version.

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
	-F "file=@/absolute/path/to/faq.txt;type=text/plain"
```

The response contains the generated document ID, safe filename, and extraction metadata. The cleaned document content remains internal for future processing.

## Ask a grounded support question

After PostgreSQL is running, migrations are applied, and embedded chunks have been stored:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/support/ask \
	-H "Content-Type: application/json" \
	-d '{"question":"How can I reset my password?","top_k":5}'
```

The response contains the grounded answer, citations derived from retrieved chunk IDs, and the ranked chunks with cosine scores. Retrieval finds potentially relevant evidence; it does not by itself guarantee answer correctness.

Generation settings are read from `backend/.env`:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
LLM_TEMPERATURE=0.0
```

## Run tests

```bash
cd backend
source .venv/bin/activate
pytest
```

Database integration tests are skipped unless the test database is configured:

```bash
set -a
source .env
set +a
pytest -m integration
```

## Architecture overview

```
                    USER
                      |
                      v
                Support API (FastAPI)
                      |
                      v
              Support Agent (LangGraph)
                      |
          +-----------+------------+
          |           |            |
          v           v            v
       RAG Flow     Tools       Escalation
          |           |            |
          v           v            v
    Retrieval      Customer     Human Queue
       |            Order       (pending only)
       +-- Vector   Refund
       +-- Keyword  Ticket
       +-- RRF
       +-- Rerank
          |
          v
     Evidence Guardrails
          |
          v
          LLM
          |
          v
   Grounded Answer + Citations
```

Supporting infrastructure: PostgreSQL + pgvector, short-term conversation
memory, security/prompt-injection defenses, structured observability, an
evaluation framework, and Docker packaging.

## Feature map (with docs)

| Area | Docs |
|---|---|
| Retrieval evaluation methodology & metrics | [docs/rag-evaluation.md](docs/rag-evaluation.md) |
| Reranking (candidate_k vs top_k, fallback, RRF vs reranking) | [docs/rag-evaluation.md](docs/rag-evaluation.md) |
| Confidence/answerability guardrails | [docs/guardrails.md](docs/guardrails.md) |
| Customer/order/refund/ticket tools | [docs/support-tools.md](docs/support-tools.md) |
| LangGraph support agent | [docs/agent.md](docs/agent.md) |
| Conversation memory | [docs/memory.md](docs/memory.md) |
| Human escalation / HITL | [docs/escalation.md](docs/escalation.md) |
| Security & prompt-injection defense | [docs/security.md](docs/security.md) |
| Observability, latency, cost | [docs/observability.md](docs/observability.md) |
| Final evaluation report | [docs/final-evaluation.md](docs/final-evaluation.md) |
| Docker & deployment | [docs/deployment.md](docs/deployment.md) |

## Technology stack

Python, FastAPI, PostgreSQL + pgvector, SQLAlchemy, Alembic, OpenAI
(embeddings + structured LLM output), LangGraph, Pydantic, pytest, Docker.

## Agent & tools

The support agent (`app/agent/`) routes each request through a bounded
LangGraph graph to one of: grounded knowledge retrieval (`app/services/rag.py`),
a customer/order/refund/ticket tool (`app/tools/`), a clarification prompt,
or human escalation (`app/escalation/`). Tool authorization is enforced in
application code, never by the LLM (see [docs/support-tools.md](docs/support-tools.md)
and [docs/security.md](docs/security.md)).

## Human escalation

Escalations are created automatically when the user asks for a human,
retrieval evidence is insufficient, a tool fails, or a conversation
repeatedly fails to resolve. Every escalation record starts (and, in this
project, remains) `pending` — creating one never means a human has actually
reviewed anything (see [docs/escalation.md](docs/escalation.md)).

## Security

Retrieved documents and user input are treated as data, never as
instructions — this is stated explicitly in the system prompt and enforced
architecturally through tool authorization, structured LLM output, and a
system-prompt-leak guardrail. Prompt injection is not claimed to be fully
solved. See [docs/security.md](docs/security.md).

## Limitations

- The evaluation dataset is small (10 synthetic cases) and insufficient to
  claim production-level retrieval/generation accuracy; see
  [docs/final-evaluation.md](docs/final-evaluation.md).
- No real authentication system exists; tool authorization relies on an
  `AuthContext` supplied by trusted application code, not a verified session.
- Conversation memory, escalation records, and demo customer/order data are
  all in-memory and process-local (not persisted across restarts or shared
  across instances).
- This project has not been deployed anywhere; see
  [docs/deployment.md](docs/deployment.md) for what "deployment-ready" does
  and does not mean here.

## Future improvements

- Wire a persistent backend (Redis/PostgreSQL) for conversation memory and
  escalation records.
- Add a real authentication/session layer feeding `AuthContext`.
- Expand the evaluation dataset with a real, larger, human-reviewed corpus.
- Add an OpenTelemetry-backed `MetricsSink` implementation.
- Expose the LangGraph agent over the HTTP API (currently exercised
  directly in tests) once conversation/session identity exists end-to-end.
