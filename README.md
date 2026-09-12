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
