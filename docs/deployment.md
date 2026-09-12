# Docker, Deployment & Production Polish (Step 20)

## Status: deployment-ready, NOT deployed

**Docker is not installed in this development sandbox** (`docker`/`docker-compose`
commands are unavailable here). The Dockerfile and `docker-compose.yml` below
were written and reviewed carefully, but **have not been built or run in this
environment** — that would require the user to run them locally or in CI.
No live URL, running container, or successful build is claimed. This section
documents what was created and how to actually run it.

## What was added

```
Dockerfile                        # backend image: install deps, run migrations, start uvicorn
.dockerignore                     # excludes .venv, __pycache__, .env, .git, etc. from the build context
docker/backend/entrypoint.sh      # runs `alembic upgrade head` then execs uvicorn
docker-compose.yml                # postgres (pgvector) + backend services
.env.example                      # updated with reranking/rate-limit/cost settings
```

### Dockerfile

- Base: `python:3.12-slim`.
- Copies only `backend/pyproject.toml`, `alembic.ini`, `app/`, `migrations/`,
  `scripts/`, `evaluation/` — **no `.env` file, no secrets, no `.venv`, no
  test files** are baked into the image (enforced by `.dockerignore` and by
  only copying the specific directories needed).
- Installs the package via `pip install .` (uses `pyproject.toml`
  dependencies, which now include `langgraph` — see `docs/agent.md`).
- Creates and switches to a non-root user (`appuser`) before running the app.
- `ENTRYPOINT` runs `docker/backend/entrypoint.sh`, which runs
  `alembic upgrade head` and then execs `uvicorn` — migrations run
  automatically and predictably on every container start, and the container
  fails fast (non-zero exit) if migrations fail, rather than serving traffic
  against an out-of-date schema.

### docker-compose.yml

- `postgres`: the existing `pgvector/pgvector:0.8.6-pg17` service (unchanged
  from Steps 1-11), with a `pg_isready` healthcheck.
- `backend` (new): builds from the repo-root `Dockerfile`, loads
  `backend/.env` via `env_file`, but **overrides `DATABASE_URL`** to point at
  the `postgres` service hostname instead of `localhost` (the two run in
  different network contexts: on the host during local dev, vs. the compose
  network inside containers). Depends on `postgres` being healthy before
  starting. Has its own healthcheck against `GET /health`.
- No secrets are stored in `docker-compose.yml` itself — all secret values
  (`OPENAI_API_KEY`, `POSTGRES_PASSWORD`) come from `backend/.env`, which is
  git-ignored and never copied into the image.

## Health endpoints

- `GET /health` — **liveness**. No dependencies (no database call); always
  returns `{"status": "ok"}` if the process is running. An orchestrator
  should use this to decide whether to restart the container.
- `GET /health/ready` — **readiness**. Runs a lightweight `SELECT 1` against
  the configured database. Returns `{"status": "ready"}` (200) or
  `{"status": "not_ready"}` (503) — **never** the connection string,
  credentials, or the underlying exception. An orchestrator should use this
  to decide whether to route traffic to the container.

This separation matters: a container can be alive (liveness passes) while
temporarily unable to reach Postgres (readiness fails) — restarting it
would not help and would cause unnecessary churn, whereas removing it from
the load balancer's rotation (readiness) is the correct response.

## Running locally (without Docker)

Already documented in [docs/rag-evaluation.md](rag-evaluation.md):

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Running with Docker (once Docker is available)

```bash
cp .env.example backend/.env   # then edit backend/.env with real values
docker compose up --build
```

This should build the `backend` image, start `postgres`, wait for it to be
healthy, run migrations via the entrypoint script, and start the API on
`http://localhost:8000` (configurable via `BACKEND_PORT`). **This has not
been executed in this environment** — verify locally before relying on it.

## Environment variables

See `.env.example` (repo root) for the full list, now including the
settings added in Steps 11/17/18:

- Retrieval/reranking: `RETRIEVAL_STRATEGY`, `HYBRID_CANDIDATE_TOP_K`,
  `RRF_CONSTANT`, `RERANKING_ENABLED`, `RERANK_CANDIDATE_TOP_K`, `RERANK_TOP_K`.
- Rate limiting: `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`,
  `RATE_LIMIT_WINDOW_SECONDS`.
- Cost (optional, never fabricated if unset): `LLM_PROMPT_PRICE_PER_1K`,
  `LLM_COMPLETION_PRICE_PER_1K`.
- Everything from Steps 1-10 (embeddings, LLM, Postgres) is unchanged.

## Known limitations

- **Not verified to build or run in this environment** — Docker is not
  installed here. The user should run `docker compose build` and
  `docker compose up` locally to confirm before relying on this for a real
  deployment.
- No cloud deployment (AWS/GCP/Azure/Fly/Render/etc.) was configured or
  attempted — no cloud credentials are available in this environment, and
  none should be fabricated. The application is deployment-ready (containerized,
  health-checked, migration-safe) but has not actually been deployed anywhere.
- The `backend` service's healthcheck uses a `python -c` HTTP request rather
  than `curl`/`wget` (not guaranteed present in `python:3.12-slim`) — this
  avoids adding an extra OS package solely for the healthcheck.
- No HTTPS/TLS termination, reverse proxy, or horizontal scaling
  configuration is included — appropriate for a single-instance portfolio
  deployment behind a developer's own reverse proxy/load balancer, not a
  complete production topology.
