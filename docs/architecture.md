# Architecture

This is a **production-oriented portfolio implementation**, not a claim of
full production readiness — see the Limitations section in each linked
document and in [docs/final-evaluation.md](final-evaluation.md).

## High-level flow

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

## Layer-by-layer

| Layer | Module(s) | Docs |
|---|---|---|
| Document ingestion & chunking | `app/services/document_ingestion.py`, `app/services/document_chunking.py` | [rag-evaluation.md](rag-evaluation.md) |
| Embeddings | `app/services/embeddings.py` | [rag-evaluation.md](rag-evaluation.md) |
| Vector storage | `app/db/models.py`, PostgreSQL + pgvector | root `README.md` |
| Retrieval (vector/keyword/hybrid/RRF) | `app/services/retrieval.py`, `app/services/keyword_retrieval.py`, `app/services/hybrid_retrieval.py` | [rag-evaluation.md](rag-evaluation.md) |
| Reranking | `app/services/reranking.py` | [rag-evaluation.md](rag-evaluation.md) |
| Evidence guardrails | `app/services/guardrails.py` | [guardrails.md](guardrails.md) |
| Grounded generation | `app/services/rag.py`, `app/services/llm.py` | [rag-evaluation.md](rag-evaluation.md), [security.md](security.md) |
| Support tools | `app/tools/` | [support-tools.md](support-tools.md) |
| Agent orchestration | `app/agent/` | [agent.md](agent.md) |
| Conversation memory | `app/memory/` | [memory.md](memory.md) |
| Human escalation | `app/escalation/` | [escalation.md](escalation.md) |
| Security | `app/services/security.py`, `app/api/rate_limit.py`, `app/main.py` | [security.md](security.md) |
| Observability | `app/observability/` | [observability.md](observability.md) |
| Evaluation | `backend/evaluation/`, `backend/scripts/evaluate_rag.py`, `backend/scripts/run_extended_evaluation.py` | [rag-evaluation.md](rag-evaluation.md), [final-evaluation.md](final-evaluation.md) |
| Deployment | `Dockerfile`, `docker-compose.yml`, `docker/backend/entrypoint.sh` | [deployment.md](deployment.md) |

## Design principles followed throughout

- **Composition over hardcoding.** Reranking, escalation, and memory are all
  optional collaborators injected into existing services (`RAGService`,
  `SupportAgent`) via constructor parameters with safe defaults — none of
  them required rewriting the components they extend.
- **Application code owns authorization and data access.** The LLM never
  executes SQL, never supplies its own identity, and never decides whether
  it's allowed to see a piece of data (`app/tools/service.py`).
- **Explicit states over invented confidence scores.** Guardrails,
  escalation, and readiness checks use discrete, named states
  (`sufficient_evidence`, `pending`, `not_ready`) rather than fabricated
  percentages.
- **Never fabricate metrics, cost, or deployment status.** Every numeric
  claim in this project's docs was either actually measured in this
  environment, or explicitly marked as not measured and why.
