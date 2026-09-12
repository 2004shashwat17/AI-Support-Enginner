import asyncio
from uuid import UUID

from app.db.repositories.retrieval import RetrievedChunk
from app.models.rag import Citation, RAGResponse, RetrievedChunkResponse
from evaluation.models import EvaluationDataset
from evaluation.runner import run_evaluation, run_retrieval_comparison


class FakeRetriever:
    def __init__(self, result: RetrievedChunk) -> None:
        self.result = result
        self.top_k_values: list[int | None] = []

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
        embedding_model: str | None = None,
    ) -> list[RetrievedChunk]:
        self.top_k_values.append(top_k)
        return [self.result]


class FakeRAGService:
    def __init__(self, response: RAGResponse) -> None:
        self.response = response
        self.questions: list[str] = []

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: UUID | None = None,
    ) -> RAGResponse:
        self.questions.append(question)
        return self.response


def test_runner_aggregates_retrieval_and_mocked_generation() -> None:
    dataset = EvaluationDataset.model_validate(
        {
            "version": "test",
            "description": "One-case runner test",
            "cases": [
                {
                    "case_id": "reset",
                    "category": "direct_factual",
                    "question": "How do I reset my password?",
                    "answerability": "answerable",
                    "expected_answer": "Reset it from Settings.",
                    "expected_key_facts": ["Settings"],
                    "relevant_sources": [
                        {"source_filename": "guide.txt", "chunk_index": 0}
                    ],
                }
            ],
        }
    )
    retrieved = RetrievedChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=100),
        chunk_index=0,
        content="Reset it from Settings.",
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
        cosine_distance=0.1,
        cosine_similarity=0.9,
    )
    response_chunk = RetrievedChunkResponse(
        chunk_id=retrieved.chunk_id,
        document_id=retrieved.document_id,
        chunk_index=retrieved.chunk_index,
        content=retrieved.content,
        metadata=retrieved.metadata,
        embedding_model=retrieved.embedding_model,
        cosine_distance=retrieved.cosine_distance,
        cosine_similarity=retrieved.cosine_similarity,
    )
    response = RAGResponse(
        answer="Reset it from Settings.",
        citations=[
            Citation(
                chunk_id=retrieved.chunk_id,
                document_id=retrieved.document_id,
                chunk_index=retrieved.chunk_index,
                metadata=retrieved.metadata,
            )
        ],
        retrieved_chunks=[response_chunk],
    )
    retriever = FakeRetriever(retrieved)
    rag_service = FakeRAGService(response)

    report = asyncio.run(
        run_evaluation(
            dataset,
            retriever,
            k_values=(1, 3),
            rag_service=rag_service,
        )
    )

    assert report.dataset_size == 1
    assert [metric.recall_at_k for metric in report.retrieval] == [1.0, 1.0]
    assert report.generation is not None
    assert report.generation.groundedness == 1.0
    assert report.generation.citation_correctness == 1.0
    assert retriever.top_k_values == [3]
    assert rag_service.questions == ["How do I reset my password?"]


def test_runner_reports_vector_keyword_and_hybrid_metrics_separately() -> None:
    dataset = EvaluationDataset.model_validate(
        {
            "version": "test",
            "description": "Strategy comparison",
            "cases": [
                {
                    "case_id": "reset",
                    "category": "direct_factual",
                    "question": "How do I reset my password?",
                    "answerability": "answerable",
                    "expected_answer": "Reset it from Settings.",
                    "expected_key_facts": ["Settings"],
                    "relevant_sources": [
                        {"source_filename": "guide.txt", "chunk_index": 0}
                    ],
                }
            ],
        }
    )
    relevant = RetrievedChunk(
        chunk_id=UUID(int=1),
        document_id=UUID(int=100),
        chunk_index=0,
        content="Reset it from Settings.",
        metadata={"source_filename": "guide.txt"},
        embedding_model="test-model",
    )
    irrelevant = RetrievedChunk(
        chunk_id=UUID(int=2),
        document_id=UUID(int=100),
        chunk_index=1,
        content="Other",
        metadata={"source_filename": "other.txt"},
        embedding_model="test-model",
    )

    report = asyncio.run(
        run_retrieval_comparison(
            dataset,
            {
                "vector": FakeRetriever(irrelevant),
                "keyword": FakeRetriever(relevant),
                "hybrid": FakeRetriever(relevant),
            },
            k_values=(1,),
        )
    )

    assert list(report.strategies) == ["vector", "keyword", "hybrid"]
    assert report.strategies["vector"][0].hit_rate_at_k == 0.0
    assert report.strategies["keyword"][0].hit_rate_at_k == 1.0
    assert report.strategies["hybrid"][0].hit_rate_at_k == 1.0