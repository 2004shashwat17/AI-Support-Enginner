import argparse
import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import DatabaseSettings, RetrievalStrategy, Settings
from app.db.repositories.keyword_retrieval import KeywordRetrievalRepository
from app.db.repositories.retrieval import RetrievalRepository
from app.db.session import create_database_engine
from app.services.embeddings import EmbeddingService, OpenAIEmbeddingProvider
from app.services.hybrid_retrieval import HybridRetriever
from app.services.keyword_retrieval import KeywordRetriever
from app.services.llm import OpenAILLMProvider
from app.services.rag import RAGService
from app.services.retrieval import RetrievalService, Retriever
from evaluation.dataset import EvaluationDatasetError, load_evaluation_dataset
from evaluation.reporting import print_comparison_report
from evaluation.runner import (
    EvaluationReport,
    RetrievalComparisonReport,
    run_evaluation,
    run_retrieval_comparison,
)


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "evaluation" / "dataset" / "rag_dataset.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the current RAG baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Call the configured LLM and calculate deterministic generation signals.",
    )
    parser.add_argument(
        "--generation-strategy",
        choices=("vector", "keyword", "hybrid"),
        default="vector",
        help="Retrieval strategy used when --run-generation is enabled.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        dataset = load_evaluation_dataset(args.dataset)
    except EvaluationDatasetError as exc:
        print(f"Evaluation dataset error: {exc}")
        return 2

    print("RAG Evaluation")
    print("--------------")
    print(f"Dataset size: {len(dataset.cases)}")

    try:
        database_settings = DatabaseSettings()
        engine = create_database_engine(database_settings)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except (ValidationError, SQLAlchemyError):
        print("PostgreSQL available: no")
        print("Retrieval metrics: not measured")
        print("Generation metrics: not measured")
        print("LLM judge used: no")
        return 2

    print("PostgreSQL available: yes")
    try:
        settings = Settings()
    except ValidationError:
        print("Retrieval metrics: not measured (embedding configuration unavailable)")
        print("Generation metrics: not measured")
        print("LLM judge used: no")
        engine.dispose()
        return 2

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    try:
        with Session(engine) as session:
            embedding_service = EmbeddingService(
                OpenAIEmbeddingProvider(client),
                model=settings.embedding_model,
                batch_size=settings.embedding_batch_size,
            )
            repository = RetrievalRepository(
                session,
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
            )
            retriever = RetrievalService(
                embedding_service,
                repository,
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
            )
            keyword_retriever = KeywordRetriever(
                KeywordRetrievalRepository(
                    session,
                    embedding_model=settings.embedding_model,
                ),
                embedding_model=settings.embedding_model,
            )
            retrievers: dict[str, Retriever] = {
                "vector": retriever,
                "keyword": keyword_retriever,
                "hybrid": HybridRetriever(
                    retriever,
                    keyword_retriever,
                    rrf_constant=settings.rrf_constant,
                    candidate_top_k=settings.hybrid_candidate_top_k,
                ),
            }
            comparison = await run_retrieval_comparison(dataset, retrievers)
            generation_report = None
            if args.run_generation:
                generation_strategy: RetrievalStrategy = args.generation_strategy
                rag_service = RAGService(
                    retrievers[generation_strategy],
                    OpenAILLMProvider(
                        client,
                        model=settings.llm_model,
                        temperature=settings.llm_temperature,
                    ),
                )
                generation_report = await run_evaluation(
                    dataset,
                    retrievers[generation_strategy],
                    rag_service=rag_service,
                )
    except Exception as exc:
        print(f"Evaluation could not complete: {type(exc).__name__}")
        print("No partial metrics are reported.")
        return 2
    finally:
        await client.close()
        engine.dispose()

    print_comparison_report(comparison)
    print_generation_report(
        generation_report,
        generation_requested=args.run_generation,
        generation_strategy=args.generation_strategy,
    )
    return 0


def print_generation_report(
    report: EvaluationReport | None,
    *,
    generation_requested: bool,
    generation_strategy: str,
) -> None:
    if report is None or report.generation is None:
        print("Generation metrics: not run")
    else:
        print(f"Generation signals ({generation_strategy} retrieval):")
        print(f"  Groundedness: {report.generation.groundedness:.1%}")
        print(f"  Answer relevance: {report.generation.answer_relevance:.1%}")
        print(f"  Citation correctness: {report.generation.citation_correctness:.1%}")
        print(
            f"  Answerability accuracy: "
            f"{report.generation.answerability_accuracy:.1%}"
        )
    print(f"Real LLM generation performed: {'yes' if generation_requested else 'no'}")
    print("LLM judge used: no")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
