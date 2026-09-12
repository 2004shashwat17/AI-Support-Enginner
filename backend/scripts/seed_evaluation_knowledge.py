import argparse
import asyncio
import os
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.repositories.chunks import ChunkRepository
from app.services.embeddings import EmbeddingService, OpenAIEmbeddingProvider
from evaluation.knowledge import KnowledgeSnapshotError, load_knowledge_snapshot
from evaluation.seeding import seed_evaluation_knowledge


DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "knowledge"
    / "support_knowledge.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed the versioned evaluation knowledge snapshot."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--database-url-env",
        choices=("TEST_DATABASE_URL", "DATABASE_URL"),
        default="TEST_DATABASE_URL",
        help="Environment variable containing the target PostgreSQL URL.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        snapshot = load_knowledge_snapshot(args.snapshot)
    except KnowledgeSnapshotError as exc:
        print(f"Knowledge snapshot error: {exc}")
        return 2

    database_url = os.getenv(args.database_url_env)
    if not database_url:
        print(f"Seeding unavailable: {args.database_url_env} is not configured.")
        print("No database changes were made.")
        return 2

    try:
        settings = Settings()
    except ValidationError:
        print("Seeding unavailable: embedding configuration is invalid.")
        print("No database changes were made.")
        return 2

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        print("Seeding unavailable: PostgreSQL could not be reached.")
        print("Apply migrations before running the seed command.")
        engine.dispose()
        return 2

    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    try:
        with Session(engine) as session:
            embedder = EmbeddingService(
                OpenAIEmbeddingProvider(client),
                model=settings.embedding_model,
                batch_size=settings.embedding_batch_size,
            )
            store = ChunkRepository(
                session,
                embedding_model=settings.embedding_model,
                embedding_dimensions=settings.embedding_dimensions,
            )
            result = await seed_evaluation_knowledge(snapshot, embedder, store)
    except Exception as exc:
        print(f"Seeding failed: {type(exc).__name__}")
        print("The scoped database transaction was rolled back.")
        return 2
    finally:
        await client.close()
        engine.dispose()

    print("Evaluation knowledge seeded")
    print(f"Target: {args.database_url_env}")
    print(f"Namespace: {snapshot.namespace}")
    print(f"Documents: {result.document_count}")
    print(f"Chunks: {result.chunk_count}")
    print(f"Embedding model: {settings.embedding_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
