from collections.abc import AsyncGenerator, Generator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import AsyncOpenAI
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DatabaseSettings, Settings, get_settings
from app.db.repositories.keyword_retrieval import KeywordRetrievalRepository
from app.db.repositories.retrieval import RetrievalRepository
from app.db.session import create_database_engine, create_session_factory
from app.services.embeddings import EmbeddingService, OpenAIEmbeddingProvider
from app.services.hybrid_retrieval import HybridRetriever
from app.services.keyword_retrieval import KeywordRetriever
from app.services.llm import OpenAILLMProvider
from app.services.rag import RAGService
from app.services.reranking import LexicalOverlapRerankProvider, Reranker, RerankingRetriever
from app.services.retrieval import RetrievalService, Retriever


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    engine = create_database_engine(DatabaseSettings())
    return create_session_factory(engine)


def get_database_session() -> Generator[Session, None, None]:
    with get_session_factory()() as session:
        yield session


async def get_rag_service(
    session: Annotated[Session, Depends(get_database_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[RAGService, None]:
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    embedding_service = EmbeddingService(
        OpenAIEmbeddingProvider(client),
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    retrieval_repository = RetrievalRepository(
        session,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )
    retrieval_service = RetrievalService(
        embedding_service,
        retrieval_repository,
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
        "vector": retrieval_service,
        "keyword": keyword_retriever,
        "hybrid": HybridRetriever(
            retrieval_service,
            keyword_retriever,
            rrf_constant=settings.rrf_constant,
            candidate_top_k=settings.hybrid_candidate_top_k,
        ),
    }
    llm_provider = OpenAILLMProvider(
        client,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    active_retriever: Retriever = retrievers[settings.retrieval_strategy]
    if settings.reranking_enabled:
        active_retriever = RerankingRetriever(
            active_retriever,
            Reranker(LexicalOverlapRerankProvider()),
            candidate_top_k=settings.rerank_candidate_top_k,
            default_top_k=settings.rerank_top_k,
        )
    try:
        yield RAGService(active_retriever, llm_provider)
    finally:
        await client.close()
