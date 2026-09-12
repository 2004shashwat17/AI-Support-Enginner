from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_EMBEDDING_DIMENSIONS = 1536
RetrievalStrategy = Literal["vector", "keyword", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    embedding_dimensions: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        ge=DEFAULT_EMBEDDING_DIMENSIONS,
        le=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    embedding_batch_size: int = Field(default=100, gt=0, le=2048)
    retrieval_strategy: RetrievalStrategy = "vector"
    hybrid_candidate_top_k: int = Field(default=20, gt=0, le=100)
    rrf_constant: int = Field(default=60, gt=0)
    reranking_enabled: bool = False
    rerank_candidate_top_k: int = Field(default=20, gt=0, le=100)
    rerank_top_k: int = Field(default=5, gt=0, le=100)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(default=30, gt=0)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)
    llm_provider: str = Field(default="openai", pattern="^openai$")
    llm_model: str = Field(default="gpt-4.1-mini", min_length=1)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_prompt_price_per_1k: float | None = Field(default=None, ge=0.0)
    llm_completion_price_per_1k: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_rerank_top_k(self) -> "Settings":
        if self.rerank_candidate_top_k < self.rerank_top_k:
            raise ValueError(
                "rerank_candidate_top_k must be greater than or equal to rerank_top_k."
            )
        return self


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: SecretStr


@lru_cache
def get_settings() -> Settings:
    return Settings()
