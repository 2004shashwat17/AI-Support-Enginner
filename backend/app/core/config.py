from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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
    llm_provider: str = Field(default="openai", pattern="^openai$")
    llm_model: str = Field(default="gpt-4.1-mini", min_length=1)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)


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
