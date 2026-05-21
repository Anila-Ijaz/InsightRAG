"""Application configuration via environment variables.

All settings are loaded from environment variables (or a `.env` file in dev).
This follows the 12-factor app methodology — never hardcode secrets or env-specific values.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "InsightRAG"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── LLM ──────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # ── Reranker ─────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k: int = 5

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_top_k: int = 20  # initial dense + sparse retrieval
    hybrid_alpha: float = 0.6  # weight for dense vs sparse (0=BM25 only, 1=dense only)
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Qdrant ───────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "sec_filings"

    # ── Postgres ─────────────────────────────────────────────────────
    postgres_url: str = "postgresql+asyncpg://insightrag:insightrag@localhost:5432/insightrag"

    # ── Redis ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── S3 (for raw documents) ───────────────────────────────────────
    s3_bucket: str = "insightrag-documents"
    s3_endpoint_url: str | None = None  # e.g. http://minio:9000 for local
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None

    # ── Observability ────────────────────────────────────────────────
    enable_tracing: bool = True
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Guardrails ───────────────────────────────────────────────────
    enable_pii_redaction: bool = True
    enable_injection_detection: bool = True
    max_query_length: int = 1000


@lru_cache
def get_settings() -> Settings:
    return Settings()
