"""Embedding model wrapper.

Two interchangeable providers, selected via `EMBEDDING_PROVIDER`:

- "local"  → BGE (BAAI General Embeddings) via sentence-transformers.
  Top of MTEB for English retrieval, open weights (no API cost), supports
  query/passage asymmetric encoding. Needs torch (~1GB+ RAM). Used by the full stack.

- "openai" → OpenAI embeddings API (e.g. text-embedding-3-small).
  No local model, no torch — keeps the container small enough for free-tier
  hosting. Used by the lite deployment.

Both expose the same interface (`encode_passages`, `encode_query`, `.dim`) so the
rest of the pipeline is provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from loguru import logger

from insightrag.config import get_settings

# BGE recommends prefixing queries (not passages) with a retrieval instruction.
# This asymmetric setup matches how the model was trained.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(ABC):
    """Provider-agnostic embedding interface."""

    dim: int

    @abstractmethod
    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode passages (documents to be indexed)."""

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""


class EmbeddingModel(Embedder):
    """Local sentence-transformers embedder (BGE)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        # Lazy import: only the full stack installs sentence-transformers/torch.
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.is_bge = "bge" in model_name.lower()

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # Normalize for cosine similarity via dot product
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        text = BGE_QUERY_INSTRUCTION + query if self.is_bge else query
        return self.model.encode([text], normalize_embeddings=True)[0]


class OpenAIEmbeddingModel(Embedder):
    """OpenAI embeddings API embedder.

    Uses the synchronous OpenAI client so the interface stays sync (matching the
    local embedder). `dimensions` is requested explicitly so the vector size always
    matches the configured Qdrant collection dim (text-embedding-3-* support this).
    """

    def __init__(self, api_key: str, model: str, dim: int):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dim,
        )
        vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Normalize for cosine-via-dot-product consistency with the local embedder.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode_passages(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            out.append(self._embed(texts[start : start + batch_size]))
        return np.vstack(out) if out else np.empty((0, self.dim), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed([query])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> Embedder:
    settings = get_settings()
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when embedding_provider=openai")
        logger.info(
            f"Using OpenAI embeddings: model={settings.openai_embedding_model} "
            f"dim={settings.embedding_dim}"
        )
        return OpenAIEmbeddingModel(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
            dim=settings.embedding_dim,
        )
    return EmbeddingModel(settings.embedding_model, device=settings.embedding_device)
