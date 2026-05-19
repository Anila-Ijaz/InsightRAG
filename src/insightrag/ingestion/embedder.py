"""Embedding model wrapper.

We use BGE (BAAI General Embeddings) because:
- Top of MTEB leaderboard for English retrieval
- Open weights — no API costs at inference
- Supports query/passage asymmetric encoding (better than symmetric)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from insightrag.config import get_settings


# BGE recommends prefixing queries (not passages) with a retrieval instruction.
# This asymmetric setup matches how the model was trained.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.is_bge = "bge" in model_name.lower()

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode passages (documents to be indexed)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # Normalize for cosine similarity via dot product
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query, applying BGE instruction prefix when applicable."""
        text = BGE_QUERY_INSTRUCTION + query if self.is_bge else query
        return self.model.encode([text], normalize_embeddings=True)[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    settings = get_settings()
    return EmbeddingModel(settings.embedding_model, device=settings.embedding_device)
