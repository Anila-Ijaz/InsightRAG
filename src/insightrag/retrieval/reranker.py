"""Cross-encoder reranker.

The retrieval pipeline is intentionally two-stage:
1. Fast bi-encoder (dense) + BM25 fetch top-K (e.g. 20) — cheap, recall-oriented
2. Slow cross-encoder reranks those K — expensive, precision-oriented

A cross-encoder scores (query, passage) jointly, which gives far better relevance
than the bi-encoder used for indexing. The base model is BGE reranker; in production
you'd fine-tune it on your domain (see `training/train_reranker.py`).
"""
from __future__ import annotations

from functools import lru_cache

import torch
from loguru import logger
from sentence_transformers import CrossEncoder

from insightrag.config import get_settings
from insightrag.retrieval.hybrid import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading reranker: {model_name} on {self.device}")
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Overwrite score with reranker score, sort, take top_k
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.score = float(score)

        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker(settings.reranker_model)
