"""Hybrid retriever combining dense (vector) and sparse (BM25) retrieval.

Why hybrid:
- Dense embeddings excel at semantic similarity ("revenue decline" ↔ "shrinking sales")
- BM25 excels at exact matches (specific dollar amounts, ticker symbols, product names)
- Financial filings need both: semantic understanding AND precise numerical anchors

Fusion strategy: Reciprocal Rank Fusion (RRF) is parameter-free and consistently strong
across benchmarks. Weighted score fusion (alpha) is also offered for tuning experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from insightrag.ingestion.embedder import EmbeddingModel
from insightrag.retrieval.bm25_index import BM25Index
from insightrag.retrieval.vector_store import QdrantStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    source: str  # "dense", "sparse", or "hybrid"


class HybridRetriever:
    def __init__(
        self,
        vector_store: QdrantStore,
        bm25_index: BM25Index,
        embedder: EmbeddingModel,
        alpha: float = 0.6,
        fusion: str = "rrf",
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if fusion not in {"rrf", "weighted"}:
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.alpha = alpha
        self.fusion = fusion

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter_payload: dict | None = None,
    ) -> list[RetrievedChunk]:
        # Run both retrievals
        query_vec = self.embedder.encode_query(query)
        dense_results = await self.vector_store.search(query_vec, top_k=top_k, filter_payload=filter_payload)
        sparse_results = self.bm25_index.search(query, top_k=top_k)

        # Apply payload filter to sparse results too (BM25 doesn't natively support it)
        if filter_payload:
            sparse_results = [
                r for r in sparse_results
                if all(r.metadata.get(k) == v for k, v in filter_payload.items())
            ]

        if self.fusion == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        else:
            fused = self._weighted_fusion(dense_results, sparse_results, top_k)

        logger.debug(
            f"Retrieved {len(fused)} chunks (dense={len(dense_results)}, sparse={len(sparse_results)})"
        )
        return fused

    def _reciprocal_rank_fusion(
        self, dense, sparse, top_k: int, k: int = 60
    ) -> list[RetrievedChunk]:
        """RRF: score(d) = sum over rankers of 1/(k + rank(d)).

        k=60 is the constant from the original Cormack et al. (2009) RRF paper. It dampens
        the influence of top-ranked items so a single ranker can't dominate the fusion.
        """
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for rank, hit in enumerate(dense):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunks[hit.chunk_id] = RetrievedChunk(
                chunk_id=hit.chunk_id, text=hit.text, score=0.0,
                metadata=hit.metadata, source="dense",
            )

        for rank, hit in enumerate(sparse):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if hit.chunk_id in chunks:
                chunks[hit.chunk_id].source = "hybrid"
            else:
                chunks[hit.chunk_id] = RetrievedChunk(
                    chunk_id=hit.chunk_id, text=hit.text, score=0.0,
                    metadata=hit.metadata, source="sparse",
                )

        for cid, score in scores.items():
            chunks[cid].score = score

        return sorted(chunks.values(), key=lambda c: c.score, reverse=True)[:top_k]

    def _weighted_fusion(
        self, dense, sparse, top_k: int
    ) -> list[RetrievedChunk]:
        """Min-max normalize scores per ranker, then weighted sum."""
        def normalize(hits):
            if not hits:
                return {}
            scores = [h.score for h in hits]
            lo, hi = min(scores), max(scores)
            rng = hi - lo if hi > lo else 1.0
            return {h.chunk_id: (h.score - lo) / rng for h in hits}

        dense_norm = normalize(dense)
        sparse_norm = normalize(sparse)
        all_ids = set(dense_norm) | set(sparse_norm)

        chunks: dict[str, RetrievedChunk] = {}
        for h in dense + sparse:
            if h.chunk_id not in chunks:
                chunks[h.chunk_id] = RetrievedChunk(
                    chunk_id=h.chunk_id, text=h.text, score=0.0,
                    metadata=h.metadata, source="dense",
                )

        for cid in all_ids:
            d = dense_norm.get(cid, 0.0)
            s = sparse_norm.get(cid, 0.0)
            chunks[cid].score = self.alpha * d + (1 - self.alpha) * s
            if cid in dense_norm and cid in sparse_norm:
                chunks[cid].source = "hybrid"
            elif cid in sparse_norm:
                chunks[cid].source = "sparse"

        return sorted(chunks.values(), key=lambda c: c.score, reverse=True)[:top_k]
