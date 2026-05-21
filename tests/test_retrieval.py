"""Tests for hybrid retrieval — focuses on the fusion logic which is pure-Python and testable
without requiring Qdrant/embedder. We use stubs for the IO-bound dependencies.
"""
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from insightrag.retrieval.bm25_index import BM25Hit
from insightrag.retrieval.hybrid import HybridRetriever
from insightrag.retrieval.vector_store import ScoredPoint


def _make_retriever(dense_hits, sparse_hits, fusion="rrf", alpha=0.6):
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=dense_hits)
    bm25 = MagicMock()
    bm25.search = MagicMock(return_value=sparse_hits)
    embedder = MagicMock()
    embedder.encode_query = MagicMock(return_value=np.array([0.1] * 768))
    return HybridRetriever(vector_store, bm25, embedder, alpha=alpha, fusion=fusion)


@pytest.mark.asyncio
async def test_rrf_combines_both_rankers():
    dense = [
        ScoredPoint("a", "text_a", 0.9, {}),
        ScoredPoint("b", "text_b", 0.8, {}),
    ]
    sparse = [
        BM25Hit("b", "text_b", 5.0, {}),
        BM25Hit("c", "text_c", 4.0, {}),
    ]
    r = _make_retriever(dense, sparse, fusion="rrf")
    results = await r.retrieve("query", top_k=3)
    ids = [c.chunk_id for c in results]
    assert "a" in ids and "b" in ids and "c" in ids
    # b appears in both rankers — should outrank items appearing in only one
    b_chunk = next(c for c in results if c.chunk_id == "b")
    assert b_chunk.source == "hybrid"


@pytest.mark.asyncio
async def test_weighted_fusion_alpha_extremes():
    dense = [ScoredPoint("d1", "t", 0.9, {}), ScoredPoint("d2", "t", 0.5, {})]
    sparse = [BM25Hit("s1", "t", 10.0, {}), BM25Hit("s2", "t", 5.0, {})]

    # alpha=1.0 → dense only
    r_dense = _make_retriever(dense, sparse, fusion="weighted", alpha=1.0)
    results = await r_dense.retrieve("q", top_k=2)
    assert results[0].chunk_id == "d1"

    # alpha=0.0 → sparse only
    r_sparse = _make_retriever(dense, sparse, fusion="weighted", alpha=0.0)
    results = await r_sparse.retrieve("q", top_k=2)
    assert results[0].chunk_id == "s1"


@pytest.mark.asyncio
async def test_rrf_respects_top_k():
    dense = [ScoredPoint(f"d{i}", "t", 1.0 - 0.01 * i, {}) for i in range(20)]
    sparse = [BM25Hit(f"d{i}", "t", 10 - i, {}) for i in range(20)]
    r = _make_retriever(dense, sparse, fusion="rrf")
    results = await r.retrieve("q", top_k=5)
    assert len(results) == 5


def test_invalid_fusion_raises():
    with pytest.raises(ValueError):
        _make_retriever([], [], fusion="bogus")


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        _make_retriever([], [], alpha=2.0)
