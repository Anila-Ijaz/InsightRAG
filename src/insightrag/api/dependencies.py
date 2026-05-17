"""FastAPI dependencies.

Singletons are cached via lru_cache. The retriever has runtime state (the BM25 index),
which is loaded on first request from disk or built from existing Qdrant payloads.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from insightrag.config import get_settings
from insightrag.generation.llm_client import get_llm_client
from insightrag.generation.rag_chain import RAGChain
from insightrag.guardrails.input_guard import get_input_guard
from insightrag.guardrails.output_guard import get_output_guard
from insightrag.ingestion.embedder import get_embedding_model
from insightrag.retrieval.bm25_index import BM25Index
from insightrag.retrieval.hybrid import HybridRetriever
from insightrag.retrieval.reranker import get_reranker
from insightrag.retrieval.vector_store import get_vector_store

BM25_PATH = Path("data/bm25_index.pkl")


@lru_cache(maxsize=1)
def _bm25_index() -> BM25Index:
    if BM25_PATH.exists():
        return BM25Index.load(BM25_PATH)
    return BM25Index()


@lru_cache(maxsize=1)
def get_rag_chain() -> RAGChain:
    settings = get_settings()
    retriever = HybridRetriever(
        vector_store=get_vector_store(),
        bm25_index=_bm25_index(),
        embedder=get_embedding_model(),
        alpha=settings.hybrid_alpha,
        fusion="rrf",
    )
    return RAGChain(
        retriever=retriever,
        reranker=get_reranker(),
        llm=get_llm_client(),
        input_guard=get_input_guard(),
        output_guard=get_output_guard(),
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.reranker_top_k,
    )
