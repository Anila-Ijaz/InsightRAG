"""Retrieval benchmarking — compare dense / sparse / hybrid / hybrid+rerank.

This is what justifies the architectural choices in the README. Produces a table
of MRR@10, Recall@10, nDCG@10 across the four configurations.

Requires a labeled set of (query, relevant_chunk_ids) pairs. Format: JSONL with
{"query": str, "relevant_chunk_ids": [str, ...]} per line.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

from loguru import logger

from insightrag.config import get_settings
from insightrag.ingestion.embedder import get_embedding_model
from insightrag.retrieval.bm25_index import BM25Index
from insightrag.retrieval.hybrid import HybridRetriever
from insightrag.retrieval.reranker import get_reranker
from insightrag.retrieval.vector_store import get_vector_store


# ────────── metrics ──────────

def mrr_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(retrieved[:k], start=1) if d in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


async def benchmark(testset_path: Path, k: int = 10) -> dict[str, dict[str, float]]:
    settings = get_settings()
    embedder = get_embedding_model()
    vector_store = get_vector_store()
    bm25 = BM25Index.load(Path("data/bm25_index.pkl"))
    reranker = get_reranker()

    # Four configurations
    dense_only = HybridRetriever(vector_store, bm25, embedder, fusion="weighted", alpha=1.0)
    sparse_only = HybridRetriever(vector_store, bm25, embedder, fusion="weighted", alpha=0.0)
    hybrid_rrf = HybridRetriever(vector_store, bm25, embedder, fusion="rrf")

    configs = {"dense": dense_only, "sparse": sparse_only, "hybrid_rrf": hybrid_rrf}
    results: dict[str, dict[str, list[float]]] = {
        name: {"mrr": [], "recall": [], "ndcg": []} for name in list(configs) + ["hybrid_rrf+rerank"]
    }

    testset = [json.loads(l) for l in testset_path.read_text().splitlines() if l.strip()]
    logger.info(f"Benchmarking on {len(testset)} labeled queries")

    for row in testset:
        relevant = set(row["relevant_chunk_ids"])
        query = row["query"]

        for name, retriever in configs.items():
            chunks = await retriever.retrieve(query, top_k=k)
            ids = [c.chunk_id for c in chunks]
            results[name]["mrr"].append(mrr_at_k(ids, relevant, k))
            results[name]["recall"].append(recall_at_k(ids, relevant, k))
            results[name]["ndcg"].append(ndcg_at_k(ids, relevant, k))

        # hybrid + rerank
        candidates = await hybrid_rrf.retrieve(query, top_k=settings.retrieval_top_k)
        reranked = reranker.rerank(query, candidates, top_k=k)
        ids = [c.chunk_id for c in reranked]
        results["hybrid_rrf+rerank"]["mrr"].append(mrr_at_k(ids, relevant, k))
        results["hybrid_rrf+rerank"]["recall"].append(recall_at_k(ids, relevant, k))
        results["hybrid_rrf+rerank"]["ndcg"].append(ndcg_at_k(ids, relevant, k))

    summary = {
        name: {metric: sum(vals) / len(vals) if vals else 0.0 for metric, vals in metrics.items()}
        for name, metrics in results.items()
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=Path("evals/retrieval_testset.jsonl"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("evals/retrieval_benchmark.json"))
    args = parser.parse_args()

    summary = asyncio.run(benchmark(args.testset, k=args.k))

    print(f"\n{'Config':<22} {'MRR@10':>10} {'Recall@10':>12} {'nDCG@10':>10}")
    print("-" * 56)
    for name, metrics in summary.items():
        print(f"{name:<22} {metrics['mrr']:>10.4f} {metrics['recall']:>12.4f} {metrics['ndcg']:>10.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
