#!/usr/bin/env python3
"""Export indexed chunks from Qdrant to JSONL.

This is the bridge between the running RAG system and the reranker fine-tuning script
(`training/train_reranker.py`). It dumps every chunk's text + metadata so the trainer
can generate synthetic Q&A pairs and mine hard negatives over them.

Usage:
    python scripts/export_chunks.py --output data/chunks.jsonl
    python scripts/export_chunks.py --output data/aapl_chunks.jsonl --ticker AAPL
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger
from qdrant_client import models

from insightrag.config import get_settings
from insightrag.retrieval.vector_store import get_vector_store


async def export_chunks(output_path: Path, ticker: str | None, batch_size: int = 256) -> int:
    store = get_vector_store()

    # Qdrant scroll API is the right primitive for full-table dumps —
    # paginated, doesn't load everything into memory at once.
    query_filter = None
    if ticker:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker))]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_exported = 0
    next_offset = None

    with output_path.open("w") as f:
        while True:
            points, next_offset = await store.client.scroll(
                collection_name=store.collection,
                scroll_filter=query_filter,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,  # we only need text + metadata for training
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                record = {
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text"),
                    "metadata": {k: v for k, v in payload.items() if k not in {"chunk_id", "text"}},
                }
                f.write(json.dumps(record) + "\n")
                n_exported += 1

            logger.info(f"Exported {n_exported} chunks so far")
            if next_offset is None:
                break

    return n_exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--ticker", help="Filter to one ticker (optional)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    settings = get_settings()
    logger.info(f"Exporting from Qdrant collection '{settings.qdrant_collection}' "
                f"at {settings.qdrant_url}")
    total = asyncio.run(export_chunks(args.output, args.ticker, args.batch_size))
    logger.info(f"Wrote {total} chunks to {args.output}")


if __name__ == "__main__":
    main()
