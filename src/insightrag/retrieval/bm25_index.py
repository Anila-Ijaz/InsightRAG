"""BM25 sparse retriever.

We use a simple in-memory BM25 here. For production at larger scale, replace this with
OpenSearch/Elasticsearch — same interface, swap implementation. BM25 catches exact-match
signals (ticker symbols, specific dollar amounts, product names) that dense embeddings
sometimes miss.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi


@dataclass
class BM25Hit:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class BM25Index:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Lowercase, keep alphanumerics (preserves tickers/numbers)
        return re.findall(r"\b\w+\b", text.lower())

    def build(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        logger.info(f"Building BM25 index over {len(texts)} chunks")
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.metadatas = metadatas

    def add(self, chunk_ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """Add new chunks and rebuild. (BM25Okapi doesn't support incremental updates.)"""
        self.build(
            self.chunk_ids + chunk_ids,
            self.texts + texts,
            self.metadatas + metadatas,
        )

    def search(self, query: str, top_k: int = 20) -> list[BM25Hit]:
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # argsort descending
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Hit(
                chunk_id=self.chunk_ids[i],
                text=self.texts[i],
                score=float(scores[i]),
                metadata=self.metadatas[i],
            )
            for i in top_idx
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunk_ids": self.chunk_ids,
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.bm25 = data["bm25"]
        idx.chunk_ids = data["chunk_ids"]
        idx.texts = data["texts"]
        idx.metadatas = data["metadatas"]
        return idx
