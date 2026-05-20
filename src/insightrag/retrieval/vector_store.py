"""Qdrant vector store wrapper.

Why Qdrant:
- Production-grade, written in Rust (fast)
- Supports payload filtering (we use this to filter by ticker/section)
- Self-hostable + has managed cloud
- Simple HTTP/gRPC API, good Python client
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from insightrag.config import get_settings


@dataclass
class ScoredPoint:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class QdrantStore:
    """Async Qdrant client wrapper for indexing and retrieval."""

    def __init__(self, url: str, collection: str, dim: int, api_key: str | None = None):
        self.collection = collection
        self.dim = dim
        self.client = AsyncQdrantClient(url=url, api_key=api_key)

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        try:
            await self.client.get_collection(self.collection)
            logger.info(f"Qdrant collection '{self.collection}' exists")
        except (UnexpectedResponse, ValueError):
            logger.info(f"Creating Qdrant collection '{self.collection}'")
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                ),
            )
            # Payload indexes accelerate filtered search
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="ticker",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name="section",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def upsert(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
    ) -> None:
        """Upsert a batch of chunks."""
        points = [
            models.PointStruct(
                id=self._hash_id(chunk_id),
                vector=embedding.tolist(),
                payload={"chunk_id": chunk_id, "text": text, **metadata},
            )
            for chunk_id, text, embedding, metadata in zip(
                chunk_ids, texts, embeddings, metadatas, strict=True
            )
        ]
        await self.client.upsert(collection_name=self.collection, points=points)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filter_payload: dict | None = None,
    ) -> list[ScoredPoint]:
        """Dense vector search with optional metadata filtering."""
        query_filter = self._build_filter(filter_payload) if filter_payload else None

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            ScoredPoint(
                chunk_id=r.payload["chunk_id"],
                text=r.payload["text"],
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k not in {"chunk_id", "text"}},
            )
            for r in results
        ]

    @staticmethod
    def _build_filter(payload: dict) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in payload.items()
            ]
        )

    @staticmethod
    def _hash_id(chunk_id: str) -> int:
        """Qdrant point IDs must be UUIDs or unsigned ints. Use a stable 64-bit hash."""
        import hashlib

        digest = hashlib.sha256(chunk_id.encode()).digest()
        return int.from_bytes(digest[:8], "big")


def get_vector_store() -> QdrantStore:
    settings = get_settings()
    return QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
    )
