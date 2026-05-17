"""FastAPI application entry point.

Exposes:
  POST /v1/query          — synchronous Q&A
  POST /v1/query/stream   — streaming Q&A (SSE)
  POST /v1/ingest         — ingest a ticker's 10-K filings
  GET  /healthz           — liveness probe
  GET  /readyz            — readiness probe (checks Qdrant, embedder)
  GET  /metrics           — Prometheus metrics
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse
from starlette.responses import Response

from insightrag.api.dependencies import get_rag_chain
from insightrag.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from insightrag.config import get_settings
from insightrag.generation.rag_chain import RAGChain
from insightrag.guardrails.input_guard import PromptInjectionDetected
from insightrag.observability.metrics import (
    QUERY_COUNTER,
    QUERY_LATENCY,
    RETRIEVAL_LATENCY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} (env={settings.environment})")
    # Warm up the RAG chain so first request isn't slow
    _ = get_rag_chain()
    logger.info("Application ready")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="InsightRAG",
    version="0.1.0",
    description="Production-grade RAG for SEC 10-K filings",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────── Health checks ──────────────────────────

@app.get("/healthz", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Liveness: app is running. Always 200 unless the process is dead."""
    return HealthResponse(status="ok", version="0.1.0", qdrant=True, embedder=True)


@app.get("/readyz", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Readiness: dependencies are reachable. Used by k8s/load balancers."""
    settings = get_settings()
    qdrant_ok = True
    try:
        from insightrag.retrieval.vector_store import get_vector_store

        store = get_vector_store()
        await store.client.get_collections()
    except Exception as e:
        logger.error(f"Qdrant unreachable: {e}")
        qdrant_ok = False

    embedder_ok = True
    try:
        from insightrag.ingestion.embedder import get_embedding_model
        _ = get_embedding_model()
    except Exception as e:
        logger.error(f"Embedder unavailable: {e}")
        embedder_ok = False

    if qdrant_ok and embedder_ok:
        return HealthResponse(status="ok", version="0.1.0", qdrant=True, embedder=True)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"qdrant": qdrant_ok, "embedder": embedder_ok},
    )


# ────────────────────────── Metrics ──────────────────────────────

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ────────────────────────── Query endpoints ──────────────────────

@app.post("/v1/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    rag: Annotated[RAGChain, Depends(get_rag_chain)],
) -> QueryResponse:
    filter_payload = _build_filter(req)
    try:
        with QUERY_LATENCY.time():
            response = await rag.answer(req.question, filter_payload=filter_payload)
        QUERY_COUNTER.labels(status="success", streaming="false").inc()
        RETRIEVAL_LATENCY.observe(response.retrieval_latency_ms / 1000)
    except PromptInjectionDetected as e:
        QUERY_COUNTER.labels(status="rejected_injection", streaming="false").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        QUERY_COUNTER.labels(status="rejected_invalid", streaming="false").inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        QUERY_COUNTER.labels(status="error", streaming="false").inc()
        raise HTTPException(status_code=500, detail="Internal error") from e

    return QueryResponse(
        answer=response.answer,
        citations=[CitationOut(**c.__dict__) for c in response.citations],
        metrics={
            "retrieval_latency_ms": response.retrieval_latency_ms,
            "rerank_latency_ms": response.rerank_latency_ms,
            "generation_latency_ms": response.generation_latency_ms,
            "total_latency_ms": response.total_latency_ms,
        },
    )


@app.post("/v1/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    rag: Annotated[RAGChain, Depends(get_rag_chain)],
):
    filter_payload = _build_filter(req)

    async def event_generator():
        try:
            async for event in rag.stream(req.question, filter_payload=filter_payload):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream")
                    break
                yield {"event": event["type"], "data": json.dumps(event["data"])}
            QUERY_COUNTER.labels(status="success", streaming="true").inc()
        except PromptInjectionDetected as e:
            QUERY_COUNTER.labels(status="rejected_injection", streaming="true").inc()
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
        except Exception as e:
            logger.exception(f"Stream failed: {e}")
            QUERY_COUNTER.labels(status="error", streaming="true").inc()
            yield {"event": "error", "data": json.dumps({"message": "Internal error"})}

    return EventSourceResponse(event_generator())


# ────────────────────────── Ingestion endpoint ───────────────────

@app.post("/v1/ingest", response_model=IngestResponse, status_code=202)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Synchronous ingestion endpoint. In production this would enqueue a Celery task."""
    # Lazy import to keep cold-start fast
    from insightrag.api.dependencies import _bm25_index
    from insightrag.ingestion.chunker import SemanticChunker
    from insightrag.ingestion.embedder import get_embedding_model
    from insightrag.ingestion.parser import SECFilingParser
    from insightrag.retrieval.vector_store import get_vector_store

    settings = get_settings()
    parser = SECFilingParser(download_dir=__import__("pathlib").Path("data/raw"))
    chunker = SemanticChunker(settings.chunk_size, settings.chunk_overlap)
    embedder = get_embedding_model()
    store = get_vector_store()
    bm25 = _bm25_index()

    await store.ensure_collection()

    paths = parser.download(req.ticker, limit=req.limit)
    total_chunks = 0
    for path in paths:
        doc = parser.parse(path, ticker=req.ticker)
        chunks = list(chunker.chunk_document(doc))
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metas = [c.metadata for c in chunks]
        embeddings = embedder.encode_passages(texts)
        await store.upsert(ids, texts, embeddings, metas)
        bm25.add(ids, texts, metas)
        total_chunks += len(chunks)

    # Persist BM25 so the index survives restarts
    bm25.save(__import__("pathlib").Path("data/bm25_index.pkl"))

    return IngestResponse(
        ticker=req.ticker,
        filings_ingested=len(paths),
        total_chunks=total_chunks,
    )


# ────────────────────────── helpers ──────────────────────────────

def _build_filter(req: QueryRequest) -> dict | None:
    f: dict = {}
    if req.ticker:
        f["ticker"] = req.ticker.upper()
    if req.section:
        f["section"] = req.section
    return f or None
