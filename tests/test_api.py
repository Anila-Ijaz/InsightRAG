"""API smoke tests. Heavy dependencies (Qdrant, embeddings, LLM) are mocked.

These tests verify request validation and the orchestration flow without external services.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """A TestClient with the RAG chain dependency overridden by a mock.

    Uses FastAPI's `dependency_overrides` (keyed on the real `get_rag_chain`) and clears
    it on teardown, so tests don't leak state into each other. The app's lifespan builds a
    real-but-lightweight chain (lite config in conftest → no model downloads, no network).
    """
    from insightrag.api.dependencies import get_rag_chain
    from insightrag.api.main import app
    from insightrag.generation.rag_chain import Citation, RAGResponse

    chain = AsyncMock()
    chain.answer = AsyncMock(
        return_value=RAGResponse(
            answer="Apple's net sales were $383.3 billion [1].",
            citations=[
                Citation(
                    index=1,
                    chunk_id="AAPL_x_mda_0",
                    ticker="AAPL",
                    filing_date="2023-10-30",
                    section="mda",
                    text_preview="...",
                )
            ],
            retrieval_latency_ms=50.0,
            rerank_latency_ms=20.0,
            generation_latency_ms=800.0,
            total_latency_ms=870.0,
        )
    )

    app.dependency_overrides[get_rag_chain] = lambda: chain
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_health_endpoint(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_validation_rejects_empty(client):
    r = client.post("/v1/query", json={"question": "", "top_k": 5})
    assert r.status_code == 422


def test_query_validation_rejects_too_long(client):
    r = client.post("/v1/query", json={"question": "a" * 1500, "top_k": 5})
    assert r.status_code == 422


def test_query_validation_rejects_bad_ticker(client):
    r = client.post("/v1/query", json={"question": "test", "ticker": "lowercase!", "top_k": 5})
    # Pydantic pattern enforces uppercase letters/dots only
    assert r.status_code == 422


def test_query_validation_rejects_bad_top_k(client):
    r = client.post("/v1/query", json={"question": "test", "top_k": 100})
    assert r.status_code == 422


def test_query_succeeds(client):
    r = client.post("/v1/query", json={"question": "What were Apple's net sales?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "citations" in body
    assert "metrics" in body
    assert body["metrics"]["total_latency_ms"] > 0
