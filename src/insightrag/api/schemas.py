"""Request/response schemas. Strict pydantic models — no untyped dicts cross the API boundary."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    ticker: str | None = Field(None, description="Filter to a specific company ticker")
    section: str | None = Field(None, description="Filter to a specific 10-K section")
    top_k: int = Field(5, ge=1, le=20)


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    ticker: str
    filing_date: str
    section: str
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    metrics: dict[str, float]


class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Z\.]+$")
    limit: int = Field(1, ge=1, le=5)


class IngestResponse(BaseModel):
    ticker: str
    filings_ingested: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant: bool
    embedder: bool
