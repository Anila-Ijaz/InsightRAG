"""End-to-end RAG chain orchestration.

This is the main entry point that ties retrieval → reranking → generation together.
Designed for both batch (complete) and streaming (stream) modes.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter

from loguru import logger

from insightrag.generation.llm_client import LLMClient
from insightrag.generation.prompts import build_rag_prompt
from insightrag.guardrails.input_guard import InputGuard
from insightrag.guardrails.output_guard import OutputGuard
from insightrag.retrieval.hybrid import HybridRetriever, RetrievedChunk
from insightrag.retrieval.reranker import CrossEncoderReranker


@dataclass
class Citation:
    index: int
    chunk_id: str
    ticker: str
    filing_date: str
    section: str
    text_preview: str


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            index=i + 1,
            chunk_id=c.chunk_id,
            ticker=c.metadata.get("ticker", "?"),
            filing_date=c.metadata.get("filing_date", "?"),
            section=c.metadata.get("section", "?"),
            text_preview=c.text[:200] + ("…" if len(c.text) > 200 else ""),
        )
        for i, c in enumerate(chunks)
    ]


class RAGChain:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm: LLMClient,
        input_guard: InputGuard,
        output_guard: OutputGuard,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.input_guard = input_guard
        self.output_guard = output_guard
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

    async def _retrieve_and_rerank(
        self, query: str, filter_payload: dict | None
    ) -> tuple[list[RetrievedChunk], float, float]:
        t0 = perf_counter()
        candidates = await self.retriever.retrieve(
            query, top_k=self.retrieval_top_k, filter_payload=filter_payload
        )
        retrieval_ms = (perf_counter() - t0) * 1000

        t1 = perf_counter()
        top_chunks = self.reranker.rerank(query, candidates, top_k=self.rerank_top_k)
        rerank_ms = (perf_counter() - t1) * 1000

        return top_chunks, retrieval_ms, rerank_ms

    async def answer(self, question: str, filter_payload: dict | None = None) -> RAGResponse:
        t_start = perf_counter()
        question = self.input_guard.process(question)

        top_chunks, retrieval_ms, rerank_ms = await self._retrieve_and_rerank(question, filter_payload)
        system, user = build_rag_prompt(question, top_chunks)

        t_gen = perf_counter()
        raw_answer = await self.llm.complete(system, user)
        gen_ms = (perf_counter() - t_gen) * 1000

        clean_answer = self.output_guard.process(raw_answer)
        total_ms = (perf_counter() - t_start) * 1000

        logger.info(
            f"RAG completed: retrieval={retrieval_ms:.0f}ms rerank={rerank_ms:.0f}ms "
            f"gen={gen_ms:.0f}ms total={total_ms:.0f}ms chunks={len(top_chunks)}"
        )

        return RAGResponse(
            answer=clean_answer,
            citations=_build_citations(top_chunks),
            retrieval_latency_ms=retrieval_ms,
            rerank_latency_ms=rerank_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=total_ms,
        )

    async def stream(
        self, question: str, filter_payload: dict | None = None
    ) -> AsyncIterator[dict]:
        """Stream the answer as it's generated. Yields dicts:
        - {"type": "citations", "data": [...]}  (sent once, up front)
        - {"type": "delta", "data": "text..."}  (many)
        - {"type": "done", "data": {...metrics}} (sent once, at end)
        """
        t_start = perf_counter()
        question = self.input_guard.process(question)

        top_chunks, retrieval_ms, rerank_ms = await self._retrieve_and_rerank(question, filter_payload)
        system, user = build_rag_prompt(question, top_chunks)

        yield {
            "type": "citations",
            "data": [c.__dict__ for c in _build_citations(top_chunks)],
        }

        t_gen = perf_counter()
        buffer: list[str] = []
        async for delta in self.llm.stream(system, user):
            buffer.append(delta)
            # Stream raw deltas; final cleanup happens once we have the full answer.
            yield {"type": "delta", "data": delta}
        gen_ms = (perf_counter() - t_gen) * 1000

        full_answer = self.output_guard.process("".join(buffer))
        total_ms = (perf_counter() - t_start) * 1000
        yield {
            "type": "done",
            "data": {
                "full_answer": full_answer,
                "retrieval_latency_ms": retrieval_ms,
                "rerank_latency_ms": rerank_ms,
                "generation_latency_ms": gen_ms,
                "total_latency_ms": total_ms,
            },
        }
