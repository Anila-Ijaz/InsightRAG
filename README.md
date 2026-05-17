# InsightRAG

**Production-grade Retrieval-Augmented Generation system for SEC 10-K financial filings.**

[![CI](https://github.com/yourusername/insightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/insightrag/actions)
[![Coverage](https://codecov.io/gh/yourusername/insightrag/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/insightrag)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

InsightRAG answers natural-language questions over the SEC EDGAR corpus of 10-K filings with cited, grounded responses. Built to demonstrate the engineering decisions that separate a tutorial from a production RAG system.

> *"What were the principal drivers of Apple's gross margin change in fiscal 2023?"* → answer with citations to the exact MD&A paragraphs.

---

## What makes this different from typical RAG demos

Most public RAG projects stop at "embed → store → retrieve → generate". The interesting engineering is everywhere else. This repo demonstrates:

| Concern | Naive approach | This repo |
|---|---|---|
| **Retrieval** | Vector-only similarity | Hybrid (dense BGE + BM25) with RRF fusion |
| **Reranking** | None | Fine-tuned cross-encoder (BGE-reranker) on synthetic Q&A + hard negatives |
| **Chunking** | Fixed-size character splits | Recursive, token-aware, structure-preserving (10-K section metadata) |
| **Prompt safety** | Concatenate and pray | Layered guards: length, injection patterns, PII redaction (Presidio), output citation validation |
| **Evaluation** | "Looks good to me" | RAGAS in nightly CI + retrieval benchmark (MRR/nDCG/Recall) across 4 configurations |
| **Observability** | Print statements | Structured JSON logs (loguru), Prometheus metrics, Langfuse traces |
| **Deployment** | Single `python app.py` | Multi-stage Docker, compose stack, GitHub Actions CI/CD, GHCR registry |

---

## Architecture

```
                                                ┌────────────────────┐
                                                │  SEC EDGAR (10-K)  │
                                                └─────────┬──────────┘
                                                          │ download
                                                          ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                          INGESTION PIPELINE                          │
   │  Parser (SGML→HTML→sections) → Semantic Chunker (token-aware) →     │
   │  Embedder (BGE)                                                      │
   └────────────────┬────────────────────────────────┬───────────────────┘
                    ▼                                ▼
            ┌───────────────┐                ┌─────────────────┐
            │  Qdrant       │                │  BM25 index     │
            │  (dense)      │                │  (sparse)       │
            └───────┬───────┘                └────────┬────────┘
                    │                                 │
                    └────────────┬────────────────────┘
                                 ▼
                       ┌──────────────────────┐
                       │  Hybrid Retriever    │  ← Reciprocal Rank Fusion
                       │  (top-20 candidates) │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  Cross-Encoder       │  ← Fine-tuned on SEC corpus
                       │  Reranker (top-5)    │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  LLM Generation      │  ← OpenAI/Anthropic, streamable
                       │  with citation guard │
                       └──────────┬───────────┘
                                  ▼
                            FastAPI (SSE)
```

---

## Tech stack

**Core:** Python 3.11, FastAPI (async), Pydantic v2
**Retrieval:** sentence-transformers (BGE), Qdrant, rank-bm25
**Reranker:** BGE-reranker-base (fine-tuned via `training/train_reranker.py`)
**Generation:** OpenAI / Anthropic SDK (provider-abstracted)
**Storage:** Postgres (metadata), Redis (cache), S3 (raw docs)
**Async tasks:** Celery
**Guardrails:** Presidio (PII), regex pattern matching (injection), output citation validation
**Observability:** loguru (structured logs), prometheus-client, OpenTelemetry, Langfuse
**Eval:** RAGAS, custom retrieval benchmark (MRR/nDCG/Recall@k)
**Infra:** Docker (multi-stage), docker-compose, GitHub Actions, GHCR
**Tests:** pytest, pytest-asyncio, ruff, mypy

---

## Quick start

```bash
# 1. Clone and set env
git clone https://github.com/yourusername/insightrag
cd insightrag
cp .env.example .env
# edit .env: set OPENAI_API_KEY

# 2. Bring up the full stack
make up

# 3. Ingest a sample 10-K
make ingest TICKER=AAPL

# 4. Query
curl -X POST http://localhost:8000/v1/query \
  -H "content-type: application/json" \
  -d '{"question":"What were Apple total net sales?","ticker":"AAPL","top_k":5}'

# 5. Streaming
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "content-type: application/json" \
  -d '{"question":"Summarize the principal risk factors","ticker":"AAPL"}'
```

Open:
- API docs: http://localhost:8000/docs
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

---

## Evaluation results

> Replace these numbers with your actual eval output after running `make eval`. The benchmarking script is in `evals/benchmark.py`.

### Retrieval (n=200 labeled queries from synthetic + manual annotations)

| Configuration | MRR@10 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| Dense only (BGE) | TBD | TBD | TBD |
| Sparse only (BM25) | TBD | TBD | TBD |
| Hybrid (RRF) | TBD | TBD | TBD |
| Hybrid + fine-tuned reranker | **TBD** | **TBD** | **TBD** |

### End-to-end RAG (RAGAS, n=50 questions)

| Metric | Score |
|---|---:|
| Faithfulness | TBD |
| Answer Relevancy | TBD |
| Context Precision | TBD |
| Context Recall | TBD |

### Latency (p50/p95, 4-core CPU, gpt-4o-mini)

| Stage | p50 | p95 |
|---|---:|---:|
| Retrieval (hybrid) | TBD | TBD |
| Rerank (CPU) | TBD | TBD |
| Generation (streaming, time-to-first-token) | TBD | TBD |
| **Total** | TBD | TBD |

---

## Design decisions (and why)

### Why hybrid retrieval, not just vector search?
Dense embeddings miss exact-match signals critical in financial filings — specific dollar amounts, dates, ticker symbols, product codenames. BM25 catches these. Each alone produces visible failure modes that diverge; together via RRF they cover each other's blind spots. The retrieval benchmark above shows the lift.

### Why Reciprocal Rank Fusion over weighted score fusion?
RRF is parameter-free (no `alpha` to tune per corpus), tolerant of score-distribution differences between rankers, and consistently strong across published benchmarks (Cormack et al., 2009). Weighted fusion is implemented for ablation studies.

### Why fine-tune the reranker?
Base BGE-reranker is trained on MS-MARCO. Financial filings have distinct terminology — fine-tuning on synthetic Q&A pairs generated from your own corpus pushes nDCG@10 meaningfully on in-domain queries. Hard negatives mined from the existing retriever close the loop.

### Why a separate output guard?
LLMs occasionally invent citations like `[99]` when only 5 chunks were provided. The output guard strips invalid indices before the response leaves the API. Cheap, robust, defense-in-depth.

### Why streaming with SSE, not WebSockets?
SSE is HTTP-native (works through CDNs, load balancers, proxies without special config), unidirectional which fits our use case, and trivially consumable from any client (`fetch` with `EventSource`). WebSockets would be overkill.

### Why provider-abstract the LLM?
Production systems get burned by vendor lock-in and provider outages. The `LLMClient` ABC lets you switch from OpenAI to Anthropic via one env var, and makes it trivial to add a self-hosted Llama fallback later.

---

## Project structure

```
insightrag/
├── src/insightrag/
│   ├── ingestion/          # SEC parser, semantic chunker, embedder
│   ├── retrieval/          # Qdrant store, BM25 index, hybrid retriever, reranker
│   ├── generation/         # LLM client (OpenAI/Anthropic), prompts, RAG chain
│   ├── guardrails/         # Input (injection, PII) + output (citation validation)
│   ├── api/                # FastAPI app, schemas, dependencies
│   ├── observability/      # Logging, Prometheus metrics
│   └── config.py           # pydantic-settings
├── training/               # Reranker fine-tuning (synthetic Q&A + hard negatives)
├── evals/                  # RAGAS runner + retrieval benchmark + test set
├── tests/                  # pytest suite (chunker, retrieval, guardrails, API)
├── infra/                  # Prometheus config, k8s manifests, terraform
├── .github/workflows/      # ci.yml (lint/test/build), eval.yml (nightly RAGAS)
├── Dockerfile              # Multi-stage build, non-root user, healthcheck
├── docker-compose.yml      # api + qdrant + postgres + redis + prom + grafana
├── pyproject.toml          # Dependencies, ruff/mypy/pytest config
└── Makefile                # install / dev / test / eval / up / down / ingest
```

---

## Roadmap

- [ ] Streamlit / Next.js chat UI
- [ ] Caching layer for embeddings + LLM responses (Redis)
- [ ] Add table-aware chunking (10-Ks have lots of tables)
- [ ] Citation hover preview in UI showing the source paragraph
- [ ] Multi-document comparison ("compare AAPL vs MSFT risk factors")
- [ ] Self-hosted Llama 3 fallback via vLLM

---

## License

MIT — see [LICENSE](LICENSE).
