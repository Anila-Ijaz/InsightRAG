# InsightRAG

**Production-grade Retrieval-Augmented Generation system for SEC 10-K financial filings.**

[![CI](https://github.com/Anila-Ijaz/insightrag/actions/workflows/ci.yml/badge.svg)](https://github.com/Anila-Ijaz/insightrag/actions)
[![Coverage](https://codecov.io/gh/Anila-Ijaz/insightrag/branch/main/graph/badge.svg)](https://codecov.io/gh/Anila-Ijaz/insightrag)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

InsightRAG answers natural-language questions over the SEC EDGAR corpus of 10-K filings with cited, grounded responses. Built to demonstrate the engineering decisions that separate a tutorial from a production RAG system.

> *"What were the principal drivers of Apple's gross margin change in fiscal 2023?"* → answer with citations to the exact MD&A paragraphs.

It ships with a **Streamlit chat UI** and runs in two profiles:

| Profile | What it shows | Footprint | Use |
|---|---|---|---|
| **Full** (`docker-compose.yml`) | Local BGE embeddings + cross-encoder reranker, full observability stack | ~5 GB image, 6–8 GB RAM | Showcase the complete retrieval engineering, run locally |
| **Lite** (`docker-compose.lite.yml`) | OpenAI embeddings, reranker off, just `api + qdrant + UI` | ~508 MB image, <1 GB RAM | Free-tier cloud deploy + the live demo |

The two share one codebase; the embedding provider and reranker are switched by env (`EMBEDDING_PROVIDER`, `ENABLE_RERANKER`). The lite profile is what gets deployed to AWS free tier.

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
**Frontend:** Streamlit chat UI (citations + live latency panel)
**Embeddings:** local BGE (sentence-transformers) **or** OpenAI API — provider-switchable
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

### Lite profile (recommended — chat UI, no GPU, ~1 GB RAM)

```bash
git clone https://github.com/Anila-Ijaz/insightrag
cd insightrag
cp .env.lite.example .env          # then set OPENAI_API_KEY in .env

docker compose -f docker-compose.lite.yml up -d --build
```

Then open the **chat UI** and ask away:

- **Chat UI:**  http://127.0.0.1:8501   *(use `127.0.0.1`, not `localhost`, if your Docker has an IPv6 quirk)*
- **API docs:** http://127.0.0.1:8000/docs

Ingest a filing from the UI sidebar, or via the API:

```bash
# Index Apple's latest 10-K from SEC EDGAR
curl -X POST http://127.0.0.1:8000/v1/ingest \
  -H "content-type: application/json" \
  -d '{"ticker":"AAPL","limit":1}'

# Ask a question
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "content-type: application/json" \
  -d '{"question":"What were Apple total net sales?","ticker":"AAPL","top_k":5}'
```

### Full profile (local BGE + cross-encoder reranker + observability)

```bash
cp .env.example .env               # set OPENAI_API_KEY
make up                            # api + qdrant + postgres + redis + prometheus + grafana
make ingest TICKER=AAPL
```

> Note: the full profile downloads local ML models (torch + sentence-transformers) and needs ~6–8 GB RAM.

---

## Evaluation results

### Observed latency — lite profile, gpt-4o-mini

Measured end-to-end on the lite stack (OpenAI embeddings, reranker off), single AAPL 10-K indexed (110 chunks):

| Stage | Observed |
|---|---:|
| Retrieval (hybrid, OpenAI embed + Qdrant + BM25) | ~1.1–2.3 s |
| Rerank | 0 ms *(disabled in lite)* |
| Generation (gpt-4o-mini) | ~1.5–3.3 s |
| **Total** | **~3.8–4.4 s** |

> These are wall-clock observations (n≈2), not statistical p50/p95. Retrieval is dominated by two sequential network calls (OpenAI embedding + Qdrant). The full profile's local BGE embeddings remove the embedding round-trip.

### Retrieval & answer quality — not yet benchmarked

The harnesses are in [`evals/`](evals/) (retrieval MRR/nDCG/Recall and RAGAS faithfulness/relevancy), but the labeled test set is small and these tables are **not yet populated** — running them on a full corpus is the next milestone.

| Configuration | MRR@10 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|
| Dense only (BGE) | _pending_ | _pending_ | _pending_ |
| Sparse only (BM25) | _pending_ | _pending_ | _pending_ |
| Hybrid (RRF) | _pending_ | _pending_ | _pending_ |
| Hybrid + fine-tuned reranker | _pending_ | _pending_ | _pending_ |

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
│   └── config.py           # pydantic-settings (provider switches)
├── frontend/               # Streamlit chat UI (app.py + Dockerfile)
├── training/               # Reranker fine-tuning (synthetic Q&A + hard negatives)
├── evals/                  # RAGAS runner + retrieval benchmark + test set
├── tests/                  # pytest suite (chunker, retrieval, guardrails, API)
├── infra/                  # Prometheus config, k8s manifests, terraform
├── .github/workflows/      # ci.yml (lint/test/build), eval.yml (nightly RAGAS)
├── Dockerfile              # Full image (local ML models)
├── Dockerfile.lite         # Lite image (~508 MB, OpenAI embeddings, no torch)
├── docker-compose.yml      # Full: api + qdrant + postgres + redis + prom + grafana
├── docker-compose.lite.yml # Lite: api + qdrant + Streamlit UI
├── requirements-lite.txt   # Lite runtime deps (subset of pyproject)
├── pyproject.toml          # Dependencies, ruff/mypy/pytest config
└── Makefile                # install / dev / test / eval / up / down / ingest
```

---

## Roadmap

- [x] Streamlit chat UI
- [ ] Deploy lite profile to AWS free tier (in progress)
- [ ] Populate retrieval + RAGAS benchmark tables on a full corpus
- [ ] Caching layer for embeddings + LLM responses (Redis)
- [ ] Add table-aware chunking (10-Ks have lots of tables)
- [ ] Citation hover preview in UI showing the source paragraph
- [ ] Multi-document comparison ("compare AAPL vs MSFT risk factors")
- [ ] Self-hosted Llama 3 fallback via vLLM

---

## License

MIT — see [LICENSE](LICENSE).
