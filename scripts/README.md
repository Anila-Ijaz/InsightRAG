# Operational scripts

Helper scripts for running, benchmarking, and validating the InsightRAG system.

| Script | Purpose |
|---|---|
| `bulk_ingest.py` | Ingest many tickers in parallel against a running API. Bounded concurrency to respect SEC EDGAR rate limits. |
| `export_chunks.py` | Dump indexed chunks from Qdrant to JSONL. Required input for reranker training and synthetic testset generation. |
| `generate_retrieval_testset.py` | LLM-generates `(query, relevant_chunk_id)` pairs for retrieval benchmarking. Filters out generic questions. |
| `smoke_test.py` | Post-deploy sanity check — hits health, query, streaming, guardrails, and validation endpoints. Suitable for CD pipelines. |
| `sample_tickers.txt` | Example ticker list for `bulk_ingest.py --tickers-file`. |

## Typical workflow

```bash
# 1. Ingest a portfolio of companies
python scripts/bulk_ingest.py --tickers-file scripts/sample_tickers.txt --limit 1

# 2. Verify the deployment
python scripts/smoke_test.py --base-url http://localhost:8000

# 3. Export chunks for training & evaluation
python scripts/export_chunks.py --output data/chunks.jsonl

# 4. Build a retrieval benchmark testset
python scripts/generate_retrieval_testset.py \
    --chunks data/chunks.jsonl \
    --output evals/retrieval_testset.jsonl \
    --n-queries 200

# 5. Run the retrieval benchmark
python evals/benchmark.py --testset evals/retrieval_testset.jsonl
```
