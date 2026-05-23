# Exploratory notebooks

These notebooks back up the design decisions documented in the main README with actual data.
They are **not** part of the production code path — they live alongside it as evidence.

| Notebook | What it answers |
|---|---|
| `01_chunking_ablation.ipynb` | Why is the default `chunk_size=512`? Sweeps {128, 256, 512, 1024} and measures recall vs cost. |
| `02_retrieval_comparison.ipynb` | When does dense win, when does sparse win, what does the reranker add? Per-query win analysis. |
| `03_error_analysis.ipynb` | When the system fails, *why*? Categorizes failures into retrieval miss / hallucination / refusal-when-shouldn't / etc. |

## How to use these

1. Run the system end-to-end first: `make up && make ingest TICKER=AAPL`
2. Generate a labeled testset: `python scripts/generate_retrieval_testset.py --chunks data/chunks.jsonl`
3. Run the benchmarks: `python evals/benchmark.py` and `python evals/run_ragas.py`
4. Open each notebook and re-execute — the placeholder data gets replaced with your real numbers
5. Commit the executed notebooks so they render on GitHub

## Why these exist

A recruiter scanning a portfolio repo will open one notebook. If they see real ablation
numbers with clear takeaways, you're a different candidate than someone with code-only repos.
These three notebooks are arranged to tell a story:

- **#1 justifies a hyperparameter** — shows you tune deliberately, not by feel
- **#2 justifies an architecture choice** — shows you understand when components matter
- **#3 surfaces failure modes** — shows post-deployment maturity and iteration discipline

Treat them as living documents. Re-run them each release; track how the distributions shift.
