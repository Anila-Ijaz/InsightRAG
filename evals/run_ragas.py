"""RAGAS-based evaluation of the RAG pipeline.

RAGAS provides reference-free metrics that don't require a labeled golden answer:
  - faithfulness: is every claim in the answer supported by the retrieved context?
  - answer_relevancy: does the answer actually address the question?
  - context_precision: are the relevant chunks ranked higher?
  - context_recall: did we retrieve everything we needed (requires ground truth)?

This script runs the eval set through the live API and prints + saves results.
Plug this into CI to catch retrieval regressions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd
from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


async def run_query(client: httpx.AsyncClient, base_url: str, question: str) -> dict:
    resp = await client.post(
        f"{base_url}/v1/query",
        json={"question": question, "top_k": 5},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


async def collect_predictions(testset: list[dict], base_url: str) -> list[dict]:
    """Hit the running API for each test question and collect predictions."""
    out: list[dict] = []
    async with httpx.AsyncClient() as client:
        for i, row in enumerate(testset):
            logger.info(f"Query {i + 1}/{len(testset)}: {row['question'][:80]}")
            try:
                pred = await run_query(client, base_url, row["question"])
                out.append({
                    "question": row["question"],
                    "answer": pred["answer"],
                    "contexts": [c["text_preview"] for c in pred["citations"]],
                    "ground_truth": row.get("ground_truth", ""),
                })
            except Exception as e:
                logger.error(f"Query failed: {e}")
                out.append({
                    "question": row["question"],
                    "answer": "",
                    "contexts": [],
                    "ground_truth": row.get("ground_truth", ""),
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", type=Path, default=Path("evals/testset.jsonl"))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path("evals/results.json"))
    args = parser.parse_args()

    testset = [json.loads(line) for line in args.testset.read_text().splitlines() if line.strip()]
    logger.info(f"Running eval on {len(testset)} questions")

    predictions = asyncio.run(collect_predictions(testset, args.base_url))
    dataset = Dataset.from_list(predictions)

    metrics = [faithfulness, answer_relevancy, context_precision]
    # context_recall requires non-empty ground_truth on every row
    if all(p["ground_truth"] for p in predictions):
        metrics.append(context_recall)

    result = evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output.with_suffix(".csv"), index=False)
    summary = {m.name: float(df[m.name].mean()) for m in metrics if m.name in df.columns}
    args.output.write_text(json.dumps(summary, indent=2))

    logger.info("=== Eval Summary ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {v:.4f}")
    pd.set_option("display.max_colwidth", 80)
    print(df.head())


if __name__ == "__main__":
    main()
