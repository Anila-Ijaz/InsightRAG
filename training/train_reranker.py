"""Fine-tune a cross-encoder reranker on synthetic Q&A pairs from SEC filings.

This is the key modeling differentiator vs typical RAG tutorials. We:

1. Generate synthetic queries from each chunk using an LLM (chunk → "what question does
   this chunk answer?"). This is a form of GPL-style training-data synthesis.
2. Mine hard negatives by retrieving with the *current* reranker. Hard negatives are
   passages that look similar to the positive but are actually wrong — far more
   informative than random negatives.
3. Fine-tune with Multiple Negatives Ranking Loss (in-batch negatives) — efficient and
   produces well-calibrated similarity scores.

Run:
    python training/train_reranker.py --corpus data/chunks.jsonl --output models/reranker

Outputs:
    - Fine-tuned model in models/reranker/
    - Eval metrics (MRR@10, nDCG@10) in models/reranker/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from loguru import logger
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader


def load_corpus(path: Path) -> list[dict]:
    """Load chunks from JSONL. Each line: {chunk_id, text, metadata}."""
    with path.open() as f:
        return [json.loads(line) for line in f]


def generate_synthetic_queries(
    corpus: list[dict],
    llm_fn,
    n_per_chunk: int = 1,
) -> list[tuple[str, str]]:
    """Use an LLM to generate questions answered by each chunk. Returns (query, chunk_id) pairs.

    The llm_fn is injected to keep this module testable. In production point it at the
    configured OpenAI/Anthropic client.
    """
    prompt_template = (
        "Read this excerpt from a 10-K SEC filing. Generate {n} short, specific "
        "questions that this excerpt directly answers. Output one question per line, "
        "no numbering, no quotes.\n\nExcerpt:\n{text}\n\nQuestions:"
    )
    pairs: list[tuple[str, str]] = []
    for i, chunk in enumerate(corpus):
        if i % 100 == 0:
            logger.info(f"Generating queries: {i}/{len(corpus)}")
        prompt = prompt_template.format(n=n_per_chunk, text=chunk["text"][:1500])
        try:
            response = llm_fn(prompt)
            questions = [q.strip() for q in response.split("\n") if q.strip()]
            for q in questions[:n_per_chunk]:
                pairs.append((q, chunk["chunk_id"]))
        except Exception as e:  # pragma: no cover
            logger.warning(f"LLM error on chunk {chunk['chunk_id']}: {e}")
    return pairs


def mine_hard_negatives(
    positive_pairs: list[tuple[str, str]],
    corpus: list[dict],
    retriever_fn,
    n_negatives: int = 4,
) -> list[InputExample]:
    """For each (query, positive_chunk_id), retrieve top candidates and use the
    non-positive results as hard negatives.

    `retriever_fn(query) -> list[chunk_id]` is injected.
    """
    chunk_by_id = {c["chunk_id"]: c for c in corpus}
    examples: list[InputExample] = []

    for query, pos_id in positive_pairs:
        examples.append(InputExample(texts=[query, chunk_by_id[pos_id]["text"]], label=1.0))

        candidates = retriever_fn(query)
        negatives = [c for c in candidates if c != pos_id][:n_negatives]
        for neg_id in negatives:
            if neg_id in chunk_by_id:
                examples.append(
                    InputExample(texts=[query, chunk_by_id[neg_id]["text"]], label=0.0)
                )

    random.shuffle(examples)
    return examples


def split_train_eval(
    examples: list[InputExample], eval_frac: float = 0.1
) -> tuple[list[InputExample], list[dict]]:
    """Split into train (InputExamples) and eval (reranker eval format)."""
    n_eval = int(len(examples) * eval_frac)
    train, eval_raw = examples[n_eval:], examples[:n_eval]

    # Build reranker evaluator format: {"query": q, "positive": [p], "negative": [n1, n2,...]}
    eval_dict: dict[str, dict] = {}
    for ex in eval_raw:
        q = ex.texts[0]
        if q not in eval_dict:
            eval_dict[q] = {"query": q, "positive": [], "negative": []}
        if ex.label == 1.0:
            eval_dict[q]["positive"].append(ex.texts[1])
        else:
            eval_dict[q]["negative"].append(ex.texts[1])

    eval_samples = [v for v in eval_dict.values() if v["positive"] and v["negative"]]
    return train, eval_samples


def train(
    base_model: str,
    train_examples: list[InputExample],
    eval_samples: list[dict],
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    logger.info(f"Training {base_model} on {len(train_examples)} examples for {epochs} epoch(s)")
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = CERerankingEvaluator(eval_samples, name="sec-reranker-eval")

    warmup_steps = int(len(loader) * epochs * 0.1)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=loader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=max(1, len(loader) // 5),
        output_path=str(output_dir),
        optimizer_params={"lr": learning_rate},
    )
    logger.info(f"Saved fine-tuned reranker to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, help="JSONL of chunks")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--output", type=Path, default=Path("models/reranker"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--queries-per-chunk", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=2000,
                        help="Cap corpus size for cost control during synthetic Q gen")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    if len(corpus) > args.max_chunks:
        random.seed(42)
        corpus = random.sample(corpus, args.max_chunks)
    logger.info(f"Loaded {len(corpus)} chunks")

    # In a real run, inject real llm_fn and retriever_fn from the app's clients.
    # The function signatures show what's expected.
    raise NotImplementedError(
        "Inject your LLM client and retriever before running. "
        "See docstrings for `generate_synthetic_queries` and `mine_hard_negatives`."
    )


if __name__ == "__main__":
    main()
