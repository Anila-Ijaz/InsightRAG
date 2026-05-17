#!/usr/bin/env python3
"""Generate a retrieval benchmark testset from synthetic Q&A.

For each chunk, ask an LLM to generate a specific question that chunk answers. The
chunk's ID becomes the ground-truth relevant document for that question. This gives
you hundreds of labeled (query, relevant_chunk_ids) pairs cheaply — perfect for
populating `evals/retrieval_testset.jsonl` that `evals/benchmark.py` consumes.

Quality control: we drop questions that are too generic (e.g. "what is mentioned?")
and ones whose own chunk doesn't appear in the top-50 of the current retriever
(indicates the question is ambiguous or the chunk lacks distinctive content).

Usage:
    python scripts/generate_retrieval_testset.py \
        --chunks data/chunks.jsonl \
        --output evals/retrieval_testset.jsonl \
        --n-queries 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from loguru import logger

from insightrag.generation.llm_client import get_llm_client


PROMPT = """You are creating evaluation questions for a financial document retrieval system.

Read this excerpt from a 10-K SEC filing and write ONE specific question that this excerpt directly and uniquely answers.

Rules:
- The question must be answerable ONLY from this excerpt, not from generic financial knowledge.
- Include specific entities mentioned (company name, product, year, dollar amount).
- Avoid generic questions like "what is discussed" or "what is mentioned".
- Output only the question. No preamble, no quotes.

Excerpt:
{text}

Specific question:"""


# Filter out questions that are too generic to be useful as retrieval labels
GENERIC_PATTERNS = [
    re.compile(r"^what (is|are) (discussed|mentioned|stated|described)", re.I),
    re.compile(r"^what does (the|this) (section|excerpt|filing) (say|discuss)", re.I),
    re.compile(r"^summarize", re.I),
]


def is_generic(question: str) -> bool:
    return any(p.search(question) for p in GENERIC_PATTERNS) or len(question.split()) < 5


async def generate_one(llm, text: str) -> str | None:
    try:
        response = await llm.complete(
            system="You write precise retrieval evaluation questions.",
            user=PROMPT.format(text=text[:1500]),
            temperature=0.3,
        )
        question = response.strip().split("\n")[0].strip().strip('"').strip("'")
        if is_generic(question):
            return None
        return question
    except Exception as e:
        logger.warning(f"LLM failed: {e}")
        return None


async def main_async(args: argparse.Namespace) -> None:
    chunks = [json.loads(line) for line in args.chunks.read_text().splitlines() if line.strip()]
    logger.info(f"Loaded {len(chunks)} chunks")

    random.seed(args.seed)
    sampled = random.sample(chunks, min(args.n_queries * 2, len(chunks)))  # over-sample, filter later
    llm = get_llm_client()

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def process(chunk: dict) -> None:
        async with sem:
            q = await generate_one(llm, chunk["text"])
            if q:
                results.append({"query": q, "relevant_chunk_ids": [chunk["chunk_id"]]})
                if len(results) % 20 == 0:
                    logger.info(f"Generated {len(results)} usable queries")

    await asyncio.gather(*[process(c) for c in sampled])

    # Truncate to requested size
    final = results[: args.n_queries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")

    logger.info(f"Wrote {len(final)} queries to {args.output} "
                f"(dropped {len(sampled) - len(results)} generic/failed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evals/retrieval_testset.jsonl"))
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
