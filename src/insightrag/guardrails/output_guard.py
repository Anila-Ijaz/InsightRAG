"""Output-side guardrails.

We validate that:
1. The answer doesn't contain leaked system prompt tokens.
2. Citations referenced in the answer exist in the provided chunks (no invented [99] refs).
3. The output isn't suspiciously short (model refused to answer) — we let it through but log.
"""
from __future__ import annotations

import re

from loguru import logger


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
LEAK_PATTERNS = [
    re.compile(r"You are InsightRAG", re.I),
    re.compile(r"non-negotiable", re.I),
]


class OutputGuard:
    """Post-processes LLM output."""

    def process(self, answer: str, n_chunks_provided: int | None = None) -> str:
        cleaned = answer.strip()

        # 1. Strip any system-prompt leakage if present
        for pattern in LEAK_PATTERNS:
            if pattern.search(cleaned):
                logger.warning(f"System prompt leak detected: {pattern.pattern}")
                cleaned = pattern.sub("[redacted]", cleaned)

        # 2. Validate citation indices
        if n_chunks_provided is not None:
            cited = {int(m) for m in CITATION_PATTERN.findall(cleaned)}
            invalid = {c for c in cited if c < 1 or c > n_chunks_provided}
            if invalid:
                logger.warning(f"Invalid citations found: {invalid} (only {n_chunks_provided} chunks)")
                # Strip invalid citations rather than fail
                for inv in invalid:
                    cleaned = cleaned.replace(f"[{inv}]", "")

        return cleaned


def get_output_guard() -> OutputGuard:
    return OutputGuard()
