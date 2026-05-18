"""Input-side guardrails.

We do three things on incoming user queries:
1. Length check — reject excessively long queries (cost + DoS protection)
2. Prompt injection detection — heuristic patterns (good enough for portfolio; production
   would add a classifier like Lakera or a fine-tuned BERT model)
3. PII redaction — strip emails/SSNs/credit cards before sending to the LLM provider
"""
from __future__ import annotations

import re

from loguru import logger

from insightrag.config import get_settings


# Heuristic prompt-injection patterns. Not exhaustive — defense in depth requires
# also separating context from instructions structurally (which we do in prompts.py).
INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (the |all )?(system|above|prior)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"new instructions?:", re.I),
    re.compile(r"```\s*system", re.I),
    re.compile(r"</?system>", re.I),
    re.compile(r"forget (everything|all instructions)", re.I),
]


class PromptInjectionDetected(Exception):
    """Raised when a query is flagged as a probable prompt injection attempt."""


class InputGuard:
    def __init__(self, max_length: int, enable_pii: bool, enable_injection: bool):
        self.max_length = max_length
        self.enable_pii = enable_pii
        self.enable_injection = enable_injection
        self._analyzer = None
        self._anonymizer = None
        if enable_pii:
            self._lazy_init_presidio()

    def _lazy_init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Presidio init failed, disabling PII redaction: {e}")
            self.enable_pii = False

    def process(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Empty query")

        if len(text) > self.max_length:
            raise ValueError(f"Query exceeds max length of {self.max_length} characters")

        if self.enable_injection:
            for pattern in INJECTION_PATTERNS:
                if pattern.search(text):
                    logger.warning(f"Prompt injection detected: pattern={pattern.pattern}")
                    raise PromptInjectionDetected(
                        "Your query contains patterns that look like a prompt injection attempt."
                    )

        if self.enable_pii and self._analyzer and self._anonymizer:
            results = self._analyzer.analyze(text=text, language="en")
            if results:
                anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
                logger.info(f"Redacted {len(results)} PII entities from query")
                return anonymized.text

        return text


def get_input_guard() -> InputGuard:
    s = get_settings()
    return InputGuard(
        max_length=s.max_query_length,
        enable_pii=s.enable_pii_redaction,
        enable_injection=s.enable_injection_detection,
    )
