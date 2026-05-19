"""Prometheus metrics.

These get scraped by Prometheus at /metrics and visualized in Grafana.
Conventions follow Prometheus naming: <name>_<unit>{labels}.
"""
from prometheus_client import Counter, Histogram

QUERY_COUNTER = Counter(
    "insightrag_queries_total",
    "Total number of queries by outcome and streaming mode",
    labelnames=("status", "streaming"),
)

QUERY_LATENCY = Histogram(
    "insightrag_query_latency_seconds",
    "End-to-end query latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

RETRIEVAL_LATENCY = Histogram(
    "insightrag_retrieval_latency_seconds",
    "Retrieval-only latency (hybrid search)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

LLM_TOKEN_COUNTER = Counter(
    "insightrag_llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=("provider", "model", "kind"),  # kind: prompt | completion
)
