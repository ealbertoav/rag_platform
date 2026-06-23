from __future__ import annotations

from prometheus_client import Counter, Histogram

# ── Metric definitions ─────────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Pipeline stage latency in seconds",
    labelnames=["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total pipeline requests",
    labelnames=["status"],
)

RETRIEVAL_CHUNK_COUNT = Histogram(
    "rag_retrieval_chunk_count",
    "Number of chunks returned after retrieval",
    buckets=[1, 3, 5, 10, 20, 50, 100],
)

LLM_TOKENS_TOTAL = Counter(
    "rag_llm_tokens_total",
    "Total tokens produced by the LLM",
)

# ── Recording helpers ──────────────────────────────────────────────────────────


def record_request(stage: str, latency_seconds: float, *, success: bool = True) -> None:
    """Record latency and increment the request counter for *stage*."""
    REQUEST_LATENCY.labels(stage=stage).observe(latency_seconds)
    REQUESTS_TOTAL.labels(status="success" if success else "error").inc()


def record_retrieval(chunk_count: int, latency_seconds: float) -> None:
    """Record retrieval-specific metrics."""
    RETRIEVAL_CHUNK_COUNT.observe(chunk_count)
    REQUEST_LATENCY.labels(stage="retrieval").observe(latency_seconds)


def record_generation(token_count: int, latency_seconds: float) -> None:
    """Record generation-specific metrics."""
    LLM_TOKENS_TOTAL.inc(token_count)
    REQUEST_LATENCY.labels(stage="generation").observe(latency_seconds)
