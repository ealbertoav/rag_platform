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

EMBEDDING_CACHE_HITS = Counter(
    "rag_embedding_cache_hits_total",
    "Total embedding vectors served from Redis cache",
)

EMBEDDING_CACHE_MISSES = Counter(
    "rag_embedding_cache_misses_total",
    "Total embedding vectors computed (cache miss)",
)

RATE_LIMIT_REJECTED = Counter(
    "rag_rate_limit_rejected_total",
    "Total requests rejected by API rate limiting",
    labelnames=["path"],
)

# ── Retrieval/answer-quality signal (#92) ───────────────────────────────────────

RERANKER_OUTCOME_TOTAL = Counter(
    "rag_reranker_outcome_total",
    "Reranker outcomes — 'reranked' (scored successfully) vs 'fallback' "
    "(scoring failed, raw retrieval order returned)",
    labelnames=["outcome"],
)

RERANKER_SCORE = Histogram(
    "rag_reranker_score",
    "Cross-encoder relevance scores for successfully reranked chunks",
    buckets=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
)

RELIABLE_RAG_RELEVANCE_SCORE = Histogram(
    "rag_reliable_rag_relevance_score",
    "Reliable RAG per-chunk relevance grades (0-1) when quality.reliable_rag.enabled",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

FEEDBACK_EVENTS_TOTAL = Counter(
    "rag_feedback_events_total",
    "Retrieval feedback submissions by sentiment, when quality.feedback_loop.enabled",
    labelnames=["sentiment"],
)

FEEDBACK_SCORE_ACCUMULATED = Histogram(
    "rag_feedback_score_accumulated",
    "Per-chunk accumulated feedback score after each update, when quality.feedback_loop.enabled",
    buckets=[-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0],
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


def record_rate_limit_rejection(path: str) -> None:
    """Increment the rate-limit rejection counter for a *path*."""
    RATE_LIMIT_REJECTED.labels(path=path).inc()


def record_reranker_success(scores: list[float]) -> None:
    """Record a successful reranker pass and its raw cross-encoder scores."""
    RERANKER_OUTCOME_TOTAL.labels(outcome="reranked").inc()
    for score in scores:
        RERANKER_SCORE.observe(score)


def record_reranker_fallback() -> None:
    """Record that the reranker failed and raw retrieval order was used instead."""
    RERANKER_OUTCOME_TOTAL.labels(outcome="fallback").inc()


def record_reliable_rag_scores(scores: list[float]) -> None:
    """Record Reliable RAG per-chunk relevance grades (pass + fail)."""
    for score in scores:
        RELIABLE_RAG_RELEVANCE_SCORE.observe(score)


def record_feedback_event(sentiment: str, accumulated: float) -> None:
    """Record a retrieval feedback submission and the chunk's new accumulated score.

    *sentiment* ("positive"/"negative") is classified by the caller — see
    src.rag.quality.feedback_loop.sentiment_from_score() — so this module doesn't
    duplicate that domain rule.
    """
    FEEDBACK_EVENTS_TOTAL.labels(sentiment=sentiment).inc()
    FEEDBACK_SCORE_ACCUMULATED.observe(accumulated)
