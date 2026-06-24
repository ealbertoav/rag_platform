"""Google Gemini embedding provider (dense only, API-based)."""

from __future__ import annotations

import logging
from typing import Literal

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

logger = logging.getLogger(__name__)

_MAX_BATCH = 100  # Gemini batch limit
GeminiTaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def _is_rate_limit(exc: BaseException) -> bool:
    try:
        from google.api_core.exceptions import ResourceExhausted
        if isinstance(exc, ResourceExhausted):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    keywords = ("429", "quota", "resource_exhausted", "rate_limit", "too many requests")
    return any(kw in msg for kw in keywords)


class GeminiEmbeddingProvider(EmbeddingRepository):
    """Google Gemini text-embedding-004 provider.

    Dense only (768-dim) — sparse vectors fall back to BM25.
    Uses task_type="RETRIEVAL_DOCUMENT" for ingestion and
    "RETRIEVAL_QUERY" for query embedding.
    Retries on quota exhaustion via tenacity.

    genai.configure() is called immediately before each API request rather than
    once at construction time. This avoids global-state conflicts when multiple
    GeminiEmbeddingProvider instances with different API keys coexist (e.g. in
    the provider comparison script or in tests).
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        self.api_key = api_key
        self.model = model

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Embed texts for document storage (task_type=RETRIEVAL_DOCUMENT)."""
        return self._embed_with_task(texts, "RETRIEVAL_DOCUMENT")

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{} for _ in texts]

    def embed_query(self, texts: list[str]) -> list[DenseVector]:
        """Embed query texts (task_type=RETRIEVAL_QUERY). Use during retrieval."""
        return self._embed_with_task(texts, "RETRIEVAL_QUERY")

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> GeminiEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings.gemini
        return cls(api_key=cfg.api_key.get_secret_value(), model=cfg.model)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_with_task(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        if not texts:
            return []
        results: list[DenseVector] = []
        for i in range(0, len(texts), _MAX_BATCH):
            results.extend(self._embed_batch(texts[i : i + _MAX_BATCH], task_type))
        return results

    def _embed_batch(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        try:
            return self._call_with_retry(texts, task_type)
        except Exception as exc:
            raise EmbeddingError(f"Gemini embed failed for {len(texts)} texts", cause=exc) from exc

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        return self._call_api(texts, task_type)

    def _call_api(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise EmbeddingError(
                "google-generativeai package is not installed. Run: uv sync --extra api-embeddings"
            ) from exc

        # Configure with this instance's key immediately before the call so that
        # multiple providers with different keys never interfere with each other.
        genai.configure(api_key=self.api_key)

        # embed_content with a list returns {"embedding": list[list[float]]};
        # with a single string it returns {"embedding": list[float]}. Guard against
        # both shapes so a single-item batch doesn't silently corrupt output.
        result = genai.embed_content(
            model=f"models/{self.model}",
            content=texts,
            task_type=task_type,
        )
        embedding = result["embedding"]
        if embedding and not isinstance(embedding[0], (list, tuple)):
            return [list(embedding)]
        return [list(v) for v in embedding]
