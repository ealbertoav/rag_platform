"""Voyage AI embedding provider (dense only, API-based)."""

from __future__ import annotations

import logging
from typing import Any

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

_MAX_BATCH = 128  # Voyage API limit


def _is_rate_limit(exc: BaseException) -> bool:
    try:
        from voyageai.error import RateLimitError
        return isinstance(exc, RateLimitError)
    except ImportError:
        msg = str(exc).lower()
        return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))


class VoyageEmbeddingProvider(EmbeddingRepository):
    """Voyage AI embedding API provider.

    Dense only — sparse vectors fall back to BM25.
    voyage-code-2 is recommended for technical documentation and code.
    Retries on HTTP 429 with exponential backoff via tenacity.
    """

    def __init__(self, api_key: str, model: str = "voyage-large-2") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Embed texts for document storage (input_type=document)."""
        return self._embed_with_type(texts, "document")

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{} for _ in texts]

    def embed_query(self, texts: list[str]) -> list[DenseVector]:
        """Embed query texts (input_type=query). Use during retrieval."""
        return self._embed_with_type(texts, "query")

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> VoyageEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings.voyage
        return cls(api_key=cfg.api_key.get_secret_value(), model=cfg.model)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_with_type(self, texts: list[str], input_type: str) -> list[DenseVector]:
        if not texts:
            return []
        results: list[DenseVector] = []
        for i in range(0, len(texts), _MAX_BATCH):
            results.extend(self._embed_batch(texts[i : i + _MAX_BATCH], input_type))
        return results

    def _embed_batch(self, texts: list[str], input_type: str) -> list[DenseVector]:
        try:
            return self._call_with_retry(texts, input_type)
        except Exception as exc:
            raise EmbeddingError(f"Voyage embed failed for {len(texts)} texts", cause=exc) from exc

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry(self, texts: list[str], input_type: str) -> list[DenseVector]:
        return self._call_api(texts, input_type)

    def _call_api(self, texts: list[str], input_type: str) -> list[DenseVector]:
        client = self._get_client()
        result = client.embed(texts, model=self.model, input_type=input_type)
        return [list(v) for v in result.embeddings]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:
                raise EmbeddingError(
                    "voyageai package is not installed. Run: uv sync --extra api-embeddings"
                ) from exc
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client
