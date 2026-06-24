"""Voyage AI embedding provider (dense only, API-based)."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

logger = logging.getLogger(__name__)

_MAX_BATCH = 128  # Voyage API limit


class VoyageEmbeddingProvider(EmbeddingRepository):
    """Voyage AI embedding API provider.

    Dense only — sparse vectors fall back to BM25.
    voyage-code-2 is recommended for technical documentation and code.
    Retries on HTTP 429 with exponential backoff.
    """

    def __init__(self, api_key: str, model: str = "voyage-large-2") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[DenseVector]:
        if not texts:
            return []
        results: list[DenseVector] = []
        for i in range(0, len(texts), _MAX_BATCH):
            results.extend(self._embed_batch(texts[i : i + _MAX_BATCH]))
        return results

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{} for _ in texts]

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> VoyageEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings.voyage
        return cls(api_key=cfg.api_key, model=cfg.model)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> list[DenseVector]:
        for attempt in range(5):
            try:
                return self._call_api(texts)
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < 4:
                    wait = min(2**attempt * 2, 60)
                    logger.warning(
                        "Voyage rate limit on attempt %d, retrying in %ds", attempt + 1, wait
                    )
                    time.sleep(wait)
                    continue
                raise EmbeddingError(
                    f"Voyage embed failed for {len(texts)} texts after {attempt + 1} attempt(s)",
                    cause=exc,
                ) from exc
        raise EmbeddingError(f"Voyage embed failed after 5 retries for {len(texts)} texts")

    def _call_api(self, texts: list[str]) -> list[DenseVector]:
        client = self._get_client()
        result = client.embed(texts, model=self.model, input_type="document")
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


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))
