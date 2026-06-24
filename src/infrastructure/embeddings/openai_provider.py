"""OpenAI embedding provider (dense only, API-based)."""

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

_MAX_BATCH = 2048
_SUPPORTS_DIM_TRUNCATION = {"text-embedding-3-large", "text-embedding-3-small"}


class OpenAIEmbeddingProvider(EmbeddingRepository):
    """OpenAI text-embedding API provider.

    Dense only — sparse vectors fall back to BM25.
    Supports dimension truncation for the text-embedding-3 model family.
    Retries automatically on HTTP 429 (rate limit) with exponential backoff.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        # Dimension truncation only works on text-embedding-3-* models
        self.dimensions = dimensions if model in _SUPPORTS_DIM_TRUNCATION else None
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
    def from_settings(cls) -> OpenAIEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings.openai
        return cls(api_key=cfg.api_key, model=cfg.model, dimensions=cfg.dimensions)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> list[DenseVector]:
        for attempt in range(5):
            try:
                return self._call_api(texts)
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < 4:
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        "OpenAI rate limit on attempt %d, retrying in %ds", attempt + 1, wait
                    )
                    time.sleep(wait)
                    continue
                raise EmbeddingError(
                    f"OpenAI embed failed for {len(texts)} texts after {attempt + 1} attempt(s)",
                    cause=exc,
                ) from exc
        raise EmbeddingError(f"OpenAI embed failed after 5 retries for {len(texts)} texts")

    def _call_api(self, texts: list[str]) -> list[DenseVector]:
        client = self._get_client()
        kwargs: dict[str, object] = {"input": texts, "model": self.model}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = client.embeddings.create(**kwargs)  # type: ignore[arg-type]
        return [list(item.embedding) for item in sorted(response.data, key=lambda x: x.index)]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingError(
                    "openai package is not installed. Run: uv sync --extra api-embeddings"
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))
