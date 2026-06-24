"""Cohere embedding provider (dense only, API-based)."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

logger = logging.getLogger(__name__)

_MAX_BATCH = 96  # Cohere recommends ≤96 texts per request
CohereInputType = Literal["search_document", "search_query"]


class CohereEmbeddingProvider(EmbeddingRepository):
    """Cohere embed API provider.

    Dense only — sparse vectors fall back to BM25.
    Uses input_type="search_document" for ingestion and "search_query" for
    query embedding; override embed() input_type via embed_query() if needed.
    Retries on HTTP 429 with exponential backoff.
    """

    def __init__(self, api_key: str, model: str = "embed-english-v3.0") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any | None = None

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Embed texts for document storage (input_type=search_document)."""
        return self._embed_with_type(texts, "search_document")

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{} for _ in texts]

    def embed_query(self, texts: list[str]) -> list[DenseVector]:
        """Embed query texts (input_type=search_query). Use during retrieval."""
        return self._embed_with_type(texts, "search_query")

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> CohereEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings.cohere
        return cls(api_key=cfg.api_key, model=cfg.model)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_with_type(self, texts: list[str], input_type: CohereInputType) -> list[DenseVector]:
        if not texts:
            return []
        results: list[DenseVector] = []
        for i in range(0, len(texts), _MAX_BATCH):
            results.extend(self._embed_batch(texts[i : i + _MAX_BATCH], input_type))
        return results

    def _embed_batch(self, texts: list[str], input_type: CohereInputType) -> list[DenseVector]:
        for attempt in range(5):
            try:
                return self._call_api(texts, input_type)
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < 4:
                    wait = min(2**attempt * 2, 60)
                    logger.warning(
                        "Cohere rate limit on attempt %d, retrying in %ds", attempt + 1, wait
                    )
                    time.sleep(wait)
                    continue
                raise EmbeddingError(
                    f"Cohere embed failed for {len(texts)} texts after {attempt + 1} attempt(s)",
                    cause=exc,
                ) from exc
        raise EmbeddingError(f"Cohere embed failed after 5 retries for {len(texts)} texts")

    def _call_api(self, texts: list[str], input_type: CohereInputType) -> list[DenseVector]:
        client = self._get_client()
        response = client.embed(texts=texts, model=self.model, input_type=input_type)
        return [list(v) for v in response.embeddings]  # type: ignore[union-attr]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import cohere
            except ImportError as exc:
                raise EmbeddingError(
                    "cohere package is not installed. Run: uv sync --extra api-embeddings"
                ) from exc
            self._client = cohere.Client(api_key=self.api_key)
        return self._client


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))
