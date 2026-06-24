"""Google Gemini embedding provider (dense only, API-based)."""

from __future__ import annotations

import logging
import time
from typing import Literal

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

logger = logging.getLogger(__name__)

_MAX_BATCH = 100  # Gemini batch limit
GeminiTaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


class GeminiEmbeddingProvider(EmbeddingRepository):
    """Google Gemini text-embedding-004 provider.

    Dense only (768-dim) — sparse vectors fall back to BM25.
    Uses task_type="RETRIEVAL_DOCUMENT" for ingestion and
    "RETRIEVAL_QUERY" for query embedding.
    Retries on quota exhaustion errors.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        self.api_key = api_key
        self.model = model
        self._configured = False

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
        return cls(api_key=cfg.api_key, model=cfg.model)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_with_task(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        if not texts:
            return []
        self._configure()
        results: list[DenseVector] = []
        for i in range(0, len(texts), _MAX_BATCH):
            results.extend(self._embed_batch(texts[i : i + _MAX_BATCH], task_type))
        return results

    def _embed_batch(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        for attempt in range(5):
            try:
                return self._call_api(texts, task_type)
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < 4:
                    wait = min(2**attempt * 2, 60)
                    logger.warning(
                        "Gemini rate limit on attempt %d, retrying in %ds", attempt + 1, wait
                    )
                    time.sleep(wait)
                    continue
                raise EmbeddingError(
                    f"Gemini embed failed for {len(texts)} texts after {attempt + 1} attempt(s)",
                    cause=exc,
                ) from exc
        raise EmbeddingError(f"Gemini embed failed after 5 retries for {len(texts)} texts")

    def _call_api(self, texts: list[str], task_type: GeminiTaskType) -> list[DenseVector]:
        import google.generativeai as genai

        result = genai.embed_content(
            model=f"models/{self.model}",
            content=texts,
            task_type=task_type,
        )
        return [list(v) for v in result["embedding"]]

    def _configure(self) -> None:
        if self._configured:
            return
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise EmbeddingError(
                "google-generativeai package is not installed. Run: uv sync --extra api-embeddings"
            ) from exc
        genai.configure(api_key=self.api_key)
        self._configured = True


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    keywords = ("429", "quota", "resource_exhausted", "rate_limit", "too many requests")
    return any(kw in msg for kw in keywords)
