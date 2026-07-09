"""Base class for sentence-transformer embedding providers (dense-only)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingRepository, ABC):
    """Base for dense-only embedding providers backed by sentence-transformers.

    Concrete subclasses implement "_encode_kwargs()" to pass model-specific
    arguments to "model.encode()" and override "__init__" to accept their
    own config parameters.
    """

    model_path: str
    device: str
    batch_size: int
    normalize: bool
    _model: SentenceTransformer | None

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Return one dense vector per text."""
        if not texts:
            return []
        model = self._get_model()
        try:
            vecs = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                **self._encode_kwargs(),
            )
            return [v.tolist() for v in vecs]
        except Exception as exc:
            raise EmbeddingError(
                f"{type(self).__name__} encode failed for {len(texts)} texts", cause=exc
            ) from exc

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return empty sparse vectors — sentence-transformer models are dense-only.

        BM25 (maintained separately) still provides sparse recall; only
        Qdrant's native sparse index will be empty.
        """
        if texts:
            logger.debug(
                "%s: sparse vectors not supported; returning empty dicts for %d texts",
                type(self).__name__,
                len(texts),
            )
        return [{} for _ in texts]

    # ── To override ────────────────────────────────────────────────────────────

    @abstractmethod
    def _encode_kwargs(self) -> dict[str, Any]:
        """Return extra keyword arguments passed to "model.encode()"."""

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # lazy import

            model = SentenceTransformer(self.model_path, device=self.device, trust_remote_code=True)
            logger.info(
                "%s loaded from %s on %s", type(self).__name__, self.model_path, self.device
            )
            self._model = model
            return self._model
        except (ImportError, OSError, ValueError) as exc:
            raise EmbeddingError(
                f"Cannot load {type(self).__name__} from {self.model_path!r}", cause=exc
            ) from exc
