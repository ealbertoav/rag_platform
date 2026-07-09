from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast, override

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class BGEM3EmbeddingProvider(EmbeddingRepository):
    """BGE-M3 embedding provider via FlagEmbedding.

    Produces dense (1024-dim) and sparse (lexical) vectors in a single forward
    pass.  The underlying model is loaded lazily on the first call, so import
    time stays fast and tests can be collected without downloading weights.
    """

    def __init__(
        self,
        model_path: str = "models/embeddings/bge-m3",
        device: str = "mps",
        batch_size: int = 32,
        normalize: bool = True,
        max_length: int = 8192,
    ) -> None:
        self.model_path: str = model_path
        self.device: str = device
        self.batch_size: int = batch_size
        self.normalize: bool = normalize
        self.max_length: int = max_length
        self._model: BGEM3FlagModel | None = None

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    @override
    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Return one normalized 1024-dim dense vector per text."""
        output = self._call_model(texts, return_dense=True, return_sparse=False)
        return [v.tolist() for v in output["dense_vecs"]]

    @override
    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return one sparse {token_id: weight} dict per text."""
        output = self._call_model(texts, return_dense=False, return_sparse=True)
        return self._normalise_sparse(output["lexical_weights"])

    # ── Combined pass (used by the ingestion pipeline for efficiency) ──────────

    @override
    def embed_both(self, texts: list[str]) -> tuple[list[DenseVector], list[SparseVector]]:
        """Dense and sparse in a single model call — preferred during ingestion."""
        output = self._call_model(texts, return_dense=True, return_sparse=True)
        dense = [v.tolist() for v in output["dense_vecs"]]
        sparse = self._normalise_sparse(output["lexical_weights"])
        return dense, sparse

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> BGEM3EmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings
        return cls(
            model_path=cfg.model_path,
            device=cfg.device,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_model(self) -> BGEM3FlagModel:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel  # lazy import

            use_fp16 = self.device in ("cuda", "mps")
            model = BGEM3FlagModel(
                self.model_path,
                use_fp16=use_fp16,
                device=self.device,
            )
            logger.info("BGE-M3 loaded from %s on %s", self.model_path, self.device)
            self._model = model
            return model
        except (ImportError, OSError, ValueError) as exc:
            raise EmbeddingError(f"Cannot load BGE-M3 from {self.model_path!r}", cause=exc) from exc

    def _call_model(
        self,
        texts: list[str],
        return_dense: bool,
        return_sparse: bool,
    ) -> dict[str, Any]:
        if not texts:
            return {"dense_vecs": [], "lexical_weights": []}
        try:
            return cast(
                dict[str, Any],
                self._get_model().encode(
                    texts,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=False,
                ),
            )
        except Exception as exc:
            raise EmbeddingError(f"BGE-M3 encode failed for {len(texts)} texts", cause=exc) from exc

    @staticmethod
    def _normalise_sparse(lexical_weights: list[dict[Any, Any]]) -> list[SparseVector]:
        """Ensure sparse vector keys are int token IDs and values are floats.

        FlagEmbedding may return string keys in some versions; this normalizes
         them, so downstream code always receives "{int: float}".
        """
        result: list[SparseVector] = []
        for weights in lexical_weights:
            result.append({int(k): float(v) for k, v in weights.items() if float(v) > 0})
        return result
