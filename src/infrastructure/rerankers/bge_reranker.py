from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.exceptions import RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker as _FlagReranker

logger = logging.getLogger(__name__)


class BGERerankerProvider(RerankerRepository):
    """Cross-encoder reranker backed by BGE-Reranker-v2-M3 via FlagEmbedding.

    The model is loaded lazily on the first call.  Input pairs are processed in
    "batch_size" chunks to avoid OOM on long candidate lists.
    """

    def __init__(
        self,
        model_path: str = "models/rerankers/bge-reranker-v2-m3",
        device: str = "mps",
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: _FlagReranker | None = None

    # ── RerankerRepository interface ───────────────────────────────────────────

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return up to *top_k* chunks sorted by cross-encoder relevance score."""
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores = self._score_pairs(pairs)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        logger.debug("Reranked %d chunks → keeping top %d", len(chunks), top_k)
        return [c for c, _ in ranked[:top_k]]

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> BGERerankerProvider:
        from src.core.settings import settings

        cfg = settings.reranker
        return cls(
            model_path=cfg.model_path,
            device=cfg.device if hasattr(cfg, "device") else "mps",
            batch_size=cfg.batch_size,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_model(self) -> _FlagReranker:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker  # lazy import

            use_fp16 = self.device in ("cuda", "mps")
            model = FlagReranker(self.model_path, use_fp16=use_fp16, device=self.device)
            self._model = model
            logger.info("BGE-Reranker loaded from %s on %s", self.model_path, self.device)
            return model
        except (ImportError, OSError, ValueError) as exc:
            raise RetrievalError(
                f"Cannot load BGE-Reranker from {self.model_path!r}", cause=exc
            ) from exc

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score all pairs in batches, returning one float per pair."""
        model = self._get_model()
        all_scores: list[float] = []
        try:
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                raw: Any = model.compute_score(batch, normalize=self.normalize)
                # compute_score returns float for a single pair, list for multiple.
                if isinstance(raw, float):
                    all_scores.append(raw)
                else:
                    all_scores.extend(float(s) for s in raw)
        except Exception as exc:
            raise RetrievalError(
                f"BGE-Reranker scoring failed for {len(pairs)} pairs", cause=exc
            ) from exc
        return all_scores
