from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from src.core.exceptions import RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as _SentenceTransformersCrossEncoder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"


class QwenRerankerProvider(RerankerRepository):
    """Cross-encoder reranker backed by Qwen3-Reranker via sentence-transformers.

    Qwen3-Reranker ships as a causal-LM checkpoint; sentence-transformers'
    "CrossEncoder" auto-detects the "*ForCausalLM" architecture and appends a
    LogitScore head, so loading it is no different from any other CrossEncoder
    model. The model is loaded lazily on the first call; pairs are scored in
    "batch_size" chunks to avoid OOM on long candidate lists — mirrors
    "BGERerankerProvider".
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        device: str = "mps",
        batch_size: int = 16,
    ) -> None:
        self.model_path: str = model_path
        self.device: str = device
        self.batch_size: int = batch_size
        self._model: _SentenceTransformersCrossEncoder | None = None

    # ── RerankerRepository interface ───────────────────────────────────────────

    @override
    def score(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        """Return cross-encoder relevance scores for each chunk (input order)."""
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._score_pairs(pairs)
        return list(zip(chunks, scores, strict=True))

    @override
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return up to *top_k* chunks sorted by cross-encoder relevance score."""
        ranked = sorted(self.score(query, chunks), key=lambda x: x[1], reverse=True)
        logger.debug("Qwen-Reranker: %d chunks → keeping top %d", len(chunks), top_k)
        return [c for c, _ in ranked[:top_k]]

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> QwenRerankerProvider:
        from src.core.settings import settings

        cfg = settings.reranker
        return cls(
            model_path=cfg.model_path,
            device=getattr(cfg, "device", "mps"),
            batch_size=cfg.batch_size,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_model(self) -> _SentenceTransformersCrossEncoder:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder

            self._model = SentenceTransformersCrossEncoder(self.model_path, device=self.device)
            logger.info("Qwen-Reranker loaded from %s on %s", self.model_path, self.device)
            return self._model
        except (ImportError, OSError, ValueError) as exc:
            raise RetrievalError(
                f"Cannot load Qwen-Reranker from {self.model_path!r}", cause=exc
            ) from exc

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score all pairs in batches, returning one float per pair."""
        model = self._get_model()
        all_scores: list[float] = []
        try:
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                raw = model.predict(batch)  # type: ignore[arg-type]
                all_scores.extend(float(s) for s in raw)
        except Exception as exc:
            raise RetrievalError(
                f"Qwen-Reranker scoring failed for {len(pairs)} pairs", cause=exc
            ) from exc
        return all_scores
