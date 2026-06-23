from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk


class RerankerRepository(ABC):
    """Contract for re-scoring and re-ordering retrieved chunks."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return at most *top_k* chunks, sorted by relevance to a *query* (the best first).

        The input *chunks* may already carry retrieval scores; implementations
        replace that ordering with cross-encoder scores.
        """
