from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import SearchResult


class RerankerRepository(ABC):
    """Contract for re-scoring and re-ordering retrieved chunks."""

    @abstractmethod
    def score(self, query: str, chunks: list[Chunk]) -> list[SearchResult]:
        """Return a cross-encoder relevance score for each *chunk* (unsorted)."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return at most *top_k* chunks, sorted by relevance to a *query* (the best first).

        The input *chunks* may already carry retrieval scores; implementations
        replace that ordering with cross-encoder scores.
        """
