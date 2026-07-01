from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class WebSearchResult:
    """One web search hit returned by a search provider."""

    title: str
    url: str
    snippet: str


class WebSearchRepository(ABC):
    """Contract for external web search used by Corrective RAG."""

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        """Return ranked web results for a *query*."""
