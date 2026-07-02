from __future__ import annotations

from src.domain.repositories.web_search_repository import WebSearchRepository
from src.infrastructure.search.web_search import get_web_search_provider

__all__ = ["WebSearchRepository", "get_web_search_provider"]
