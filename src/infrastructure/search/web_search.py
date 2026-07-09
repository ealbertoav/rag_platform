from __future__ import annotations

import logging
from typing import cast, override

import httpx
from bs4 import BeautifulSoup, Tag

from src.core.settings import Settings
from src.core.settings import settings as default_settings
from src.domain.repositories.web_search_repository import WebSearchRepository, WebSearchResult

logger = logging.getLogger(__name__)

_DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"
_TAVILY_URL = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT = 15.0


class NullWebSearchProvider(WebSearchRepository):
    """No-op provider — returns empty results (CRAG disabled or provider=none)."""

    @override
    async def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        return []


class DuckDuckGoWebSearchProvider(WebSearchRepository):
    """DuckDuckGo Lite HTML search — no API key required."""

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout: float = timeout

    @override
    async def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        normalized = query.strip()
        if not normalized:
            return []

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            response = await client.post(
                _DUCKDUCKGO_LITE_URL,
                data={"q": normalized},
                headers={"User-Agent": "rag-platform/0.1"},
            )
            _ = response.raise_for_status()

        return parse_duckduckgo_lite(response.text, max_results=max_results)


class TavilyWebSearchProvider(WebSearchRepository):
    """Tavily REST search — requires an API key."""

    def __init__(self, api_key: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._api_key: str = api_key
        self._timeout: float = timeout

    @override
    async def search(self, query: str, *, max_results: int = 5) -> list[WebSearchResult]:
        normalized = query.strip()
        if not normalized:
            return []

        payload = {
            "api_key": self._api_key,
            "query": normalized,
            "max_results": max_results,
            "search_depth": "basic",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_TAVILY_URL, json=payload)
            _ = response.raise_for_status()
            data = response.json()

        results: list[WebSearchResult] = []
        for item in data.get("results", [])[:max_results]:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("content", item.get("snippet", ""))).strip()
            if title or snippet:
                results.append(WebSearchResult(title=title, url=url, snippet=snippet))
        return results


def _is_sponsored_row(row: Tag) -> bool:
    return "result-sponsored" in (row.get("class") or [])


def _find_following_snippet_row(rows: list[Tag], start_index: int) -> Tag | None:
    """DuckDuckGo Lite puts title/link and snippet in consecutive table rows."""
    for row in rows[start_index + 1 :]:
        if _is_sponsored_row(row):
            return None
        if row.select_one("a.result-link"):
            return None
        snippet_cell = row.select_one("td.result-snippet")
        if snippet_cell is not None:
            return snippet_cell
    return None


def parse_duckduckgo_lite(html: str, *, max_results: int) -> list[WebSearchResult]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[WebSearchResult] = []
    rows = soup.select("tr")

    for index, row in enumerate(rows):
        if len(results) >= max_results:
            break
        if _is_sponsored_row(row):
            continue

        link = row.select_one("a.result-link")
        if link is None:
            continue

        snippet_cell = row.select_one("td.result-snippet")
        if snippet_cell is None:
            snippet_cell = _find_following_snippet_row(rows, index)

        title = link.get_text(strip=True)
        url = str(link.get("href", "")).strip()
        snippet = snippet_cell.get_text(strip=True) if snippet_cell is not None else ""
        if title or snippet:
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))

    return results


def get_web_search_provider(app_settings: Settings | None = None) -> WebSearchRepository:
    """Return the configured web search provider."""
    if app_settings is None:
        app_settings = default_settings
    cfg = app_settings.web_search
    provider = cast(str, cfg.provider)
    match provider:
        case "none":
            return NullWebSearchProvider()
        case "duckduckgo":
            return DuckDuckGoWebSearchProvider()
        case "tavily":
            api_key = cfg.tavily.api_key.get_secret_value()
            if not api_key:
                from src.core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "Provider 'tavily' requires an API key. "
                    + "Set WEB_SEARCH__TAVILY__API_KEY in your environment or .env file."
                )
            return TavilyWebSearchProvider(api_key=api_key)
        case _:
            raise ValueError(f"Unknown web search provider: {provider!r}")


def format_web_results(results: list[WebSearchResult]) -> str:
    """Format web hits as plain text for LLM knowledge refinement."""
    if not results:
        return ""
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        header = result.title or result.url or f"Result {index}"
        body = result.snippet.strip()
        if result.url:
            lines.append(f"[{index}] {header}\nURL: {result.url}\n{body}")
        else:
            lines.append(f"[{index}] {header}\n{body}")
    return "\n\n".join(lines)
