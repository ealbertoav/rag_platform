"""Unit tests for src/infrastructure/search/web_search.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.repositories.web_search_repository import WebSearchResult
from src.infrastructure.search.web_search import (
    DuckDuckGoWebSearchProvider,
    TavilyWebSearchProvider,
    format_web_results,
    get_web_search_provider,
    parse_duckduckgo_lite,
)
from tests.unit.web_search_helpers import patch_web_search_httpx_post


class TestDuckDuckGoWebSearchProvider:
    @pytest.mark.asyncio
    async def test_empty_query_returns_no_results(self):
        provider = DuckDuckGoWebSearchProvider()
        assert await provider.search("   ") == []

    @pytest.mark.asyncio
    async def test_search_posts_query_and_parses_response(self):
        html = """
        <html><body><table>
        <tr>
          <td><a class="result-link" href="https://example.com">Example</a></td>
          <td class="result-snippet">Snippet text.</td>
        </tr>
        </table></body></html>
        """
        with patch_web_search_httpx_post(text=html) as mock_client:
            results = await DuckDuckGoWebSearchProvider().search("example query", max_results=3)

        mock_client.post.assert_awaited_once()
        assert len(results) == 1
        assert results[0].title == "Example"


class TestTavilyWebSearchProvider:
    @pytest.mark.asyncio
    async def test_empty_query_returns_no_results(self):
        provider = TavilyWebSearchProvider(api_key="test-key")
        assert await provider.search("\t") == []

    @pytest.mark.asyncio
    async def test_search_maps_api_response(self):
        api_payload = {
            "results": [
                {
                    "title": "Tavily Hit",
                    "url": "https://example.com",
                    "content": "Primary content field.",
                },
                {
                    "title": "",
                    "url": "",
                    "snippet": "Fallback snippet field.",
                },
                {
                    "title": "",
                    "url": "",
                    "content": "",
                    "snippet": "",
                },
            ]
        }
        with patch_web_search_httpx_post(json_payload=api_payload) as mock_client:
            results = await TavilyWebSearchProvider(api_key="secret").search(
                "example query",
                max_results=2,
            )

        call_kwargs = mock_client.post.await_args.kwargs
        assert call_kwargs["json"]["api_key"] == "secret"
        assert call_kwargs["json"]["query"] == "example query"
        assert call_kwargs["json"]["max_results"] == 2
        assert len(results) == 2
        assert results[0].title == "Tavily Hit"
        assert results[0].snippet == "Primary content field."
        assert results[1].snippet == "Fallback snippet field."


class TestParseDuckduckgoLite:
    def test_respects_max_results_limit(self):
        html = """
        <html><body><table>
        <tr><td><a class="result-link" href="https://a.com">A</a></td></tr>
        <tr><td><a class="result-link" href="https://b.com">B</a></td></tr>
        <tr><td><a class="result-link" href="https://c.com">C</a></td></tr>
        </table></body></html>
        """
        results = parse_duckduckgo_lite(html, max_results=2)
        assert len(results) == 2
        assert [result.title for result in results] == ["A", "B"]

    def test_snippet_lookup_stops_at_sponsored_row(self):
        html = """
        <html><body><table>
        <tr>
          <td><a class="result-link" href="https://example.com">Example</a></td>
        </tr>
        <tr class="result-sponsored">
          <td class="result-snippet">Paid snippet should not attach.</td>
        </tr>
        </table></body></html>
        """
        results = parse_duckduckgo_lite(html, max_results=3)
        assert len(results) == 1
        assert results[0].title == "Example"
        assert results[0].snippet == ""


class TestFormatWebResults:
    def test_empty_results_returns_empty_string(self):
        assert format_web_results([]) == ""

    def test_result_without_url_omits_url_line(self):
        text = format_web_results(
            [WebSearchResult(title="Title only", url="", snippet="Body text.")]
        )
        assert text == "[1] Title only\nBody text."
        assert "URL:" not in text


class TestGetWebSearchProvider:
    def test_uses_default_settings_when_none(self):
        with patch("src.infrastructure.search.web_search.default_settings") as default_settings:
            default_settings.web_search.provider = "none"
            provider = get_web_search_provider()
        from src.infrastructure.search.web_search import NullWebSearchProvider

        assert isinstance(provider, NullWebSearchProvider)

    def test_unknown_provider_raises(self):
        settings = MagicMock()
        settings.web_search.provider = "unknown-provider"
        with pytest.raises(ValueError, match="Unknown web search provider"):
            get_web_search_provider(settings)
