"""T-020 — QueryExpander tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.entities.query import Query
from src.rag.retrieval.query_expansion import QueryExpander

# ── helpers ────────────────────────────────────────────────────────────────────


def _query(text: str = "How do IAM roles work in EKS?") -> Query:
    return Query(text=text)


def _llm(response: str = "Variant one\nVariant two\nVariant three") -> MagicMock:
    mock = MagicMock()
    mock.generate.return_value = response
    return mock


def _expander(
    response: str = "Variant one\nVariant two\nVariant three",
    n: int = 3,
    enabled: bool = True,
) -> QueryExpander:
    return QueryExpander(llm=_llm(response), n_variants=n, enabled=enabled)


# ── Response parsing (tested through expand()) ────────────────────────────────


class TestResponseParsing:
    def test_plain_lines_become_variants(self):
        result = _expander("A\nB\nC", n=3).expand(_query())
        assert result.expanded_texts == ["A", "B", "C"]

    def test_numbered_prefixes_stripped(self):
        result = _expander("1. First\n2. Second", n=2).expand(_query())
        assert result.expanded_texts == ["First", "Second"]

    def test_dash_prefixes_stripped(self):
        result = _expander("- foo\n- bar", n=2).expand(_query())
        assert result.expanded_texts == ["foo", "bar"]

    def test_bullet_prefixes_stripped(self):
        result = _expander("• one\n• two", n=2).expand(_query())
        assert result.expanded_texts == ["one", "two"]

    def test_empty_lines_skipped(self):
        result = _expander("\nA\n\nB\n", n=2).expand(_query())
        assert result.expanded_texts == ["A", "B"]

    def test_empty_llm_response_gives_no_variants(self):
        result = _expander("", n=3).expand(_query())
        assert result.expanded_texts == []


# ── QueryExpander.expand ───────────────────────────────────────────────────────


class TestQueryExpander:
    def test_returns_query(self):
        result = _expander().expand(_query())
        assert isinstance(result, Query)

    def test_original_text_preserved(self):
        q = _query("my question")
        result = _expander().expand(q)
        assert result.text == "my question"

    def test_expanded_texts_populated(self):
        result = _expander("Line 1\nLine 2\nLine 3").expand(_query())
        assert result.expanded_texts == ["Line 1", "Line 2", "Line 3"]

    def test_n_variants_respected(self):
        result = _expander("A\nB\nC\nD", n=2).expand(_query())
        assert len(result.expanded_texts) == 2

    def test_disabled_returns_original_unchanged(self):
        exp = _expander(enabled=False)
        q = _query()
        result = exp.expand(q)
        assert result is q
        exp._llm.generate.assert_not_called()  # type: ignore[attr-defined]

    def test_disabled_expanded_texts_empty(self):
        result = _expander(enabled=False).expand(_query())
        assert result.expanded_texts == []

    def test_n_variants_zero_returns_unchanged(self):
        exp = _expander(n=0)
        q = _query()
        assert exp.expand(q) is q

    def test_llm_failure_returns_original(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        exp = QueryExpander(llm=llm, n_variants=3, enabled=True)
        result = exp.expand(_query())
        assert result.expanded_texts == []

    def test_empty_llm_response_returns_original(self):
        result = _expander(response="").expand(_query())
        assert result.expanded_texts == []


# ── Caching ────────────────────────────────────────────────────────────────────


class TestQueryExpanderCache:
    def test_same_query_calls_llm_once(self):
        llm = _llm()
        exp = QueryExpander(llm=llm, n_variants=2)
        q = _query("same question")
        exp.expand(q)
        exp.expand(q)
        llm.generate.assert_called_once()

    def test_different_queries_call_llm_each_time(self):
        llm = _llm()
        exp = QueryExpander(llm=llm, n_variants=2)
        exp.expand(_query("question A"))
        exp.expand(_query("question B"))
        assert llm.generate.call_count == 2

    def test_cached_variants_are_returned(self):
        llm = _llm("First\nSecond")
        exp = QueryExpander(llm=llm, n_variants=2)
        r1 = exp.expand(_query("q"))
        r2 = exp.expand(_query("q"))
        assert r1.expanded_texts == r2.expanded_texts

    def test_clear_cache_forces_new_llm_call(self):
        llm = _llm()
        exp = QueryExpander(llm=llm, n_variants=2)
        exp.expand(_query("q"))
        exp.clear_cache()
        exp.expand(_query("q"))
        assert llm.generate.call_count == 2

    def test_higher_n_variants_regenerates_when_cache_too_small(self):
        llm = _llm("One\nTwo\nThree")
        exp = QueryExpander(llm=llm, n_variants=3)
        q = _query("same question")
        r1 = exp.expand(q, n_variants=1)
        assert r1.expanded_texts == ["One"]
        r2 = exp.expand(q, n_variants=3)
        assert r2.expanded_texts == ["One", "Two", "Three"]
        assert llm.generate.call_count == 2

    def test_lower_n_variants_reuses_cache_without_extra_llm_call(self):
        llm = _llm("One\nTwo\nThree")
        exp = QueryExpander(llm=llm, n_variants=3)
        q = _query("same question")
        exp.expand(q, n_variants=3)
        result = exp.expand(q, n_variants=1)
        assert result.expanded_texts == ["One"]
        assert llm.generate.call_count == 1

    def test_failure_not_cached_retries_on_next_call(self):
        llm = MagicMock()
        llm.generate.side_effect = [RuntimeError("LLM down"), "Recovered variant"]
        exp = QueryExpander(llm=llm, n_variants=1, enabled=True)
        q = _query("same question")
        assert exp.expand(q).expanded_texts == []
        assert exp.expand(q).expanded_texts == ["Recovered variant"]
        assert llm.generate.call_count == 2


# ── from_settings ──────────────────────────────────────────────────────────────


class TestFromSettings:
    def test_returns_query_expander(self):
        llm = _llm()
        exp = QueryExpander.from_settings(llm)
        assert isinstance(exp, QueryExpander)

    def test_uses_settings_n_variants(self):
        from src.core.settings import settings

        llm = _llm("\n".join(f"v{i}" for i in range(10)))
        exp = QueryExpander.from_settings(llm)
        result = exp.expand(_query())
        assert len(result.expanded_texts) <= settings.query_expansion.n_variants
