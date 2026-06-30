"""T-133 — Step-back query transformation tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.entities.query import Query
from src.rag.retrieval.query_expansion import QueryExpander
from src.rag.retrieval.step_back import StepBackGenerator, generate_step_back

# ── helpers ────────────────────────────────────────────────────────────────────


def _query(text: str = "Could IAM roles in EKS perform lawful arrests?") -> Query:
    return Query(text=text)


def _llm(response: str = "What are AWS IAM roles and how do they work?") -> MagicMock:
    mock = MagicMock()
    mock.generate.return_value = response
    return mock


def _generator(
    response: str = "What are AWS IAM roles and how do they work?",
    enabled: bool = True,
) -> StepBackGenerator:
    return StepBackGenerator(llm=_llm(response), enabled=enabled)


# ── generate_step_back ─────────────────────────────────────────────────────────


class TestGenerateStepBack:
    def test_returns_llm_response(self):
        llm = _llm("What is Kubernetes access control?")
        assert generate_step_back(_query().text, llm) == "What is Kubernetes access control?"

    def test_strips_whitespace(self):
        llm = _llm("  broader question  \n")
        assert generate_step_back(_query().text, llm) == "broader question"


# ── StepBackGenerator ──────────────────────────────────────────────────────────


class TestStepBackGenerator:
    def test_disabled_returns_empty(self):
        gen = _generator(enabled=False)
        assert gen.generate(_query().text) == ""

    def test_empty_query_returns_empty(self):
        gen = _generator()
        assert gen.generate("   ") == ""

    def test_enabled_returns_response(self):
        gen = _generator("What is cloud identity management?")
        assert gen.generate(_query().text) == "What is cloud identity management?"

    def test_llm_failure_returns_empty(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        gen = StepBackGenerator(llm=llm, enabled=True)
        assert gen.generate(_query().text) == ""

    def test_empty_llm_response_returns_empty(self):
        gen = _generator(response="")
        assert gen.generate(_query().text) == ""


class TestStepBackGeneratorCache:
    def test_same_query_calls_llm_once(self):
        llm = _llm()
        gen = StepBackGenerator(llm=llm, enabled=True)
        text = _query().text
        gen.generate(text)
        gen.generate(text)
        llm.generate.assert_called_once()

    def test_clear_cache_forces_new_llm_call(self):
        llm = _llm()
        gen = StepBackGenerator(llm=llm, enabled=True)
        text = _query().text
        gen.generate(text)
        gen.clear_cache()
        gen.generate(text)
        assert llm.generate.call_count == 2

    def test_failure_not_cached_retries_on_next_call(self):
        llm = MagicMock()
        llm.generate.side_effect = [RuntimeError("LLM down"), "Recovered background question"]
        gen = StepBackGenerator(llm=llm, enabled=True)
        text = _query().text
        assert gen.generate(text) == ""
        assert gen.generate(text) == "Recovered background question"
        assert llm.generate.call_count == 2

    def test_empty_response_not_cached_retries_on_next_call(self):
        llm = MagicMock()
        llm.generate.side_effect = ["", "Recovered background question"]
        gen = StepBackGenerator(llm=llm, enabled=True)
        text = _query().text
        assert gen.generate(text) == ""
        assert gen.generate(text) == "Recovered background question"
        assert llm.generate.call_count == 2


class TestStepBackFromSettings:
    def test_returns_generator(self):
        gen = StepBackGenerator.from_settings(_llm())
        assert isinstance(gen, StepBackGenerator)


# ── QueryExpander integration ──────────────────────────────────────────────────


class TestQueryExpanderStepBack:
    def test_step_back_stored_in_metadata(self):
        step_back = _generator("What is AWS identity and access management?")
        exp = QueryExpander(llm=_llm("v1\nv2"), n_variants=2, enabled=True, step_back=step_back)
        result = exp.expand(_query())
        assert result.metadata["step_back"] == "What is AWS identity and access management?"

    def test_step_back_runs_when_expansion_disabled(self):
        step_back = _generator("Background question")
        exp = QueryExpander(llm=_llm(), n_variants=3, enabled=False, step_back=step_back)
        result = exp.expand(_query())
        assert result.expanded_texts == []
        assert result.metadata["step_back"] == "Background question"

    def test_step_back_failure_does_not_block_expansion(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            "Variant one\nVariant two",
            RuntimeError("step-back down"),
        ]
        step_back = StepBackGenerator(llm=llm, enabled=True)
        exp = QueryExpander(llm=llm, n_variants=2, enabled=True, step_back=step_back)
        result = exp.expand(_query())
        assert result.expanded_texts == ["Variant one", "Variant two"]
        assert "step_back" not in result.metadata

    def test_no_step_back_generator_skips_metadata(self):
        exp = QueryExpander(llm=_llm("v1"), n_variants=1, enabled=True)
        result = exp.expand(_query())
        assert "step_back" not in result.metadata

    def test_step_back_failure_retries_on_next_expand(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            RuntimeError("step-back down"),
            "Recovered background question",
        ]
        step_back = StepBackGenerator(llm=llm, enabled=True)
        exp = QueryExpander(llm=_llm(), n_variants=0, enabled=False, step_back=step_back)
        q = _query()
        assert "step_back" not in exp.expand(q).metadata
        assert exp.expand(q).metadata["step_back"] == "Recovered background question"

    def test_clear_cache_clears_step_back(self):
        llm = _llm("Background question")
        step_back = StepBackGenerator(llm=llm, enabled=True)
        exp = QueryExpander(llm=_llm(), n_variants=0, enabled=False, step_back=step_back)
        q = _query()
        exp.expand(q)
        exp.clear_cache()
        exp.expand(q)
        assert llm.generate.call_count == 2


class TestQueryVariantsWithStepBack:
    def test_step_back_included_in_variants(self):
        from src.domain.services.retrieval_service import RetrievalService

        query = Query(
            text="specific question",
            expanded_texts=["variant a"],
            metadata={"step_back": "broader background question"},
        )
        variants = RetrievalService._query_variants(query)
        assert variants == ["specific question", "variant a", "broader background question"]

    def test_duplicate_step_back_deduplicated(self):
        from src.domain.services.retrieval_service import RetrievalService

        query = Query(text="same", metadata={"step_back": "same"})
        variants = RetrievalService._query_variants(query)
        assert variants == ["same"]
