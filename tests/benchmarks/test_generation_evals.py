"""T-042 benchmark tests — Generation Evals against the QA golden dataset.

Run with:
    uv run pytest tests/benchmarks/test_generation_evals.py -v -s

Requires:
    uv run pip install ragas datasets deepeval (or: uv sync --extra evals)
    Golden QA dataset populated via T-040

Skipped automatically when dependencies or data are missing.
"""
from __future__ import annotations

import json

import pytest

from src.core.constants import DATASETS_DIR

_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"


def _ragas_available() -> bool:
    try:
        import ragas  # noqa: F401

        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def _deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401
        return True
    except ImportError:
        return False


def _load_qa() -> list[dict[str, object]]:
    try:
        data = json.loads(_QA_PATH.read_text())
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and "question" in d]
        return []
    except (OSError, ValueError):
        return []


_QA_DATA = _load_qa()
_HAS_RAGAS = _ragas_available()
_HAS_DEEPEVAL = _deepeval_available()

pytestmark = pytest.mark.skipif(
    len(_QA_DATA) == 0,
    reason="QA golden dataset is empty — populate via T-040 first.",
)


def _str(val: object) -> str:
    return val if isinstance(val, str) else ""


def _str_list(val: object) -> list[str]:
    return [v for v in val if isinstance(v, str)] if isinstance(val, list) else []


@pytest.fixture(scope="module")
def samples():
    from src.domain.entities.evaluation import EvalSample

    return [
        EvalSample(
            question=_str(d.get("question")),
            expected_answer=_str(d.get("answer")),
            generated_answer=_str(d.get("answer")),  # use ground truth as generated
            retrieved_chunks=_str_list(d.get("relevant_chunks")),
        )
        for d in _QA_DATA
    ]


@pytest.mark.skipif(not _HAS_RAGAS, reason="ragas not installed (pip install ragas datasets)")
class TestFaithfulnessBenchmark:
    def test_faithfulness_scores_all_samples(self, samples):
        from src.evals.generation.faithfulness import FaithfulnessMetric

        metric = FaithfulnessMetric(threshold=0.8)
        results = [metric.score(s) for s in samples]
        scores = [r.score for r in results]
        passed = sum(1 for r in results if r.passed)
        print(f"\nFaithfulness: mean={sum(scores)/len(scores):.3f}, passed={passed}/{len(results)}")
        assert len(results) == len(samples)

    def test_relevance_scores_all_samples(self, samples):
        from src.evals.generation.relevance import RelevanceMetric

        metric = RelevanceMetric(threshold=0.75)
        results = [metric.score(s) for s in samples]
        passed = sum(1 for r in results if r.passed)
        print(f"\nRelevance: passed={passed}/{len(results)}")
        assert len(results) == len(samples)


@pytest.mark.skipif(not _HAS_DEEPEVAL, reason="deepeval not installed (pip install deepeval)")
class TestHallucinationBenchmark:
    def test_hallucination_scores_all_samples(self, samples):
        from src.evals.generation.hallucination import HallucinationMetric

        metric = HallucinationMetric(threshold=0.1)
        results = [metric.score(s) for s in samples]
        passed = sum(1 for r in results if r.passed)
        print(f"\nHallucination: passed={passed}/{len(results)}")
        assert len(results) == len(samples)
