"""Guard committed golden datasets stay in sync (T-152)."""

from __future__ import annotations

from src.core.constants import DATASETS_DIR
from src.evals.golden_dataset import (
    MIN_QA_PAIRS,
    count_real_qa_pairs,
    load_qa_dicts,
    retrieval_rows_match_qa,
)
from src.evals.regression_gate import check_regression_gate, load_real_retrieval_rows

_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"
_RETRIEVAL_PATH = DATASETS_DIR / "goldens" / "retrieval_dataset.json"
_BASELINE_PATH = DATASETS_DIR / "goldens" / "retrieval_baseline.json"


class TestCommittedGoldens:
    def test_qa_dataset_meets_minimum_sample_count(self):
        assert count_real_qa_pairs(_QA_PATH) >= MIN_QA_PAIRS

    def test_retrieval_matches_qa_golden(self):
        qa_pairs = load_qa_dicts(_QA_PATH)
        real_rows = load_real_retrieval_rows(_RETRIEVAL_PATH)
        assert retrieval_rows_match_qa(qa_pairs, real_rows)

    def test_regression_gate_passes_on_committed_goldens(self):
        result = check_regression_gate(
            qa_path=_QA_PATH,
            retrieval_path=_RETRIEVAL_PATH,
            baseline_path=_BASELINE_PATH,
        )
        assert result.status.value == "passed", result.message
