"""Unit tests for src/evals/regression_gate.py (T-152)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evals.golden_dataset import MIN_QA_PAIRS
from src.evals.regression_gate import (
    GateStatus,
    _baseline_float,
    _baseline_int,
    check_regression_gate,
    load_real_retrieval_rows,
    load_regression_baseline,
    main,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _real_retrieval_row(i: int = 1) -> dict[str, object]:
    return {
        "id": f"retrieval_{i:03d}",
        "query": f"Question {i}?",
        "relevant_chunk_ids": [f"rag_c{i:03d}"],
    }


def _real_qa_row(i: int = 1) -> dict[str, object]:
    return {
        "question": f"Question {i}?",
        "answer": f"Answer {i}.",
        "relevant_chunks": [f"rag_c{i:03d}"],
    }


class TestLoadRegressionBaseline:
    def test_loads_committed_baseline(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        _write_json(path, {"min_samples": 5, "min_recall_at_5": 0.6})
        assert load_regression_baseline(path) == {"min_samples": 5, "min_recall_at_5": 0.6}

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_regression_baseline(tmp_path / "missing.json") == {}

    def test_invalid_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        path.write_text("not json", encoding="utf-8")
        assert load_regression_baseline(path) == {}

    def test_non_dict_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "baseline.json"
        _write_json(path, ["not", "a", "dict"])
        assert load_regression_baseline(path) == {}


class TestLoadRealRetrievalRows:
    def test_filters_placeholder_rows(self, tmp_path: Path):
        path = tmp_path / "retrieval.json"
        _write_json(
            path,
            [
                _real_retrieval_row(1),
                {"id": "p1", "query": "?", "relevant_chunk_ids": ["chunk_id_1"]},
            ],
        )
        rows = load_real_retrieval_rows(path)
        assert len(rows) == 1
        assert rows[0]["id"] == "retrieval_001"

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_real_retrieval_rows(tmp_path / "missing.json") == []

    def test_invalid_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "retrieval.json"
        path.write_text("{bad", encoding="utf-8")
        assert load_real_retrieval_rows(path) == []

    def test_non_list_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "retrieval.json"
        _write_json(path, {"not": "a list"})
        assert load_real_retrieval_rows(path) == []


class TestBaselineCoercion:
    def test_baseline_int_accepts_int(self):
        assert _baseline_int({"min_samples": 25}, "min_samples", MIN_QA_PAIRS) == 25

    def test_baseline_int_accepts_float(self):
        assert _baseline_int({"min_samples": 25.0}, "min_samples", MIN_QA_PAIRS) == 25

    def test_baseline_int_accepts_numeric_string(self):
        assert _baseline_int({"min_samples": "25"}, "min_samples", MIN_QA_PAIRS) == 25

    def test_baseline_int_rejects_bool_and_invalid(self):
        assert _baseline_int({"min_samples": True}, "min_samples", MIN_QA_PAIRS) == MIN_QA_PAIRS
        assert _baseline_int({"min_samples": "bad"}, "min_samples", MIN_QA_PAIRS) == MIN_QA_PAIRS
        assert _baseline_int({"min_samples": []}, "min_samples", MIN_QA_PAIRS) == MIN_QA_PAIRS

    def test_baseline_float_accepts_number(self):
        assert _baseline_float({"min_recall_at_5": 0.75}, "min_recall_at_5", 0.5) == 0.75

    def test_baseline_float_accepts_numeric_string(self):
        assert _baseline_float({"min_recall_at_5": "0.75"}, "min_recall_at_5", 0.5) == 0.75

    def test_baseline_float_rejects_bool_and_invalid(self):
        assert _baseline_float({"min_recall_at_5": False}, "min_recall_at_5", 0.5) == 0.5
        assert _baseline_float({"min_recall_at_5": "bad"}, "min_recall_at_5", 0.5) == 0.5
        assert _baseline_float({"min_recall_at_5": {}}, "min_recall_at_5", 0.5) == 0.5


class TestCheckRegressionGate:
    def test_skips_when_retrieval_file_missing(self, tmp_path: Path):
        result = check_regression_gate(
            qa_path=tmp_path / "qa.json",
            retrieval_path=tmp_path / "retrieval.json",
        )
        assert result.status == GateStatus.SKIPPED
        assert "skipping" in result.message.lower()

    def test_skips_when_only_placeholders(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        _write_json(
            retrieval,
            [{"id": "p1", "query": "?", "relevant_chunk_ids": ["chunk_id_1"]}],
        )
        result = check_regression_gate(
            qa_path=tmp_path / "qa.json",
            retrieval_path=retrieval,
        )
        assert result.status == GateStatus.SKIPPED

    def test_fails_when_below_min_samples(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        _write_json(qa, [_real_qa_row(1)])
        _write_json(retrieval, [_real_retrieval_row(1)])
        _write_json(baseline, {"min_samples": MIN_QA_PAIRS})

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.FAILED
        assert "need >=" in result.message

    def test_fails_when_row_has_no_relevant_ids(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        rows = [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)]
        rows[0] = {"id": "retrieval_001", "query": "?", "relevant_chunk_ids": []}
        _write_json(qa, [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(retrieval, rows)
        _write_json(baseline, {"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.5})

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.FAILED
        assert "no relevant_chunk_ids" in result.message

    def test_fails_when_recall_below_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        _write_json(qa, [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(retrieval, [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(baseline, {"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.99})

        monkeypatch.setattr(
            "src.evals.regression_gate.recall_at_k",
            lambda *_args, **_kwargs: 0.0,
        )
        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.FAILED
        assert "Recall@5" in result.message

    def test_passes_with_real_data(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        _write_json(qa, [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(retrieval, [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(baseline, {"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.5})

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.PASSED
        assert "PASSED" in result.message

    def test_coerces_string_baseline_thresholds(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        _write_json(qa, [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(retrieval, [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(
            baseline,
            {"min_samples": str(MIN_QA_PAIRS), "min_recall_at_5": "0.5"},
        )

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.PASSED

    def test_falls_back_on_invalid_baseline_values(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        _write_json(qa, [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(retrieval, [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(
            baseline,
            {"min_samples": "not-a-number", "min_recall_at_5": True},
        )

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.PASSED

    def test_fails_when_relevant_chunk_ids_not_a_list(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        rows = [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)]
        rows[0] = {"id": "retrieval_001", "query": "?", "relevant_chunk_ids": "bad"}
        _write_json(qa, [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)])
        _write_json(retrieval, rows)
        _write_json(baseline, {"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.5})

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.FAILED
        assert "no relevant_chunk_ids" in result.message

    def test_fails_when_retrieval_out_of_sync_with_qa(self, tmp_path: Path):
        qa = tmp_path / "qa.json"
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        qa_rows = [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)]
        _write_json(qa, qa_rows)
        _write_json(
            retrieval,
            [
                {
                    "id": f"retrieval_{i:03d}",
                    "query": f"Mismatched question {i}?",
                    "relevant_chunk_ids": [f"rag_c{i:03d}"],
                }
                for i in range(1, MIN_QA_PAIRS + 1)
            ],
        )
        _write_json(baseline, {"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.5})

        result = check_regression_gate(
            qa_path=qa,
            retrieval_path=retrieval,
            baseline_path=baseline,
        )
        assert result.status == GateStatus.FAILED
        assert "out of sync" in result.message


class TestRegressionGateMain:
    def test_main_prints_pass_message(self, capsys: pytest.CaptureFixture[str]):
        with patch(
            "src.evals.regression_gate.check_regression_gate",
            return_value=type(
                "R",
                (),
                {"status": GateStatus.PASSED, "message": "ok"},
            )(),
        ):
            main()
        assert "ok" in capsys.readouterr().out

    def test_main_exits_one_on_failure(self, capsys: pytest.CaptureFixture[str]):
        with (
            patch(
                "src.evals.regression_gate.check_regression_gate",
                return_value=type(
                    "R",
                    (),
                    {"status": GateStatus.FAILED, "message": "bad"},
                )(),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
        assert "bad" in capsys.readouterr().out

    def test_main_prints_skip_message(self, capsys: pytest.CaptureFixture[str]):
        with patch(
            "src.evals.regression_gate.check_regression_gate",
            return_value=type(
                "R",
                (),
                {"status": GateStatus.SKIPPED, "message": "skip"},
            )(),
        ):
            main()
        assert "skip" in capsys.readouterr().out
