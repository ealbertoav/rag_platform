"""Unit tests for src/evals/multimodal_regression_gate.py (T-282)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evals.multimodal_regression_gate import check_multimodal_regression_gate, main
from src.evals.regression_gate import GateStatus


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _table_row(i: int = 1, relevant: list[str] | None = None) -> dict[str, object]:
    return {
        "question": f"What does table {i} show?",
        "answer": f"Answer {i}.",
        "relevant_chunks": relevant if relevant is not None else [f"table_c{i:03d}"],
        "modality": "table",
    }


def _figure_row(i: int = 1, relevant: list[str] | None = None) -> dict[str, object]:
    return {
        "question": f"What does figure {i} show?",
        "answer": f"Answer {i}.",
        "relevant_chunks": relevant if relevant is not None else [f"figure_c{i:03d}"],
        "modality": "figure",
    }


def _standard_baseline() -> dict[str, object]:
    return {
        "min_table_samples": 1,
        "min_figure_samples": 1,
        "min_table_recall_at_5": 0.5,
        "min_figure_recall_at_5": 0.5,
    }


def _write_and_run_gate(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    baseline: dict[str, object] | None = None,
):
    dataset_path = tmp_path / "multimodal_qa_dataset.jsonl"
    baseline_path = tmp_path / "multimodal_baseline.json"
    _write_jsonl(dataset_path, rows if rows is not None else [_table_row(1), _figure_row(1)])
    _write_json(baseline_path, baseline if baseline is not None else _standard_baseline())
    return check_multimodal_regression_gate(
        dataset_path=dataset_path,
        baseline_path=baseline_path,
    )


class TestCheckMultimodalRegressionGate:
    def test_skips_when_dataset_file_missing(self, tmp_path: Path):
        result = check_multimodal_regression_gate(
            dataset_path=tmp_path / "missing.jsonl",
            baseline_path=tmp_path / "baseline.json",
        )
        assert result.status == GateStatus.SKIPPED
        assert "skipping" in result.message.lower()

    def test_skips_when_dataset_empty(self, tmp_path: Path):
        dataset_path = tmp_path / "multimodal_qa_dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("", encoding="utf-8")
        result = check_multimodal_regression_gate(dataset_path=dataset_path)
        assert result.status == GateStatus.SKIPPED

    def test_skips_when_no_table_or_figure_samples(self, tmp_path: Path):
        rows = [
            {
                "question": "Untagged?",
                "answer": "A.",
                "relevant_chunks": ["c1"],
                "modality": "text",
            }
        ]
        result = _write_and_run_gate(tmp_path, rows=rows)
        assert result.status == GateStatus.SKIPPED

    def test_passes_with_table_and_figure_samples(self, tmp_path: Path):
        result = _write_and_run_gate(tmp_path)
        assert result.status == GateStatus.PASSED
        assert "PASSED" in result.message

    def test_passes_with_only_table_samples(self, tmp_path: Path):
        result = _write_and_run_gate(tmp_path, rows=[_table_row(1), _table_row(2)])
        assert result.status == GateStatus.PASSED

    def test_passes_with_only_figure_samples(self, tmp_path: Path):
        result = _write_and_run_gate(tmp_path, rows=[_figure_row(1), _figure_row(2)])
        assert result.status == GateStatus.PASSED

    def test_fails_when_below_min_table_samples(self, tmp_path: Path):
        result = _write_and_run_gate(
            tmp_path,
            rows=[_table_row(1), _figure_row(1)],
            baseline={
                "min_table_samples": 2,
                "min_figure_samples": 1,
                "min_table_recall_at_5": 0.5,
                "min_figure_recall_at_5": 0.5,
            },
        )
        assert result.status == GateStatus.FAILED
        assert "table" in result.message.lower()
        assert "need >=" in result.message

    def test_fails_when_row_has_no_relevant_chunks(self, tmp_path: Path):
        result = _write_and_run_gate(tmp_path, rows=[_table_row(1, relevant=[]), _figure_row(1)])
        assert result.status == GateStatus.FAILED
        assert "no relevant_chunks" in result.message

    def test_fails_when_oracle_recall_below_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "src.evals.multimodal_regression_gate.oracle_recall_at_k",
            lambda *_args, **_kwargs: 0.0,
        )
        result = _write_and_run_gate(
            tmp_path,
            baseline={
                "min_table_samples": 1,
                "min_figure_samples": 1,
                "min_table_recall_at_5": 0.99,
                "min_figure_recall_at_5": 0.5,
            },
        )
        assert result.status == GateStatus.FAILED
        assert "oracle Recall@5" in result.message

    def test_uses_defaults_when_baseline_missing(self, tmp_path: Path):
        dataset_path = tmp_path / "multimodal_qa_dataset.jsonl"
        _write_jsonl(dataset_path, [_table_row(1), _figure_row(1)])
        result = check_multimodal_regression_gate(
            dataset_path=dataset_path,
            baseline_path=tmp_path / "missing_baseline.json",
        )
        assert result.status == GateStatus.PASSED

    def test_only_checks_modalities_with_samples(self, tmp_path: Path):
        """A missing figure baseline threshold must not fail when there are no figure rows."""
        result = _write_and_run_gate(
            tmp_path,
            rows=[_table_row(1)],
            baseline={
                "min_table_samples": 1,
                "min_figure_samples": 1,
                "min_table_recall_at_5": 0.5,
                "min_figure_recall_at_5": 1.1,  # would fail if figure were checked
            },
        )
        assert result.status == GateStatus.PASSED


class TestMultimodalRegressionGateMain:
    def test_main_prints_pass_message(self, capsys: pytest.CaptureFixture[str]):
        with patch(
            "src.evals.multimodal_regression_gate.check_multimodal_regression_gate",
            return_value=type("R", (), {"status": GateStatus.PASSED, "message": "ok"})(),
        ):
            main()
        assert "ok" in capsys.readouterr().out

    def test_main_exits_one_on_failure(self, capsys: pytest.CaptureFixture[str]):
        with (
            patch(
                "src.evals.multimodal_regression_gate.check_multimodal_regression_gate",
                return_value=type("R", (), {"status": GateStatus.FAILED, "message": "bad"})(),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
        assert "bad" in capsys.readouterr().out

    def test_main_prints_skip_message(self, capsys: pytest.CaptureFixture[str]):
        with patch(
            "src.evals.multimodal_regression_gate.check_multimodal_regression_gate",
            return_value=type("R", (), {"status": GateStatus.SKIPPED, "message": "skip"})(),
        ):
            main()
        assert "skip" in capsys.readouterr().out
