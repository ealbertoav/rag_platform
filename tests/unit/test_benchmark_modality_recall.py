"""Unit tests for scripts/benchmark_modality_recall.py (T-281)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import MODALITY_FIGURE, MODALITY_TABLE


def _pair(question: str, relevant_chunks: list[str], modality: str) -> dict[str, object]:
    return {
        "question": question,
        "answer": "A",
        "relevant_chunks": relevant_chunks,
        "modality": modality,
    }


def _pipeline_returning(chunk_ids: list[str]) -> MagicMock:
    pipeline = MagicMock()
    pipeline.retrieve_sync.return_value = SimpleNamespace(
        chunks=[SimpleNamespace(id=cid) for cid in chunk_ids]
    )
    return pipeline


class TestBenchmarkModalityRecallMain:
    def test_exits_when_dataset_empty(self, monkeypatch: pytest.MonkeyPatch):
        import benchmark_modality_recall

        monkeypatch.setattr(sys, "argv", ["benchmark_modality_recall.py"])
        with (
            patch("benchmark_modality_recall.load_jsonl", return_value=[]),
            pytest.raises(SystemExit) as exc,
        ):
            benchmark_modality_recall.main()
        assert exc.value.code == 1

    def test_exits_when_no_relevant_chunks(self, monkeypatch: pytest.MonkeyPatch):
        import benchmark_modality_recall

        pairs = [_pair("Q1", [], MODALITY_TABLE)]
        monkeypatch.setattr(sys, "argv", ["benchmark_modality_recall.py"])
        with (
            patch("benchmark_modality_recall.load_jsonl", return_value=pairs),
            pytest.raises(SystemExit) as exc,
        ):
            benchmark_modality_recall.main()
        assert exc.value.code == 1

    def test_computes_and_prints_metrics(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        import benchmark_modality_recall

        pairs = [
            _pair("Table question", ["table0"], MODALITY_TABLE),
            _pair("Figure question", ["figure0"], MODALITY_FIGURE),
        ]
        pipeline = _pipeline_returning(["table0", "figure0"])
        monkeypatch.setattr(sys, "argv", ["benchmark_modality_recall.py", "--k", "1"])
        with (
            patch("benchmark_modality_recall.load_jsonl", return_value=pairs),
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                return_value=pipeline,
            ),
        ):
            benchmark_modality_recall.main()

        captured = capsys.readouterr()
        assert "Modality Retrieval Metrics" in captured.out
        assert MODALITY_TABLE in captured.out
        assert MODALITY_FIGURE in captured.out

    def test_skips_pairs_without_question(self, monkeypatch: pytest.MonkeyPatch):
        import benchmark_modality_recall

        pairs = [_pair("", ["table0"], MODALITY_TABLE)]
        pipeline = _pipeline_returning([])
        monkeypatch.setattr(sys, "argv", ["benchmark_modality_recall.py"])
        with (
            patch("benchmark_modality_recall.load_jsonl", return_value=pairs),
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                return_value=pipeline,
            ),
        ):
            benchmark_modality_recall.main()

        pipeline.retrieve_sync.assert_not_called()
