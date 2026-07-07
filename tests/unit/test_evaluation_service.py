"""Unit tests for EvaluationService."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.services.evaluation_service import EvaluationService
from src.evals.e2e.rag_benchmark import BenchmarkReport


def _service(qa_path: Path | None = None) -> EvaluationService:
    return EvaluationService(chat_pipeline=MagicMock(), qa_dataset_path=qa_path)


class TestLoadQa:
    def test_loads_valid_rows(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps(
                [
                    {"question": "What is EKS?", "relevant_chunks": ["c1", "c2"]},
                    {"question": "What is IAM?", "relevant_chunks": ["c3"]},
                ]
            ),
            encoding="utf-8",
        )
        pairs = _service(path)._load_qa()
        assert len(pairs) == 2

    def test_skips_placeholder_rows(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps(
                [
                    {"question": "Real?", "relevant_chunks": ["actual-id"]},
                    {"question": "Placeholder?", "relevant_chunks": ["chunk_id_001"]},
                ]
            ),
            encoding="utf-8",
        )
        pairs = _service(path)._load_qa()
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Real?"

    def test_skips_empty_relevant_chunks(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps(
                [
                    {"question": "Real?", "relevant_chunks": ["c0"]},
                    {"question": "Empty?", "relevant_chunks": []},
                ]
            ),
            encoding="utf-8",
        )
        pairs = _service(path)._load_qa()
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Real?"

    def test_non_list_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        assert _service(path)._load_qa() == []

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert _service(tmp_path / "missing.json")._load_qa() == []

    def test_invalid_json_returns_empty(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text("not json", encoding="utf-8")
        assert _service(path)._load_qa() == []


class TestEvaluationServiceRun:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_zero_sample_report(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        svc = _service(path)
        report = await svc.run()
        assert report.total_samples == 0
        assert report.passed is False

    @pytest.mark.asyncio
    async def test_run_saves_report(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps([{"question": "Q?", "relevant_chunks": ["c1"]}]),
            encoding="utf-8",
        )
        expected = BenchmarkReport(
            timestamp="20250101T000000",
            total_samples=1,
            mean_recall_at_5=0.9,
            mean_faithfulness=0.9,
            mean_relevance=0.9,
            mean_context_precision=0.9,
            mean_hallucination=0.05,
            recall_threshold=0.5,
            faithfulness_threshold=0.8,
            relevance_threshold=0.75,
            context_precision_threshold=0.7,
            hallucination_threshold=0.1,
            passed=True,
        )
        svc = _service(path)
        with (
            patch.object(svc._benchmark, "run", new_callable=AsyncMock, return_value=expected),
            patch.object(BenchmarkReport, "save") as mock_save,
        ):
            report = await svc.run()
        assert report is expected
        mock_save.assert_called_once()


class TestEvaluationServiceFromSettings:
    def test_from_settings_returns_instance(self):
        pipeline = MagicMock()
        svc = EvaluationService.from_settings(pipeline)
        assert isinstance(svc, EvaluationService)
        assert svc._pipeline is pipeline
