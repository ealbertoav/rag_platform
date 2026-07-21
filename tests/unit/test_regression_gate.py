"""Unit tests for src/evals/regression_gate.py (T-152)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.evals.golden_dataset import MIN_QA_PAIRS
from src.evals.regression_gate import (
    GateStatus,
    RegressionGateResult,
    baseline_float,
    baseline_int,
    check_live_retrieval_regression,
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


def _standard_golden_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "qa.json", tmp_path / "retrieval.json", tmp_path / "baseline.json"


def _standard_qa_rows() -> list[dict[str, object]]:
    return [_real_qa_row(i) for i in range(1, MIN_QA_PAIRS + 1)]


def _standard_retrieval_rows() -> list[dict[str, object]]:
    return [_real_retrieval_row(i) for i in range(1, MIN_QA_PAIRS + 1)]


def _default_baseline() -> dict[str, object]:
    return {"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.5}


def _write_and_run_gate(
    tmp_path: Path,
    *,
    qa_rows: list[dict[str, object]] | None = None,
    retrieval_rows: list[dict[str, object]] | None = None,
    baseline: dict[str, object] | None = None,
):
    qa_path, retrieval_path, baseline_path = _standard_golden_paths(tmp_path)
    _write_json(qa_path, qa_rows or _standard_qa_rows())
    _write_json(retrieval_path, retrieval_rows or _standard_retrieval_rows())
    _write_json(baseline_path, baseline or _default_baseline())
    return check_regression_gate(
        qa_path=qa_path,
        retrieval_path=retrieval_path,
        baseline_path=baseline_path,
    )


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
        assert baseline_int({"min_samples": 25}, "min_samples", MIN_QA_PAIRS) == 25

    def test_baseline_int_accepts_float(self):
        assert baseline_int({"min_samples": 25.0}, "min_samples", MIN_QA_PAIRS) == 25

    def test_baseline_int_accepts_numeric_string(self):
        assert baseline_int({"min_samples": "25"}, "min_samples", MIN_QA_PAIRS) == 25

    def test_baseline_int_rejects_bool_and_invalid(self):
        assert baseline_int({"min_samples": True}, "min_samples", MIN_QA_PAIRS) == MIN_QA_PAIRS
        assert baseline_int({"min_samples": "bad"}, "min_samples", MIN_QA_PAIRS) == MIN_QA_PAIRS
        assert baseline_int({"min_samples": []}, "min_samples", MIN_QA_PAIRS) == MIN_QA_PAIRS

    def test_baseline_float_accepts_number(self):
        assert baseline_float({"min_recall_at_5": 0.75}, "min_recall_at_5", 0.5) == 0.75

    def test_baseline_float_accepts_numeric_string(self):
        assert baseline_float({"min_recall_at_5": "0.75"}, "min_recall_at_5", 0.5) == 0.75

    def test_baseline_float_rejects_bool_and_invalid(self):
        assert baseline_float({"min_recall_at_5": False}, "min_recall_at_5", 0.5) == 0.5
        assert baseline_float({"min_recall_at_5": "bad"}, "min_recall_at_5", 0.5) == 0.5
        assert baseline_float({"min_recall_at_5": {}}, "min_recall_at_5", 0.5) == 0.5


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

    @pytest.mark.parametrize(
        "first_row",
        [
            {"id": "retrieval_001", "query": "?", "relevant_chunk_ids": []},
            {"id": "retrieval_001", "query": "?", "relevant_chunk_ids": "bad"},
        ],
    )
    def test_fails_when_first_row_has_no_valid_relevant_chunk_ids(
        self, tmp_path: Path, first_row: dict[str, object]
    ):
        rows = _standard_retrieval_rows()
        rows[0] = first_row
        result = _write_and_run_gate(tmp_path, retrieval_rows=rows)
        assert result.status == GateStatus.FAILED
        assert "no relevant_chunk_ids" in result.message

    def test_fails_when_recall_below_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "src.evals.regression_gate.oracle_recall_at_k",
            lambda *_args, **_kwargs: 0.0,
        )
        result = _write_and_run_gate(
            tmp_path,
            baseline={"min_samples": MIN_QA_PAIRS, "min_recall_at_5": 0.99},
        )
        assert result.status == GateStatus.FAILED
        assert "Recall@5" in result.message

    def test_passes_with_real_data(self, tmp_path: Path):
        result = _write_and_run_gate(tmp_path)
        assert result.status == GateStatus.PASSED
        assert "PASSED" in result.message

    def test_passes_with_multi_chunk_rows(self, tmp_path: Path):
        """Oracle Recall@5 must not fail when a row has more than k relevant chunks."""
        multi_chunk_ids = [f"rag_c{i:03d}" for i in range(1, 12)]
        qa_rows = _standard_qa_rows()
        qa_rows[0] = {
            "question": "Question 1?",
            "answer": "Answer 1.",
            "relevant_chunks": multi_chunk_ids,
        }
        retrieval_rows = _standard_retrieval_rows()
        retrieval_rows[0] = {
            "id": "retrieval_001",
            "query": "Question 1?",
            "relevant_chunk_ids": multi_chunk_ids,
        }
        result = _write_and_run_gate(
            tmp_path,
            qa_rows=qa_rows,
            retrieval_rows=retrieval_rows,
        )
        assert result.status == GateStatus.PASSED

    def test_coerces_string_baseline_thresholds(self, tmp_path: Path):
        result = _write_and_run_gate(
            tmp_path,
            baseline={"min_samples": str(MIN_QA_PAIRS), "min_recall_at_5": "0.5"},
        )
        assert result.status == GateStatus.PASSED

    def test_falls_back_on_invalid_baseline_values(self, tmp_path: Path):
        result = _write_and_run_gate(
            tmp_path,
            baseline={"min_samples": "not-a-number", "min_recall_at_5": True},
        )
        assert result.status == GateStatus.PASSED

    def test_fails_when_retrieval_out_of_sync_with_qa(self, tmp_path: Path):
        result = _write_and_run_gate(
            tmp_path,
            retrieval_rows=[
                {
                    "id": f"retrieval_{i:03d}",
                    "query": f"Mismatched question {i}?",
                    "relevant_chunk_ids": [f"rag_c{i:03d}"],
                }
                for i in range(1, MIN_QA_PAIRS + 1)
            ],
        )
        assert result.status == GateStatus.FAILED
        assert "out of sync" in result.message


class TestRegressionGateMain:
    def test_main_prints_pass_message(self, capsys: pytest.CaptureFixture[str]):
        with (
            patch(
                "src.evals.regression_gate.check_regression_gate",
                return_value=RegressionGateResult(GateStatus.PASSED, "ok"),
            ),
            patch(
                "src.evals.regression_gate.check_live_retrieval_regression",
                new_callable=AsyncMock,
                return_value=RegressionGateResult(GateStatus.SKIPPED, "live-skip"),
            ),
        ):
            main()
        out = capsys.readouterr().out
        assert "ok" in out
        assert "live-skip" in out

    def test_main_exits_one_on_failure(self, capsys: pytest.CaptureFixture[str]):
        with (
            patch(
                "src.evals.regression_gate.check_regression_gate",
                return_value=RegressionGateResult(GateStatus.FAILED, "bad"),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
        assert "bad" in capsys.readouterr().out

    def test_main_prints_skip_message(self, capsys: pytest.CaptureFixture[str]):
        with (
            patch(
                "src.evals.regression_gate.check_regression_gate",
                return_value=RegressionGateResult(GateStatus.SKIPPED, "skip"),
            ),
            patch(
                "src.evals.regression_gate.check_live_retrieval_regression",
                new_callable=AsyncMock,
                return_value=RegressionGateResult(GateStatus.SKIPPED, "live-skip"),
            ),
        ):
            main()
        out = capsys.readouterr().out
        assert "skip" in out
        assert "live-skip" in out

    def test_main_exits_one_on_live_failure(self, capsys: pytest.CaptureFixture[str]):
        with (
            patch(
                "src.evals.regression_gate.check_regression_gate",
                return_value=RegressionGateResult(GateStatus.PASSED, "ok"),
            ),
            patch(
                "src.evals.regression_gate.check_live_retrieval_regression",
                new_callable=AsyncMock,
                return_value=RegressionGateResult(GateStatus.FAILED, "live-bad"),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1
        assert "live-bad" in capsys.readouterr().out


class TestCheckLiveRetrievalRegression:
    async def test_skips_when_no_real_rows(self, tmp_path: Path):
        result = await check_live_retrieval_regression(
            retrieval_path=tmp_path / "missing.json",
        )
        assert result.status == GateStatus.SKIPPED
        assert "skipping" in result.message.lower()

    async def test_skips_when_qdrant_unreachable(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        _write_json(retrieval, _standard_retrieval_rows())
        with patch("src.evals.regression_gate._qdrant_reachable", return_value=False):
            result = await check_live_retrieval_regression(retrieval_path=retrieval)
        assert result.status == GateStatus.SKIPPED
        assert "qdrant" in result.message.lower()

    async def test_skips_when_bm25_index_empty(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        _write_json(retrieval, _standard_retrieval_rows())
        fake_bm25 = type("FakeBM25", (), {"size": 0})()
        with (
            patch("src.evals.regression_gate._qdrant_reachable", return_value=True),
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=fake_bm25,
            ),
        ):
            result = await check_live_retrieval_regression(retrieval_path=retrieval)
        assert result.status == GateStatus.SKIPPED
        assert "bm25" in result.message.lower()

    async def test_skips_when_pipeline_construction_fails(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        _write_json(retrieval, _standard_retrieval_rows())
        fake_bm25 = type("FakeBM25", (), {"size": 1})()
        with (
            patch("src.evals.regression_gate._qdrant_reachable", return_value=True),
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=fake_bm25,
            ),
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                side_effect=RuntimeError("no model weights"),
            ),
        ):
            result = await check_live_retrieval_regression(retrieval_path=retrieval)
        assert result.status == GateStatus.SKIPPED
        assert "pipeline unavailable" in result.message.lower()

    async def test_fails_when_live_recall_below_threshold(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        _write_json(retrieval, _standard_retrieval_rows())
        _write_json(baseline, {"min_recall_at_5": 0.99})
        fake_bm25 = type("FakeBM25", (), {"size": 1})()

        fake_pipeline = type(
            "FakePipeline",
            (),
            {"retrieve": AsyncMock(return_value=type("R", (), {"chunks": []})())},
        )()

        with (
            patch("src.evals.regression_gate._qdrant_reachable", return_value=True),
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=fake_bm25,
            ),
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                return_value=fake_pipeline,
            ),
        ):
            result = await check_live_retrieval_regression(
                retrieval_path=retrieval, baseline_path=baseline
            )
        assert result.status == GateStatus.FAILED
        assert "live Recall@5" in result.message

    async def test_passes_when_live_retrieval_finds_relevant_chunks(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        baseline = tmp_path / "baseline.json"
        rows = _standard_retrieval_rows()
        _write_json(retrieval, rows)
        _write_json(baseline, {"min_recall_at_5": 0.5})
        fake_bm25 = type("FakeBM25", (), {"size": 1})()

        class FakePipeline:
            async def retrieve(self, query):
                row = next(r for r in rows if r["query"] == query.text)
                chunks = [type("C", (), {"id": cid})() for cid in row["relevant_chunk_ids"]]
                return type("R", (), {"chunks": chunks})()

        fake_pipeline = FakePipeline()

        with (
            patch("src.evals.regression_gate._qdrant_reachable", return_value=True),
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=fake_bm25,
            ),
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                return_value=fake_pipeline,
            ),
        ):
            result = await check_live_retrieval_regression(
                retrieval_path=retrieval, baseline_path=baseline
            )
        assert result.status == GateStatus.PASSED
        assert "live Recall@5" in result.message

    async def test_targets_dedicated_eval_collection_and_bm25_path(self, tmp_path: Path):
        """The live check must use the eval-dedicated collection/index (#96),
        never whatever settings.qdrant.collection a developer's environment
        happens to have configured, so unrelated ad-hoc ingestion can't
        contaminate the golden corpus's retrieval signal."""
        from src.core.constants import BM25_EVAL_DISK_PATH, EVAL_COLLECTION_NAME

        retrieval = tmp_path / "retrieval.json"
        _write_json(retrieval, _standard_retrieval_rows())
        fake_bm25 = type("FakeBM25", (), {"size": 1})()

        with (
            patch("src.evals.regression_gate._qdrant_reachable", return_value=True),
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=fake_bm25,
            ) as mock_load_or_create,
            patch(
                "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings",
            ) as mock_from_settings,
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                side_effect=RuntimeError("no model weights"),
            ),
        ):
            result = await check_live_retrieval_regression(retrieval_path=retrieval)

        mock_load_or_create.assert_called_once_with(BM25_EVAL_DISK_PATH, backend="disk")
        mock_from_settings.assert_called_once_with(collection=EVAL_COLLECTION_NAME)
        assert result.status == GateStatus.SKIPPED

    async def test_fails_when_retrieval_raises_mid_run(self, tmp_path: Path):
        retrieval = tmp_path / "retrieval.json"
        _write_json(retrieval, _standard_retrieval_rows())
        fake_bm25 = type("FakeBM25", (), {"size": 1})()
        fake_pipeline = type(
            "FakePipeline",
            (),
            {"retrieve": AsyncMock(side_effect=RuntimeError("qdrant timed out"))},
        )()

        with (
            patch("src.evals.regression_gate._qdrant_reachable", return_value=True),
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=fake_bm25,
            ),
            patch(
                "src.rag.pipelines.retrieval_pipeline.RetrievalPipeline.from_settings",
                return_value=fake_pipeline,
            ),
        ):
            result = await check_live_retrieval_regression(retrieval_path=retrieval)
        assert result.status == GateStatus.FAILED
        assert "retrieval raised" in result.message.lower()
