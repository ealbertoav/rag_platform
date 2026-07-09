"""T-172 — benchmark_infra.py CLI tests."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import benchmark_infra as cli
import pytest


class TestBenchmarkInfraCli:
    def test_parse_scenarios(self):
        assert cli._parse_scenarios("a,b , c") == ["a", "b", "c"]

    def test_needs_pipeline(self):
        assert cli._needs_pipeline(["bm25_100k"]) is False
        assert cli._needs_pipeline(["streaming_chat"]) is True

    @pytest.mark.asyncio
    async def test_run_saves_report(self, tmp_path):
        from src.evals.e2e.infra_benchmark import InfraBenchmarkReport, ScenarioMetrics

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[ScenarioMetrics("bm25_100k", p50_ms=1.0, p95_ms=2.0, samples=1)],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)

        args = argparse.Namespace(
            scenarios="bm25_100k",
            compare=False,
            save_baseline=False,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark),
            patch("src.evals.e2e.infra_benchmark.load_infra_thresholds"),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 0
        exports = list(tmp_path.glob("infra_benchmark_*.json"))
        assert exports

    @pytest.mark.asyncio
    async def test_run_with_pipeline_scenarios(self, tmp_path):
        from src.evals.e2e.infra_benchmark import InfraBenchmarkReport, ScenarioMetrics

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[ScenarioMetrics("streaming_chat", p50_ms=1.0, p95_ms=2.0, samples=1)],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)
        pipeline = MagicMock()

        args = argparse.Namespace(
            scenarios="streaming_chat,concurrent_chats",
            compare=False,
            save_baseline=False,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark) as ctor,
            patch("src.evals.e2e.infra_benchmark.load_infra_thresholds"),
            patch(
                "src.evals.e2e.infra_benchmark.build_default_pipeline",
                new=AsyncMock(return_value=pipeline),
            ),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 0
        assert ctor.call_args.kwargs["pipeline_factory"] is not None
        await ctor.call_args.kwargs["pipeline_factory"]()
        benchmark.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_compare_regression_exit_code(self, tmp_path):
        from src.evals.e2e.infra_benchmark import (
            BaselineComparisonResult,
            InfraBenchmarkReport,
            InfraBenchmarkThresholds,
            RegressionWarning,
            ScenarioMetrics,
        )

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[ScenarioMetrics("bm25_100k", p50_ms=1.0, p95_ms=200.0, samples=1)],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)
        thresholds = InfraBenchmarkThresholds(
            regression_p95_pct=10.0,
            baseline_path=tmp_path / "baseline.json",
        )

        args = argparse.Namespace(
            scenarios="bm25_100k",
            compare=True,
            save_baseline=False,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark),
            patch("src.evals.e2e.infra_benchmark.load_infra_thresholds", return_value=thresholds),
            patch(
                "src.evals.e2e.infra_benchmark.load_infra_baseline",
                return_value={"scenarios": {"bm25_100k": {"p95_ms": 100.0}}},
            ),
            patch(
                "src.evals.e2e.infra_benchmark.compare_to_baseline",
                return_value=BaselineComparisonResult(
                    regressions=[RegressionWarning("bm25_100k", "p95_ms", 100.0, 200.0, 100.0)],
                    failures=[],
                ),
            ),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 2

    @pytest.mark.asyncio
    async def test_run_compare_no_regression(self, tmp_path):
        from src.evals.e2e.infra_benchmark import (
            BaselineComparisonResult,
            InfraBenchmarkReport,
            InfraBenchmarkThresholds,
            ScenarioMetrics,
        )

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[ScenarioMetrics("bm25_100k", p50_ms=1.0, p95_ms=2.0, samples=1)],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)
        thresholds = InfraBenchmarkThresholds(
            regression_p95_pct=10.0,
            baseline_path=tmp_path / "baseline.json",
        )

        args = argparse.Namespace(
            scenarios="bm25_100k",
            compare=True,
            save_baseline=False,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark),
            patch("src.evals.e2e.infra_benchmark.load_infra_thresholds", return_value=thresholds),
            patch(
                "src.evals.e2e.infra_benchmark.load_infra_baseline",
                return_value={"scenarios": {"bm25_100k": {"p95_ms": 2.0}}},
            ),
            patch(
                "src.evals.e2e.infra_benchmark.compare_to_baseline",
                return_value=BaselineComparisonResult(regressions=[], failures=[]),
            ),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 0

    @pytest.mark.asyncio
    async def test_run_compare_scenario_failure_exit_code(self, tmp_path):
        from src.evals.e2e.infra_benchmark import (
            BaselineComparisonResult,
            BaselineScenarioFailure,
            InfraBenchmarkReport,
            InfraBenchmarkThresholds,
            ScenarioMetrics,
        )

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[ScenarioMetrics("bm25_100k", p50_ms=0.0, p95_ms=0.0, error="index down")],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)
        thresholds = InfraBenchmarkThresholds(
            regression_p95_pct=10.0,
            baseline_path=tmp_path / "baseline.json",
        )

        args = argparse.Namespace(
            scenarios="bm25_100k",
            compare=True,
            save_baseline=False,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark),
            patch("src.evals.e2e.infra_benchmark.load_infra_thresholds", return_value=thresholds),
            patch(
                "src.evals.e2e.infra_benchmark.load_infra_baseline",
                return_value={"scenarios": {"bm25_100k": {"p95_ms": 2.0}}},
            ),
            patch(
                "src.evals.e2e.infra_benchmark.compare_to_baseline",
                return_value=BaselineComparisonResult(
                    regressions=[],
                    failures=[BaselineScenarioFailure("bm25_100k", "index down")],
                ),
            ),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 2

    @pytest.mark.asyncio
    async def test_run_compare_failure_count_regression_exit_code(self, tmp_path):
        from src.evals.e2e.infra_benchmark import (
            InfraBenchmarkReport,
            InfraBenchmarkThresholds,
            ScenarioMetrics,
        )

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[
                ScenarioMetrics(
                    "concurrent_chats",
                    p50_ms=10.0,
                    p95_ms=20.0,
                    samples=8,
                    failures=2,
                )
            ],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)
        thresholds = InfraBenchmarkThresholds(
            regression_p95_pct=10.0,
            baseline_path=tmp_path / "baseline.json",
        )

        args = argparse.Namespace(
            scenarios="concurrent_chats",
            compare=True,
            save_baseline=False,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark),
            patch("src.evals.e2e.infra_benchmark.load_infra_thresholds", return_value=thresholds),
            patch(
                "src.evals.e2e.infra_benchmark.load_infra_baseline",
                return_value={
                    "scenarios": {
                        "concurrent_chats": {
                            "p95_ms": 20.0,
                            "failures": 0,
                            "samples": 10,
                        }
                    }
                },
            ),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 2

    @pytest.mark.asyncio
    async def test_run_save_baseline(self, tmp_path):
        from src.evals.e2e.infra_benchmark import InfraBenchmarkReport, ScenarioMetrics

        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[ScenarioMetrics("bm25_100k", p50_ms=1.0, p95_ms=2.0, samples=1)],
        )
        benchmark = MagicMock()
        benchmark.run = AsyncMock(return_value=report)
        baseline_path = tmp_path / "baseline.json"

        args = argparse.Namespace(
            scenarios="bm25_100k",
            compare=False,
            save_baseline=True,
        )

        with (
            patch("src.evals.e2e.infra_benchmark.InfraBenchmark", return_value=benchmark),
            patch(
                "src.evals.e2e.infra_benchmark.load_infra_thresholds",
                return_value=MagicMock(baseline_path=baseline_path),
            ),
            patch("src.core.constants.EXPORTS_DIR", tmp_path),
        ):
            code = await cli.run(args)

        assert code == 0
        assert baseline_path.is_file()


class TestBenchmarkInfraMain:
    def test_main_exits_with_run_result(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["benchmark_infra.py", "--scenarios", "bm25_100k"])
        with (
            patch.object(cli, "run", new=AsyncMock(return_value=0)),
            pytest.raises(SystemExit) as exc,
        ):
            cli.main()
        assert exc.value.code == 0

    def test_main_applies_llm_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark_infra.py", "--scenarios", "bm25_100k", "--llm-config", "cfg.yaml"],
        )
        with (
            patch.object(cli, "apply_llm_config") as apply_cfg,
            patch.object(cli, "run", new=AsyncMock(return_value=0)),
            pytest.raises(SystemExit),
        ):
            cli.main()
        apply_cfg.assert_called_once_with("cfg.yaml")

    def test_module_entrypoint(self, monkeypatch: pytest.MonkeyPatch):
        import runpy
        from pathlib import Path

        monkeypatch.setattr(sys, "argv", ["benchmark_infra.py", "--scenarios", "bm25_100k"])
        with (
            patch.object(cli, "run", new=AsyncMock(return_value=0)),
            pytest.raises(SystemExit) as exc,
        ):
            runpy.run_path(str(Path(cli.__file__)), run_name="__main__")
        assert exc.value.code == 0
