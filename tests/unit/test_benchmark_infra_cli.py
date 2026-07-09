"""T-172 — benchmark_infra.py CLI tests."""

from __future__ import annotations

import argparse
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
    async def test_run_compare_regression_exit_code(self, tmp_path):
        from src.evals.e2e.infra_benchmark import (
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
                return_value=[RegressionWarning("bm25_100k", "p95_ms", 100.0, 200.0, 100.0)],
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
