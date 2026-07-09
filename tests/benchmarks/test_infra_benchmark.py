"""T-172 benchmark tests — infrastructure latency scenarios.

Run with live services:
    RUN_INFRA_BENCHMARK=1 uv run pytest tests/benchmarks/test_infra_benchmark.py -v -s

Scenario 5 (concurrent feedback) is covered by test_feedback_concurrency.py.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from src.evals.e2e.infra_benchmark import InfraBenchmark, load_infra_thresholds

_RUN = os.environ.get("RUN_INFRA_BENCHMARK") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN,
    reason="Set RUN_INFRA_BENCHMARK=1 to run live infra benchmarks",
)


@pytest.fixture(scope="module")
def benchmark() -> InfraBenchmark:
    thresholds = load_infra_thresholds()
    return InfraBenchmark(thresholds=thresholds)


class TestInfraBenchmarkIntegration:
    @pytest.mark.asyncio
    async def test_bm25_scenario_runs(self):
        thresholds = load_infra_thresholds()
        small = InfraBenchmark(
            thresholds=replace(
                thresholds,
                bm25_fixture_chunks=2_000,
                bm25_search_iterations=5,
            )
        )
        result = await small.run_bm25_search()
        assert result.error == ""
        assert result.samples == 5
        assert result.memory_bytes is not None

    @pytest.mark.asyncio
    async def test_report_save_and_summary(self, benchmark: InfraBenchmark, tmp_path):
        report = await benchmark.run(["bm25_100k"])
        out = tmp_path / "infra.json"
        report.save(out)
        assert out.is_file()
        assert "bm25_100k" in report.summary()
