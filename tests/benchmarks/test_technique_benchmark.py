"""T-150 benchmark tests — technique comparison on the golden QA dataset.

Run with:
    uv run pytest tests/benchmarks/test_technique_benchmark.py -v -s

Skipped when the golden QA dataset contains only placeholder rows (default state).
"""

from __future__ import annotations

import pytest

from src.core.constants import DATASETS_DIR
from src.evals.e2e.technique_benchmark import (
    TechniqueBenchmark,
    filter_qa_pairs,
    has_real_qa_data,
    load_qa_pairs,
)

_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"
_QA_PAIRS = filter_qa_pairs(load_qa_pairs(_QA_PATH))

pytestmark = pytest.mark.skipif(
    not has_real_qa_data(_QA_PAIRS),
    reason="Golden QA dataset contains only placeholders — populate via T-040 first.",
)


@pytest.fixture(scope="module")
def benchmark() -> TechniqueBenchmark:
    return TechniqueBenchmark()


class TestTechniqueBenchmarkIntegration:
    @pytest.mark.asyncio
    async def test_dataset_has_real_rows(self):
        assert len(_QA_PAIRS) > 0

    @pytest.mark.asyncio
    async def test_baseline_technique_runs(self, benchmark):
        report = await benchmark.run(_QA_PAIRS[:1], ["baseline"])
        assert report.skipped is False
        assert len(report.results) == 1
        assert report.results[0].technique == "baseline"
        assert report.results[0].total_samples == 1

    @pytest.mark.asyncio
    async def test_report_summary_and_table(self, benchmark, capsys):
        report = await benchmark.run(_QA_PAIRS[:1], ["baseline"])
        report.print_table()
        assert "Technique Benchmark" in report.summary()
        assert "baseline" in capsys.readouterr().out
