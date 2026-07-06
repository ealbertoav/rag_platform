"""T-151 benchmark tests — chunk size sweep on the golden QA dataset.

Run with:
    uv run pytest tests/benchmarks/test_chunk_size_sweep.py -v -s

Skipped when the golden QA dataset contains only placeholder rows (default state).
"""

from __future__ import annotations

import pytest

from src.core.constants import DATASETS_DIR
from src.evals.e2e.chunk_size_sweep import ChunkSizeSweep, build_sweep_plan, load_sweep_sizes
from src.evals.e2e.technique_benchmark import filter_qa_pairs, has_real_qa_data, load_qa_pairs

_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"
_QA_PAIRS = filter_qa_pairs(load_qa_pairs(_QA_PATH))

pytestmark = pytest.mark.skipif(
    not has_real_qa_data(_QA_PAIRS),
    reason="Golden QA dataset contains only placeholders — populate via T-040 first.",
)


@pytest.fixture(scope="module")
def sweep() -> ChunkSizeSweep:
    return ChunkSizeSweep()


class TestChunkSizeSweepIntegration:
    def test_dry_run_plan(self):
        sizes = load_sweep_sizes()[:2]
        plan = build_sweep_plan(sizes)
        assert len(plan) == len(sizes)

    @pytest.mark.asyncio
    async def test_dataset_has_real_rows(self):
        assert len(_QA_PAIRS) > 0

    @pytest.mark.asyncio
    async def test_dry_run_report(self, sweep):
        sizes = load_sweep_sizes()[:1]
        report = await sweep.run(_QA_PAIRS[:1], sizes, dry_run=True)
        assert report.dry_run is True
        assert len(report.plan) == 1
