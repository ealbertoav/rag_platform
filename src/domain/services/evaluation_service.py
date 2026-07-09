from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.constants import DATASETS_DIR, EXPORTS_DIR
from src.evals.e2e.benchmark_samples import BenchmarkPipeline
from src.evals.e2e.rag_benchmark import BenchmarkReport, RAGBenchmark
from src.evals.golden_dataset import filter_real_qa_pairs

logger = logging.getLogger(__name__)

_DEFAULT_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"


class EvaluationService:
    """Orchestrates end-to-end evaluation against the golden QA dataset.

    Wires together "RAGBenchmark" (T-043) with the configured thresholds
    and persists the report to "data/exports/".
    """

    def __init__(
        self,
        chat_pipeline: BenchmarkPipeline,
        recall_threshold: float = 0.5,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.75,
        context_precision_threshold: float = 0.7,
        hallucination_threshold: float = 0.1,
        qa_dataset_path: Path | None = None,
    ) -> None:
        self._pipeline: BenchmarkPipeline = chat_pipeline
        self._qa_path: Path = qa_dataset_path or _DEFAULT_QA_PATH
        self._recall_threshold: float = recall_threshold
        self._faithfulness_threshold: float = faithfulness_threshold
        self._relevance_threshold: float = relevance_threshold
        self._context_precision_threshold: float = context_precision_threshold
        self._hallucination_threshold: float = hallucination_threshold
        self._benchmark: RAGBenchmark = RAGBenchmark(
            recall_threshold=recall_threshold,
            faithfulness_threshold=faithfulness_threshold,
            relevance_threshold=relevance_threshold,
            context_precision_threshold=context_precision_threshold,
            hallucination_threshold=hallucination_threshold,
        )

    async def run(self) -> BenchmarkReport:
        """Load the QA dataset, run the benchmark, save, and return the report."""
        qa_pairs = self._load_qa()
        if not qa_pairs:
            logger.warning("QA dataset is empty — returning zero-sample report")
            ts = _now()
            return BenchmarkReport(
                timestamp=ts,
                total_samples=0,
                mean_recall_at_5=0.0,
                mean_faithfulness=0.0,
                mean_relevance=0.0,
                mean_context_precision=0.0,
                mean_hallucination=0.0,
                recall_threshold=self._recall_threshold,
                faithfulness_threshold=self._faithfulness_threshold,
                relevance_threshold=self._relevance_threshold,
                context_precision_threshold=self._context_precision_threshold,
                hallucination_threshold=self._hallucination_threshold,
                passed=False,
            )

        ts = _now()
        report = await self._benchmark.run(self._pipeline, qa_pairs, timestamp=ts)

        output = EXPORTS_DIR / f"benchmark_{ts}.json"
        report.save(output)
        logger.info("Evaluation complete — report saved to %s", output)
        return report

    @classmethod
    def from_settings(cls, chat_pipeline: BenchmarkPipeline) -> EvaluationService:
        return cls(
            chat_pipeline=chat_pipeline,
            recall_threshold=0.5,
            faithfulness_threshold=0.8,
            relevance_threshold=0.75,
            context_precision_threshold=0.7,
            hallucination_threshold=0.1,
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _load_qa(self) -> list[dict[str, object]]:
        try:
            raw: object = json.loads(self._qa_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
            candidates = [item for item in raw if isinstance(item, dict)]
            return filter_real_qa_pairs(candidates)
        except (OSError, ValueError) as exc:
            logger.warning("Cannot load QA dataset from %s: %s", self._qa_path, exc)
            return []


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
