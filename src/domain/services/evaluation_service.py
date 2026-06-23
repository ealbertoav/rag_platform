from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.core.constants import DATASETS_DIR, EXPORTS_DIR
from src.evals.e2e.rag_benchmark import BenchmarkReport, RAGBenchmark

logger = logging.getLogger(__name__)

_DEFAULT_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"


class EvaluationService:
    """Orchestrates end-to-end evaluation against the golden QA dataset.

    Wires together "RAGBenchmark" (T-043) with the configured thresholds
    and persists the report to "data/exports/".
    """

    def __init__(
        self,
        chat_pipeline: object,  # ChatPipeline — avoid circular import
        recall_threshold: float = 0.5,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.75,
        qa_dataset_path: Path | None = None,
    ) -> None:
        self._pipeline = chat_pipeline
        self._qa_path = qa_dataset_path or _DEFAULT_QA_PATH
        self._recall_threshold = recall_threshold
        self._faithfulness_threshold = faithfulness_threshold
        self._relevance_threshold = relevance_threshold
        self._benchmark = RAGBenchmark(
            recall_threshold=recall_threshold,
            faithfulness_threshold=faithfulness_threshold,
            relevance_threshold=relevance_threshold,
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
                recall_threshold=self._recall_threshold,
                faithfulness_threshold=self._faithfulness_threshold,
                relevance_threshold=self._relevance_threshold,
                passed=False,
            )

        ts = _now()
        report = await self._benchmark.run(self._pipeline, qa_pairs, timestamp=ts)

        output = EXPORTS_DIR / f"benchmark_{ts}.json"
        report.save(output)
        logger.info("Evaluation complete — report saved to %s", output)
        return report

    @classmethod
    def from_settings(cls, chat_pipeline: object) -> EvaluationService:
        return cls(
            chat_pipeline=chat_pipeline,
            recall_threshold=0.5,
            faithfulness_threshold=0.8,
            relevance_threshold=0.75,
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _load_qa(self) -> list[dict[str, object]]:
        try:
            raw: object = json.loads(self._qa_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
            pairs: list[dict[str, object]] = []
            for item in raw:
                if isinstance(item, dict) and item.get("question"):
                    pairs.append(item)
            # Skip placeholder rows (relevant_chunks filled with "chunk_id_*")
            result: list[dict[str, object]] = []
            for p in pairs:
                chunks = p.get("relevant_chunks")
                if not isinstance(chunks, list) or not all(
                    str(r).startswith("chunk_id_") for r in chunks if isinstance(r, str)
                ):
                    result.append(p)
            return result
        except (OSError, ValueError) as exc:
            logger.warning("Cannot load QA dataset from %s: %s", self._qa_path, exc)
            return []


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
