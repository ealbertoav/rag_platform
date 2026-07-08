from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import cast

from src.domain.entities.evaluation import EvalSample
from src.evals.e2e.benchmark_samples import BenchmarkPipeline, pair_str, pair_str_list
from src.evals.generation.context_precision import ContextPrecisionMetric
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.evals.generation.hallucination import HallucinationMetric
from src.evals.generation.relevance import RelevanceMetric
from src.evals.retrieval.recall_at_k import recall_at_k

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SampleResult:
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    recall_at_5: float
    faithfulness: float
    relevance: float
    context_precision: float
    hallucination: float

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass
class BenchmarkReport:
    timestamp: str
    total_samples: int
    mean_recall_at_5: float
    mean_faithfulness: float
    mean_relevance: float
    mean_context_precision: float
    mean_hallucination: float
    recall_threshold: float
    faithfulness_threshold: float
    relevance_threshold: float
    context_precision_threshold: float
    hallucination_threshold: float
    passed: bool
    per_sample: list[SampleResult] = dataclasses.field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": self.timestamp,
            "total_samples": self.total_samples,
            "mean_recall_at_5": self.mean_recall_at_5,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_relevance": self.mean_relevance,
            "mean_context_precision": self.mean_context_precision,
            "mean_hallucination": self.mean_hallucination,
            "passed": self.passed,
            "per_sample": [s.to_dict() for s in self.per_sample],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Benchmark report saved to %s", path)

    def summary(self) -> str:
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        return (
            f"Benchmark {status}  [{self.total_samples} samples]\n"
            f"  Recall@5           {self.mean_recall_at_5:.3f}  (threshold {self.recall_threshold})\n"  # noqa: E501
            f"  Faithfulness       {self.mean_faithfulness:.3f}  (threshold {self.faithfulness_threshold})\n"  # noqa: E501
            f"  Relevance          {self.mean_relevance:.3f}  (threshold {self.relevance_threshold})\n"  # noqa: E501
            f"  Context Precision  {self.mean_context_precision:.3f}  (threshold {self.context_precision_threshold})\n"  # noqa: E501
            f"  Hallucination      {self.mean_hallucination:.3f}  (threshold {self.hallucination_threshold}, lower is better)"  # noqa: E501
        )


class RAGBenchmark:
    """Run end-to-end evaluation of the RAG stack against a golden QA dataset.

    For each QA pair:
      1. Runs "ChatPipeline.benchmark()" → generated answer and context texts
      2. Computes Recall@5 against ground-truth relevant chunk IDs
      3. Scores generation quality via Ragas and DeepEval hallucination
    """

    def __init__(
        self,
        faithfulness: FaithfulnessMetric | None = None,
        relevance: RelevanceMetric | None = None,
        context_precision: ContextPrecisionMetric | None = None,
        hallucination: HallucinationMetric | None = None,
        recall_k: int = 5,
        recall_threshold: float = 0.5,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.75,
        context_precision_threshold: float = 0.7,
        hallucination_threshold: float = 0.1,
    ) -> None:
        self._faith = faithfulness or FaithfulnessMetric(threshold=faithfulness_threshold)
        self._relev = relevance or RelevanceMetric(threshold=relevance_threshold)
        self._ctx = context_precision or ContextPrecisionMetric(
            threshold=context_precision_threshold
        )
        self._halluc = hallucination or HallucinationMetric(threshold=hallucination_threshold)
        self._k = recall_k
        self._recall_threshold = recall_threshold
        self._faith_threshold = faithfulness_threshold
        self._relev_threshold = relevance_threshold
        self._ctx_threshold = context_precision_threshold
        self._halluc_threshold = hallucination_threshold

    async def run(
        self,
        pipeline: BenchmarkPipeline,
        qa_pairs: list[dict[str, object]],
        timestamp: str,
    ) -> BenchmarkReport:
        """Evaluate *pipeline* on *qa_pairs* and return a "BenchmarkReport"."""
        results: list[SampleResult] = []

        for i, pair in enumerate(qa_pairs):
            question = pair_str(pair.get("question"))
            expected = pair_str(pair.get("answer"))
            relevant_ids = pair_str_list(pair.get("relevant_chunks"))

            if not question:
                continue

            try:
                run = await pipeline.benchmark(question)
            except Exception as exc:
                logger.error("Pipeline failed for question %d: %s", i, exc)
                results.append(
                    SampleResult(
                        question=question,
                        expected_answer=expected,
                        generated_answer="",
                        retrieved_ids=[],
                        relevant_ids=relevant_ids,
                        recall_at_5=0.0,
                        faithfulness=0.0,
                        relevance=0.0,
                        context_precision=0.0,
                        hallucination=0.0,
                    )
                )
                continue

            answer = run.answer
            context_texts = run.context_texts
            retrieved_ids = list(answer.sources)
            r5 = recall_at_k(retrieved_ids, relevant_ids, k=self._k)

            sample = EvalSample(
                question=question,
                expected_answer=expected,
                retrieved_chunks=context_texts,
                generated_answer=answer.text,
                parametric_answer=run.parametric_answer,
            )
            faith_score = self._faith.score(sample).score
            relev_score = self._relev.score(sample).score
            ctx_score = self._ctx.score(sample).score
            halluc_score = self._halluc.score(sample).score

            results.append(
                SampleResult(
                    question=question,
                    expected_answer=expected,
                    generated_answer=answer.text,
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    recall_at_5=r5,
                    faithfulness=faith_score,
                    relevance=relev_score,
                    context_precision=ctx_score,
                    hallucination=halluc_score,
                )
            )
            logger.debug(
                "[%d/%d] R@5=%.2f faith=%.2f relev=%.2f ctx=%.2f hall=%.2f",
                i + 1,
                len(qa_pairs),
                r5,
                faith_score,
                relev_score,
                ctx_score,
                halluc_score,
            )

        n = len(results) or 1
        mean_r = sum(r.recall_at_5 for r in results) / n
        mean_f = sum(r.faithfulness for r in results) / n
        mean_v = sum(r.relevance for r in results) / n
        mean_c = sum(r.context_precision for r in results) / n
        mean_h = sum(r.hallucination for r in results) / n

        passed = (
            mean_r >= self._recall_threshold
            and mean_f >= self._faith_threshold
            and mean_v >= self._relev_threshold
            and mean_c >= self._ctx_threshold
            and mean_h < self._halluc_threshold
        )

        return BenchmarkReport(
            timestamp=timestamp,
            total_samples=len(results),
            mean_recall_at_5=mean_r,
            mean_faithfulness=mean_f,
            mean_relevance=mean_v,
            mean_context_precision=mean_c,
            mean_hallucination=mean_h,
            recall_threshold=self._recall_threshold,
            faithfulness_threshold=self._faith_threshold,
            relevance_threshold=self._relev_threshold,
            context_precision_threshold=self._ctx_threshold,
            hallucination_threshold=self._halluc_threshold,
            passed=passed,
            per_sample=results,
        )
