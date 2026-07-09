from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import yaml
from rich.console import Console
from rich.table import Table

from src.core.constants import DATASETS_DIR, EXPORTS_DIR, ROOT
from src.core.exceptions import VectorStoreError
from src.domain.entities.evaluation import BenchmarkRun
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.evals.e2e.benchmark_samples import (
    BenchmarkPipeline,
    GenerationMetricAccumulator,
    pair_str,
    pair_str_list,
    pipeline_error_logger,
    score_pipeline_question,
)
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.evals.generation.relevance import RelevanceMetric
from src.evals.golden_dataset import filter_real_qa_pairs, is_placeholder_qa_pair
from src.rag.quality.feedback_loop import record_feedback, score_from_relevant

if TYPE_CHECKING:
    from src.rag.pipelines.agent_pipeline import AgentPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "FeedbackComparison",
    "PipelineFactory",
    "TechniqueBenchmark",
    "TechniqueBenchmarkReport",
    "TechniqueConfig",
    "TechniqueResult",
    "build_benchmark_pipeline",
    "build_feedback_boost_overrides",
    "filter_qa_pairs",
    "has_real_qa_data",
    "is_placeholder_qa_pair",
    "load_qa_pairs",
    "load_technique_configs",
    "merge_technique_overrides",
    "prepare_qa_pairs",
    "run_technique_matrix",
    "temporary_config",
]

_DEFAULT_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"
_EVALS_CONFIG_PATH = ROOT / "configs" / "evals.yaml"

# Baseline disables all optional Phase 11–14 techniques (env overrides win over YAML).
_BASELINE_OVERRIDES: dict[str, str] = {
    "QUERY_EXPANSION__ENABLED": "false",
    "COMPRESSION__ENABLED": "false",
    "RETRIEVAL__HYDE__ENABLED": "false",
    "QUALITY__RELIABLE_RAG__ENABLED": "false",
    "QUALITY__SELF_RAG__ENABLED": "false",
    "QUALITY__FEEDBACK_LOOP__ENABLED": "false",
}

# Per-technique deltas applied on top of baseline.
_TECHNIQUE_DELTA_OVERRIDES: dict[str, dict[str, str]] = {
    "baseline": {},
    "multi_query": {
        "QUERY_EXPANSION__ENABLED": "true",
        "QUERY_EXPANSION__N_VARIANTS": "3",
    },
    "hyde": {"RETRIEVAL__HYDE__ENABLED": "true"},
    "cch": {"COMPRESSION__ENABLED": "true"},
    "reliable_rag": {"QUALITY__RELIABLE_RAG__ENABLED": "true"},
    "self_rag": {"QUALITY__SELF_RAG__ENABLED": "true"},
    "feedback_loop": {
        # Benchmark compares boost on vs. off at identical fusion pool size.
        "QUALITY__FEEDBACK_LOOP__EXPAND_CANDIDATE_POOL": "false",
    },
}


class PipelineFactory(Protocol):
    def __call__(
        self,
        *,
        self_rag: bool = False,
        vector_store: VectorStoreRepository | None = None,
    ) -> BenchmarkPipeline: ...


@dataclasses.dataclass(frozen=True)
class TechniqueConfig:
    name: str
    description: str
    overrides: dict[str, str]


@dataclasses.dataclass
class TechniqueResult:
    technique: str
    total_samples: int
    mean_recall_at_5: float
    mean_faithfulness: float
    mean_relevance: float
    mean_latency_ms: float
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass
class FeedbackComparison:
    """Recall@5 with feedback boost off vs. on after pre-seeding scores."""

    recall_boost_off: float
    recall_boost_on: float
    samples: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass
class TechniqueBenchmarkReport:
    timestamp: str
    techniques: list[str]
    results: list[TechniqueResult]
    feedback_comparison: FeedbackComparison | None = None
    skipped: bool = False
    skip_reason: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "timestamp": self.timestamp,
            "techniques": self.techniques,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "results": [r.to_dict() for r in self.results],
        }
        if self.feedback_comparison is not None:
            payload["feedback_comparison"] = self.feedback_comparison.to_dict()
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Technique benchmark report saved to %s", path)

    def summary(self) -> str:
        if self.skipped:
            return f"Technique benchmark skipped: {self.skip_reason}"
        lines = [
            f"Technique Benchmark [{self.timestamp}]",
            f"  Samples per technique: {self.results[0].total_samples if self.results else 0}",
        ]
        if self.feedback_comparison is not None:
            fc = self.feedback_comparison
            lines.append(
                f"  Feedback loop Recall@5: off={fc.recall_boost_off:.3f} "
                + f"on={fc.recall_boost_on:.3f}"
            )
        return "\n".join(lines)

    def print_table(self, console: Console | None = None) -> None:
        if self.skipped:
            (console or Console()).print(f"[yellow]{self.summary()}[/yellow]")
            return

        table = Table(title="RAG Technique Comparison", show_header=True, header_style="bold cyan")
        table.add_column("Technique", style="white")
        table.add_column("Recall@5", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Relevance", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Status", justify="center")

        for result in self.results:
            status = "[red]ERROR[/red]" if result.error else "[green]OK[/green]"
            table.add_row(
                result.technique,
                f"{result.mean_recall_at_5:.3f}",
                f"{result.mean_faithfulness:.3f}",
                f"{result.mean_relevance:.3f}",
                f"{result.mean_latency_ms:.1f}",
                status,
            )

        (console or Console()).print(table)


filter_qa_pairs = filter_real_qa_pairs


def prepare_qa_pairs(
    pairs: list[dict[str, object]],
    max_samples: int | None = None,
) -> list[dict[str, object]]:
    """Filter placeholders, then cap to *max_samples* real rows."""
    filtered = filter_qa_pairs(pairs)
    if max_samples is not None and max_samples > 0:
        return filtered[:max_samples]
    return filtered


def load_qa_pairs(path: Path | None = None) -> list[dict[str, object]]:
    """Load and filter QA pairs from a *path* (defaults to golden QA dataset)."""
    qa_path = path or _DEFAULT_QA_PATH
    try:
        raw: object = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Cannot load QA dataset from %s: %s", qa_path, exc)
        return []
    if not isinstance(raw, list):
        return []
    candidates = [item for item in raw if isinstance(item, dict)]
    return filter_qa_pairs(candidates)


def has_real_qa_data(pairs: list[dict[str, object]]) -> bool:
    """Return False when *pairs* are empty (e.g., only placeholders in the golden file)."""
    return len(pairs) > 0


def merge_technique_overrides(
    technique: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build env overrides: baseline + single technique delta + optional extras."""
    merged = dict(_BASELINE_OVERRIDES)
    merged.update(_TECHNIQUE_DELTA_OVERRIDES.get(technique, {}))
    if extra:
        merged.update(extra)
    return merged


def build_feedback_boost_overrides(*, boost_enabled: bool) -> dict[str, str]:
    """Env overrides for feedback A/B: pre-seeded scores, same pool size, boost toggled."""
    overrides = merge_technique_overrides("feedback_loop")
    overrides["QUALITY__FEEDBACK_LOOP__ENABLED"] = "true" if boost_enabled else "false"
    overrides["QUALITY__FEEDBACK_LOOP__EXPAND_CANDIDATE_POOL"] = "false"
    return overrides


def load_technique_configs(
    config_path: Path | None = None,
) -> list[TechniqueConfig]:
    """Load technique definitions from configs/evals.yaml (falls back to built-ins)."""
    path = config_path or _EVALS_CONFIG_PATH
    yaml_configs: list[dict[str, Any]] = []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        evals = data.get("evals") or {}
        tb = evals.get("technique_benchmark") or {}
        raw_list = tb.get("configs")
        if isinstance(raw_list, list):
            yaml_configs = [c for c in raw_list if isinstance(c, dict)]
    except (OSError, ValueError) as exc:
        logger.warning("Cannot load technique configs from %s: %s", path, exc)

    if not yaml_configs:
        return _default_technique_configs()

    configs: list[TechniqueConfig] = []
    for entry in yaml_configs:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        description = str(entry.get("description", name))
        raw_overrides = entry.get("overrides") or {}
        overrides = merge_technique_overrides(name)
        if isinstance(raw_overrides, dict):
            overrides.update({str(k): str(v) for k, v in raw_overrides.items()})
        configs.append(TechniqueConfig(name=name, description=description, overrides=overrides))
    return configs if configs else _default_technique_configs()


def _default_technique_configs() -> list[TechniqueConfig]:
    return [
        TechniqueConfig(
            name=name,
            description=name.replace("_", " ").title(),
            overrides=merge_technique_overrides(name),
        )
        for name in (
            "baseline",
            "multi_query",
            "hyde",
            "cch",
            "reliable_rag",
            "self_rag",
            "feedback_loop",
        )
    ]


def reload_settings_module() -> None:
    """Reload the settings singleton after env var changes."""
    import src.core.settings as settings_mod

    _ = importlib.reload(settings_mod)


@contextmanager
def temporary_config(overrides: dict[str, str]) -> Generator[None]:
    """Apply *overrides* to os.environ for the duration of the context."""
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        reload_settings_module()
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                _ = os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        reload_settings_module()


def _relevant_chunk_ids(qa_pairs: list[dict[str, object]]) -> set[str]:
    """Collect unique relevant chunk IDs from *qa_pairs*."""
    chunk_ids: set[str] = set()
    for pair in qa_pairs:
        relevant = pair.get("relevant_chunks")
        if not isinstance(relevant, list):
            continue
        for chunk_id in relevant:
            if isinstance(chunk_id, str) and chunk_id:
                chunk_ids.add(chunk_id)
    return chunk_ids


def _resolve_benchmark_vector_store(vector_store: object | None = None) -> VectorStoreRepository:
    if vector_store is not None:
        return cast(VectorStoreRepository, vector_store)
    from src.infrastructure.vectordb.feedback_store import build_vector_store_from_settings

    return build_vector_store_from_settings()


def _snapshot_feedback_scores(
    vector_store: VectorStoreRepository,
    chunk_ids: set[str],
) -> dict[str, float]:
    """Capture current feedback scores before benchmark seeding."""
    snapshots: dict[str, float] = {}
    for chunk_id in sorted(chunk_ids):
        try:
            snapshots[chunk_id] = vector_store.get_feedback_score(chunk_id)
        except VectorStoreError as exc:
            logger.warning("Could not snapshot feedback for %s: %s", chunk_id, exc)
    return snapshots


def _seed_feedback_scores(
    vector_store: VectorStoreRepository,
    chunk_ids: set[str],
    *,
    seeded_ids: set[str],
) -> int:
    """Pre-seed positive feedback for *chunk_ids*; track successes in *seeded_ids*."""
    seeded = 0
    for chunk_id in sorted(chunk_ids):
        try:
            record_feedback(
                vector_store,
                query_id="technique-benchmark-seed",
                chunk_id=chunk_id,
                score=score_from_relevant(True),
            )
            seeded_ids.add(chunk_id)
            seeded += 1
        except VectorStoreError as exc:
            logger.warning("Could not seed feedback for %s: %s", chunk_id, exc)
    return seeded


def _restore_feedback_scores(
    vector_store: VectorStoreRepository,
    snapshots: dict[str, float],
    *,
    seeded_ids: set[str],
) -> None:
    """Restore pre-benchmark feedback scores for chunks that were seeded."""
    for chunk_id in sorted(seeded_ids):
        try:
            vector_store.set_feedback_score(chunk_id, snapshots[chunk_id])
        except VectorStoreError as exc:
            logger.warning("Could not restore feedback for %s: %s", chunk_id, exc)


@contextmanager
def temporary_feedback_seed(
    qa_pairs: list[dict[str, object]],
    *,
    vector_store: object | None = None,
) -> Generator[int]:
    """Pre-seed positive feedback for benchmark, restoring originals on exit."""
    store = _resolve_benchmark_vector_store(vector_store)
    chunk_ids = _relevant_chunk_ids(qa_pairs)
    snapshots = _snapshot_feedback_scores(store, chunk_ids)
    seeded_ids: set[str] = set()
    try:
        yield _seed_feedback_scores(store, set(snapshots), seeded_ids=seeded_ids)
    finally:
        _restore_feedback_scores(store, snapshots, seeded_ids=seeded_ids)


class _AgentBenchmarkAdapter:
    """Adapter so RAGBenchmark-style eval can run against AgentPipeline (Self-RAG)."""

    def __init__(self, agent: AgentPipeline) -> None:
        self._agent: AgentPipeline = agent

    async def benchmark(self, question: str) -> BenchmarkRun:
        result = await self._agent.chat_full(question)
        return BenchmarkRun(
            answer=result.answer,
            context_texts=list(result.context_texts),
            parametric_answer=result.parametric_answer,
        )


def build_benchmark_pipeline(
    *,
    self_rag: bool = False,
    vector_store: VectorStoreRepository | None = None,
) -> BenchmarkPipeline:
    """Construct a pipeline from the current (reloaded) settings."""
    store = _resolve_benchmark_vector_store(vector_store)
    if self_rag:
        from src.rag.pipelines.agent_pipeline import AgentPipeline

        return _AgentBenchmarkAdapter(AgentPipeline.from_settings(vector_store=store))
    from src.rag.pipelines.chat_pipeline import ChatPipeline

    return cast(BenchmarkPipeline, ChatPipeline.from_settings(vector_store=store))


class TechniqueBenchmark:
    """Compare RAG techniques side-by-side via independent config overrides."""

    def __init__(
        self,
        *,
        faithfulness: FaithfulnessMetric | None = None,
        relevance: RelevanceMetric | None = None,
        recall_k: int = 5,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.75,
    ) -> None:
        self._faith: Any = faithfulness or FaithfulnessMetric(threshold=faithfulness_threshold)
        self._relev: Any = relevance or RelevanceMetric(threshold=relevance_threshold)
        self._k: Any = recall_k

    async def run(
        self,
        qa_pairs: list[dict[str, object]],
        techniques: list[str],
        *,
        timestamp: str | None = None,
        pipeline_factory: PipelineFactory | None = None,
        vector_store: object | None = None,
    ) -> TechniqueBenchmarkReport:
        """Run each technique independently and return a comparison report."""
        ts = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        if not has_real_qa_data(qa_pairs):
            return TechniqueBenchmarkReport(
                timestamp=ts,
                techniques=techniques,
                results=[],
                skipped=True,
                skip_reason="Golden QA dataset contains only placeholders — populate via T-040.",
            )

        configs = {cfg.name: cfg for cfg in load_technique_configs()}
        results: list[TechniqueResult] = []
        feedback_comparison: FeedbackComparison | None = None

        factory = pipeline_factory or build_benchmark_pipeline

        for technique in techniques:
            if technique == "feedback_loop":
                try:
                    feedback_comparison, on_result = await self._run_feedback_comparison(
                        qa_pairs,
                        factory=factory,
                        vector_store=vector_store,
                    )
                    if on_result is not None:
                        results.append(
                            TechniqueResult(
                                technique="feedback_loop",
                                total_samples=on_result.total_samples,
                                mean_recall_at_5=on_result.mean_recall_at_5,
                                mean_faithfulness=on_result.mean_faithfulness,
                                mean_relevance=on_result.mean_relevance,
                                mean_latency_ms=on_result.mean_latency_ms,
                            )
                        )
                except Exception as exc:
                    logger.exception("Technique %s failed", technique)
                    results.append(
                        TechniqueResult(
                            technique="feedback_loop",
                            total_samples=0,
                            mean_recall_at_5=0.0,
                            mean_faithfulness=0.0,
                            mean_relevance=0.0,
                            mean_latency_ms=0.0,
                            error=str(exc),
                        )
                    )
                continue

            cfg = configs.get(technique)
            overrides = cfg.overrides if cfg else merge_technique_overrides(technique)
            try:
                with temporary_config(overrides):
                    pipeline = factory(self_rag=(technique == "self_rag"))
                    result = await self._evaluate_technique(technique, pipeline, qa_pairs)
            except Exception as exc:
                logger.exception("Technique %s failed", technique)
                result = TechniqueResult(
                    technique=technique,
                    total_samples=0,
                    mean_recall_at_5=0.0,
                    mean_faithfulness=0.0,
                    mean_relevance=0.0,
                    mean_latency_ms=0.0,
                    error=str(exc),
                )
            results.append(result)

        return TechniqueBenchmarkReport(
            timestamp=ts,
            techniques=techniques,
            results=results,
            feedback_comparison=feedback_comparison,
        )

    async def _run_feedback_comparison(
        self,
        qa_pairs: list[dict[str, object]],
        *,
        factory: PipelineFactory,
        vector_store: object | None,
    ) -> tuple[FeedbackComparison | None, TechniqueResult | None]:
        """Pre-seed feedback scores and compare Recall@5 with boost off vs. on."""
        store = _resolve_benchmark_vector_store(vector_store)

        def bound_factory(*, self_rag: bool = False) -> BenchmarkPipeline:
            try:
                return factory(self_rag=self_rag, vector_store=store)
            except TypeError:
                return factory(self_rag=self_rag)

        with temporary_feedback_seed(qa_pairs, vector_store=store):
            with temporary_config(build_feedback_boost_overrides(boost_enabled=False)):
                pipeline_off = bound_factory(self_rag=False)
                off_result = await self._evaluate_technique("feedback_off", pipeline_off, qa_pairs)

            with temporary_config(build_feedback_boost_overrides(boost_enabled=True)):
                pipeline_on = bound_factory(self_rag=False)
                on_result = await self._evaluate_technique("feedback_on", pipeline_on, qa_pairs)

        comparison = FeedbackComparison(
            recall_boost_off=off_result.mean_recall_at_5,
            recall_boost_on=on_result.mean_recall_at_5,
            samples=off_result.total_samples,
        )
        return comparison, on_result

    async def _evaluate_technique(
        self,
        technique: str,
        pipeline: BenchmarkPipeline,
        qa_pairs: list[dict[str, object]],
    ) -> TechniqueResult:
        accumulator = GenerationMetricAccumulator()

        for pair in qa_pairs:
            question = pair_str(pair.get("question"))
            expected = pair_str(pair.get("answer"))
            relevant_ids = pair_str_list(pair.get("relevant_chunks"))
            if not question:
                continue

            scores = await score_pipeline_question(
                pipeline=pipeline,
                question=question,
                expected_answer=expected,
                relevant_ids=relevant_ids,
                recall_k=self._k,
                faithfulness=self._faith,
                relevance=self._relev,
                on_pipeline_error=pipeline_error_logger(
                    logger.error,
                    "Pipeline failed for %s / %r: %s",
                    technique,
                    question[:40],
                ),
            )
            accumulator.append(scores)

        means = accumulator.means()
        return TechniqueResult(
            technique=technique,
            total_samples=accumulator.total_samples,
            mean_recall_at_5=means.mean_recall_at_5,
            mean_faithfulness=means.mean_faithfulness,
            mean_relevance=means.mean_relevance,
            mean_latency_ms=means.mean_latency_ms,
        )


def run_technique_matrix(
    techniques: list[str],
    qa_pairs: list[dict[str, object]],
    *,
    output_dir: Path | None = None,
    benchmark: TechniqueBenchmark | None = None,
) -> TechniqueBenchmarkReport:
    """Synchronous entry point for scripts — runs the async benchmark matrix."""
    import asyncio

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    runner = benchmark or TechniqueBenchmark()
    report = asyncio.run(runner.run(qa_pairs, techniques, timestamp=ts))

    if not report.skipped:
        out_dir = output_dir or EXPORTS_DIR
        report.save(out_dir / f"technique_benchmark_{ts}.json")

    return report


# Backward-compatible aliases for tests and internal callers.
_str = pair_str
_str_list = pair_str_list
