from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import time
import tracemalloc
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import yaml
from rich.console import Console
from rich.table import Table

from src.core.constants import EXPORTS_DIR, ROOT
from src.domain.entities.chunk import Chunk
from src.infrastructure.vectordb.bm25_disk import DiskBM25Index
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.graph_retriever import GraphRetriever

if TYPE_CHECKING:
    from src.rag.pipelines.chat_pipeline import ChatPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "InfraBenchmark",
    "InfraBenchmarkReport",
    "InfraBenchmarkThresholds",
    "RegressionWarning",
    "ScenarioMetrics",
    "build_bm25_fixture_chunks",
    "build_default_graph_retriever",
    "build_default_pipeline",
    "compare_to_baseline",
    "default_baseline_path",
    "load_infra_baseline",
    "load_infra_thresholds",
    "measure_stream_inter_token_latencies_ms",
    "percentile",
    "report_to_baseline_payload",
    "save_infra_baseline",
]

_EVALS_CONFIG_PATH = ROOT / "configs" / "evals.yaml"
_DEFAULT_BASELINE_PATH = EXPORTS_DIR / "infra_baseline.json"


class _Bm25SearchMetrics(TypedDict):
    p50_ms: float
    p95_ms: float
    samples: int
    memory_bytes: int


@dataclasses.dataclass(frozen=True)
class InfraBenchmarkThresholds:
    regression_p95_pct: float = 10.0
    concurrent_chat_count: int = 10
    concurrent_chat_timeout_s: float = 120.0
    bm25_fixture_chunks: int = 100_000
    bm25_search_iterations: int = 20
    graph_search_iterations: int = 10
    streaming_sample_question: str = "What is retrieval augmented generation?"
    baseline_path: Path = _DEFAULT_BASELINE_PATH


@dataclasses.dataclass
class ScenarioMetrics:
    name: str
    p50_ms: float
    p95_ms: float
    samples: int = 0
    failures: int = 0
    memory_bytes: int | None = None
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "samples": self.samples,
            "failures": self.failures,
            "error": self.error,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }
        if self.memory_bytes is not None:
            payload["memory_bytes"] = self.memory_bytes
        return payload


@dataclasses.dataclass
class RegressionWarning:
    scenario: str
    metric: str
    baseline_value: float
    current_value: float
    pct_change: float

    def message(self) -> str:
        return (
            f"{self.scenario}.{self.metric}: p95 regressed "
            + f"{self.pct_change:+.1f}% "
            + f"({self.baseline_value:.2f} → {self.current_value:.2f} ms)"
        )


@dataclasses.dataclass
class InfraBenchmarkReport:
    timestamp: str
    scenarios: list[ScenarioMetrics]
    skipped: bool = False
    skip_reason: str = ""

    def scenario_map(self) -> dict[str, ScenarioMetrics]:
        return {scenario.name: scenario for scenario in self.scenarios}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "timestamp": self.timestamp,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Infra benchmark report saved to %s", path)

    def summary(self) -> str:
        if self.skipped:
            return f"Infra benchmark skipped: {self.skip_reason}"
        lines = [f"Infra Benchmark [{self.timestamp}]"]
        for scenario in self.scenarios:
            if scenario.skipped:
                lines.append(f"  {scenario.name}: skipped ({scenario.skip_reason})")
            elif scenario.error:
                lines.append(f"  {scenario.name}: error ({scenario.error})")
            else:
                mem = (
                    f", memory={scenario.memory_bytes}B"
                    if scenario.memory_bytes is not None
                    else ""
                )
                fail = f", failures={scenario.failures}" if scenario.failures else ""
                lines.append(
                    f"  {scenario.name}: p50={scenario.p50_ms:.2f}ms "
                    + f"p95={scenario.p95_ms:.2f}ms "
                    + f"samples={scenario.samples}{fail}{mem}"
                )
        return "\n".join(lines)

    def print_table(self, console: Console | None = None) -> None:
        out = console or Console()
        if self.skipped:
            out.print(f"[yellow]{self.summary()}[/yellow]")
            return

        table = Table(title="Infrastructure Benchmark", show_header=True, header_style="bold cyan")
        table.add_column("Scenario", style="white")
        table.add_column("p50 (ms)", justify="right")
        table.add_column("p95 (ms)", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("Failures", justify="right")
        table.add_column("Memory (B)", justify="right")
        table.add_column("Status", justify="center")

        for scenario in self.scenarios:
            if scenario.skipped:
                status = "[yellow]SKIP[/yellow]"
            elif scenario.error:
                status = "[red]ERROR[/red]"
            else:
                status = "[green]OK[/green]"
            table.add_row(
                scenario.name,
                f"{scenario.p50_ms:.2f}" if not scenario.skipped and not scenario.error else "—",
                f"{scenario.p95_ms:.2f}" if not scenario.skipped and not scenario.error else "—",
                str(scenario.samples),
                str(scenario.failures),
                str(scenario.memory_bytes) if scenario.memory_bytes is not None else "—",
                status,
            )
        out.print(table)


def default_baseline_path() -> Path:
    return _DEFAULT_BASELINE_PATH


def percentile(values: list[float], p: float) -> float:
    """Return the *p*th percentile (0–100) of *values*."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_infra_thresholds(config_path: Path | None = None) -> InfraBenchmarkThresholds:
    path = config_path or _EVALS_CONFIG_PATH
    defaults = InfraBenchmarkThresholds()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        logger.warning("Cannot load infra benchmark config from %s: %s", path, exc)
        return defaults

    evals = data.get("evals") or {}
    raw = evals.get("infra_benchmark") or {}
    if not isinstance(raw, dict):
        return defaults

    baseline_raw = raw.get("baseline_path", defaults.baseline_path)
    baseline_path = Path(str(baseline_raw))
    if not baseline_path.is_absolute():
        baseline_path = ROOT / baseline_path

    return InfraBenchmarkThresholds(
        regression_p95_pct=_coerce_float(
            raw.get("regression_p95_pct"), defaults.regression_p95_pct
        ),
        concurrent_chat_count=_coerce_int(
            raw.get("concurrent_chat_count"), defaults.concurrent_chat_count
        ),
        concurrent_chat_timeout_s=_coerce_float(
            raw.get("concurrent_chat_timeout_s"), defaults.concurrent_chat_timeout_s
        ),
        bm25_fixture_chunks=_coerce_int(
            raw.get("bm25_fixture_chunks"), defaults.bm25_fixture_chunks
        ),
        bm25_search_iterations=_coerce_int(
            raw.get("bm25_search_iterations"), defaults.bm25_search_iterations
        ),
        graph_search_iterations=_coerce_int(
            raw.get("graph_search_iterations"), defaults.graph_search_iterations
        ),
        streaming_sample_question=str(
            raw.get("streaming_sample_question", defaults.streaming_sample_question)
        ),
        baseline_path=baseline_path,
    )


def load_infra_baseline(path: Path | None = None) -> dict[str, object]:
    baseline_path = path or _DEFAULT_BASELINE_PATH
    try:
        raw: object = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Cannot load infra baseline from %s: %s", baseline_path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def report_to_baseline_payload(report: InfraBenchmarkReport) -> dict[str, object]:
    scenarios: dict[str, object] = {}
    for scenario in report.scenarios:
        if scenario.skipped or scenario.error:
            continue
        entry: dict[str, object] = {
            "p50_ms": round(scenario.p50_ms, 3),
            "p95_ms": round(scenario.p95_ms, 3),
            "samples": scenario.samples,
            "failures": scenario.failures,
        }
        if scenario.memory_bytes is not None:
            entry["memory_bytes"] = scenario.memory_bytes
        scenarios[scenario.name] = entry
    return {
        "version": 1,
        "description": (
            "T-172 infrastructure latency baseline — "
            + "compare with scripts/benchmark_infra.py --compare"
        ),
        "captured_at": report.timestamp,
        "scenarios": scenarios,
    }


def save_infra_baseline(report: InfraBenchmarkReport, path: Path | None = None) -> Path:
    target = path or _DEFAULT_BASELINE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report_to_baseline_payload(report)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Infra baseline saved to %s", target)
    return target


def compare_to_baseline(
    current: dict[str, ScenarioMetrics],
    baseline: object,
    *,
    regression_p95_pct: float = 10.0,
) -> list[RegressionWarning]:
    """Return warnings when the current p95 exceeds the baseline by *regression_p95_pct* or more."""
    if not isinstance(baseline, dict):
        return []
    raw_scenarios = baseline.get("scenarios")
    if not isinstance(raw_scenarios, dict):
        return []

    warnings: list[RegressionWarning] = []
    for name, metrics in current.items():
        if metrics.skipped or metrics.error:
            continue
        base_entry = raw_scenarios.get(name)
        if not isinstance(base_entry, dict):
            continue
        base_p95 = base_entry.get("p95_ms")
        if not isinstance(base_p95, (int, float)) or isinstance(base_p95, bool) or base_p95 <= 0:
            continue
        pct_change = ((metrics.p95_ms - float(base_p95)) / float(base_p95)) * 100.0
        if pct_change > regression_p95_pct:
            warnings.append(
                RegressionWarning(
                    scenario=name,
                    metric="p95_ms",
                    baseline_value=float(base_p95),
                    current_value=metrics.p95_ms,
                    pct_change=pct_change,
                )
            )
    return warnings


def build_bm25_fixture_chunks(count: int, *, needle_index: int | None = None) -> list[Chunk]:
    """Build a deterministic BM25 corpus for scale benchmarks."""
    if count <= 0:
        return []
    needle_at = needle_index if needle_index is not None else min(42_000, count - 1)
    chunks: list[Chunk] = []
    for i in range(count):
        text = f"token{i % 97} document row {i} content payload"
        if i == needle_at:
            text = "unique needle xyzzy scale fixture marker " + text
        chunks.append(
            Chunk(
                id=f"chunk-{i}",
                document_id=f"doc-{i // 1000}",
                text=text,
                metadata={"source": "infra_benchmark_fixture"},
            )
        )
    return chunks


async def measure_stream_inter_token_latencies_ms(
    stream: Any,
) -> list[float]:
    """Measure milliseconds between consecutive streamed tokens."""
    latencies: list[float] = []
    previous = time.monotonic()
    async for _token in stream:
        now = time.monotonic()
        latencies.append((now - previous) * 1000.0)
        previous = now
    return latencies


class InfraBenchmark:
    """Orchestrate infrastructure latency scenarios for T-172."""

    def __init__(
        self,
        *,
        thresholds: InfraBenchmarkThresholds | None = None,
        pipeline_factory: Callable[[], Awaitable[ChatPipeline]] | None = None,
        graph_retriever_factory: Callable[[], GraphRetriever] | None = None,
        bm25_fixture_builder: Callable[[int], list[Chunk]] | None = None,
        neo4j_available: Callable[[], bool] | None = None,
    ) -> None:
        self._thresholds = thresholds or load_infra_thresholds()
        self._pipeline_factory = pipeline_factory
        self._graph_retriever_factory = graph_retriever_factory
        self._bm25_fixture_builder = bm25_fixture_builder or build_bm25_fixture_chunks
        self._neo4j_available = neo4j_available or _neo4j_reachable

    async def run(
        self,
        scenario_names: list[str] | None = None,
        *,
        timestamp: str | None = None,
    ) -> InfraBenchmarkReport:
        ts = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        selected = scenario_names or [
            "streaming_chat",
            "concurrent_chats",
            "bm25_100k",
            "graph_retrieval",
        ]
        scenarios: list[ScenarioMetrics] = []
        for name in selected:
            runner = _SCENARIO_RUNNERS.get(name)
            if runner is None:
                scenarios.append(
                    ScenarioMetrics(
                        name=name,
                        p50_ms=0.0,
                        p95_ms=0.0,
                        error=f"unknown scenario: {name}",
                    )
                )
                continue
            scenarios.append(await runner(self))

        return InfraBenchmarkReport(timestamp=ts, scenarios=scenarios)

    async def run_streaming_chat(self) -> ScenarioMetrics:
        if self._pipeline_factory is None:
            return ScenarioMetrics(
                name="streaming_chat",
                p50_ms=0.0,
                p95_ms=0.0,
                skipped=True,
                skip_reason="no pipeline factory configured",
            )
        try:
            pipeline = await self._pipeline_factory()
            stream = await pipeline.chat(self._thresholds.streaming_sample_question)
            latencies = await measure_stream_inter_token_latencies_ms(stream)
        except Exception as exc:
            logger.exception("Streaming chat benchmark failed")
            return ScenarioMetrics(
                name="streaming_chat",
                p50_ms=0.0,
                p95_ms=0.0,
                error=str(exc),
            )

        if not latencies:
            return ScenarioMetrics(
                name="streaming_chat",
                p50_ms=0.0,
                p95_ms=0.0,
                samples=0,
                error="stream produced no tokens",
            )
        return ScenarioMetrics(
            name="streaming_chat",
            p50_ms=percentile(latencies, 50),
            p95_ms=percentile(latencies, 95),
            samples=len(latencies),
        )

    async def run_concurrent_chats(self) -> ScenarioMetrics:
        if self._pipeline_factory is None:
            return ScenarioMetrics(
                name="concurrent_chats",
                p50_ms=0.0,
                p95_ms=0.0,
                skipped=True,
                skip_reason="no pipeline factory configured",
            )

        pipeline_factory = self._pipeline_factory
        count = self._thresholds.concurrent_chat_count
        timeout_s = self._thresholds.concurrent_chat_timeout_s
        question = self._thresholds.streaming_sample_question

        async def _one_chat() -> float:
            pipeline = await pipeline_factory()
            started = time.monotonic()
            await asyncio.wait_for(pipeline.chat_full(question), timeout=timeout_s)
            return (time.monotonic() - started) * 1000.0

        results = await asyncio.gather(*[_one_chat() for _ in range(count)], return_exceptions=True)
        latencies: list[float] = []
        failures = 0
        for item in results:
            if isinstance(item, BaseException):
                failures += 1
                logger.warning("Concurrent chat failed: %s", item)
            else:
                latencies.append(item)

        if not latencies and failures:
            return ScenarioMetrics(
                name="concurrent_chats",
                p50_ms=0.0,
                p95_ms=0.0,
                samples=0,
                failures=failures,
                error="all concurrent chats failed",
            )

        return ScenarioMetrics(
            name="concurrent_chats",
            p50_ms=percentile(latencies, 50) if latencies else 0.0,
            p95_ms=percentile(latencies, 95) if latencies else 0.0,
            samples=len(latencies),
            failures=failures,
        )

    async def run_bm25_search(self) -> ScenarioMetrics:
        chunk_count = self._thresholds.bm25_fixture_chunks
        iterations = self._thresholds.bm25_search_iterations
        try:
            metrics = await asyncio.to_thread(
                _run_bm25_search_sync,
                self._bm25_fixture_builder(chunk_count),
                iterations,
            )
        except Exception as exc:
            logger.exception("BM25 benchmark failed")
            return ScenarioMetrics(
                name="bm25_100k",
                p50_ms=0.0,
                p95_ms=0.0,
                error=str(exc),
            )
        return ScenarioMetrics(name="bm25_100k", **metrics)

    async def run_graph_retrieval(self) -> ScenarioMetrics:
        if not self._neo4j_available():
            return ScenarioMetrics(
                name="graph_retrieval",
                p50_ms=0.0,
                p95_ms=0.0,
                skipped=True,
                skip_reason="Neo4j unavailable — enable graph extra and NEO4J__ENABLED=true",
            )
        if self._graph_retriever_factory is None:
            return ScenarioMetrics(
                name="graph_retrieval",
                p50_ms=0.0,
                p95_ms=0.0,
                skipped=True,
                skip_reason="no graph retriever factory configured",
            )
        try:
            retriever = self._graph_retriever_factory()
            latencies = await _run_graph_searches(
                retriever,
                iterations=self._thresholds.graph_search_iterations,
            )
        except Exception as exc:
            logger.exception("Graph retrieval benchmark failed")
            return ScenarioMetrics(
                name="graph_retrieval",
                p50_ms=0.0,
                p95_ms=0.0,
                error=str(exc),
            )
        if not latencies:
            return ScenarioMetrics(
                name="graph_retrieval",
                p50_ms=0.0,
                p95_ms=0.0,
                samples=0,
                error="graph search returned no timings",
            )
        return ScenarioMetrics(
            name="graph_retrieval",
            p50_ms=percentile(latencies, 50),
            p95_ms=percentile(latencies, 95),
            samples=len(latencies),
        )


_SCENARIO_RUNNERS: dict[str, Callable[[InfraBenchmark], Awaitable[ScenarioMetrics]]] = {
    "streaming_chat": InfraBenchmark.run_streaming_chat,
    "concurrent_chats": InfraBenchmark.run_concurrent_chats,
    "bm25_100k": InfraBenchmark.run_bm25_search,
    "graph_retrieval": InfraBenchmark.run_graph_retrieval,
}


async def _run_graph_searches(retriever: GraphRetriever, *, iterations: int) -> list[float]:
    queries = ["What does EKS use?", "How does IAM relate to AWS?", "Explain Kubernetes clusters"]
    latencies: list[float] = []
    for i in range(iterations):
        query = queries[i % len(queries)]
        started = time.monotonic()
        await retriever.search(query, top_k=5)
        latencies.append((time.monotonic() - started) * 1000.0)
    return latencies


def _run_bm25_search_sync(chunks: list[Chunk], iterations: int) -> _Bm25SearchMetrics:
    import tempfile

    tracemalloc.start()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="infra_bm25_"))
        index = DiskBM25Index(tmp, segment_size=5_000)
        batch: list[Chunk] = []
        with index.deferred_rebuild():
            for chunk in chunks:
                batch.append(chunk)
                if len(batch) >= 5_000:
                    index.add(batch)
                    batch = []
            if batch:
                index.add(batch)

        latencies: list[float] = []
        for i in range(iterations):
            query = "unique needle xyzzy" if i % 2 == 0 else f"token{i % 97} document"
            started = time.monotonic()
            hits = index.search(query, top_k=5)
            latencies.append((time.monotonic() - started) * 1000.0)
            if i == 0 and not hits:
                raise RuntimeError("BM25 fixture search returned no hits")

        memory_bytes = index.memory_resident_bytes()
        return {
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "samples": len(latencies),
            "memory_bytes": memory_bytes,
        }
    finally:
        tracemalloc.stop()


def _neo4j_reachable() -> bool:
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import Neo4jError
    except ImportError:
        return False

    try:
        from src.core.settings import settings

        if not settings.neo4j.enabled:
            return False

        cfg = settings.neo4j
        password = (
            cfg.password.get_secret_value()
            if hasattr(cfg.password, "get_secret_value")
            else str(cfg.password)
        )
        driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, password))
        try:
            driver.verify_connectivity()
            return True
        finally:
            driver.close()
    except (OSError, TimeoutError, ConnectionError, Neo4jError):
        return False


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


async def build_default_pipeline() -> ChatPipeline:
    from src.evals.e2e.technique_benchmark import build_benchmark_pipeline

    return cast("ChatPipeline", build_benchmark_pipeline())


def build_default_graph_retriever() -> GraphRetriever:
    from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
    from src.infrastructure.vectordb.bm25 import BM25Index

    llm = LlamaCppProvider.from_settings()
    bm25 = BM25Retriever(BM25Index.load_or_create())
    return GraphRetriever.from_settings(llm=llm, bm25=bm25)
