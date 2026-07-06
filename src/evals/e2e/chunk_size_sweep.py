from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from src.core.constants import CHUNKS_DIR, EXPORTS_DIR, ROOT, SUPPORTED_EXTENSIONS
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.evals.e2e.benchmark_samples import (
    GenerationMetricAccumulator,
    pair_str,
    pipeline_error_logger,
    score_pipeline_question,
)
from src.evals.e2e.technique_benchmark import (
    BenchmarkPipeline,
    has_real_qa_data,
    temporary_config,
)
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.evals.generation.relevance import RelevanceMetric

logger = logging.getLogger(__name__)

_EVALS_CONFIG_PATH = ROOT / "configs" / "evals.yaml"
_DEFAULT_SIZES = (256, 500, 768, 1024)
_DEFAULT_WEIGHTS = {
    "recall": 0.35,
    "faithfulness": 0.35,
    "relevance": 0.20,
    "latency": 0.10,
}


@dataclasses.dataclass(frozen=True)
class SweepWeights:
    recall: float
    faithfulness: float
    relevance: float
    latency: float

    def normalized(self) -> SweepWeights:
        total = self.recall + self.faithfulness + self.relevance + self.latency
        if total <= 0:
            return SweepWeights(0.25, 0.25, 0.25, 0.25)
        return SweepWeights(
            recall=self.recall / total,
            faithfulness=self.faithfulness / total,
            relevance=self.relevance / total,
            latency=self.latency / total,
        )


@dataclasses.dataclass
class ChunkSizeResult:
    chunk_size: int
    total_samples: int
    mean_recall_at_5: float
    mean_faithfulness: float
    mean_relevance: float
    mean_latency_ms: float
    weighted_score: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)  # type: ignore[return-value]


@dataclasses.dataclass
class SweepPlanEntry:
    chunk_size: int
    collection: str
    cache_path: Path
    action: str


@dataclasses.dataclass
class ChunkSizeSweepReport:
    timestamp: str
    sizes: list[int]
    results: list[ChunkSizeResult]
    recommended_size: int | None = None
    skipped: bool = False
    skip_reason: str = ""
    dry_run: bool = False
    plan: list[SweepPlanEntry] = dataclasses.field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "timestamp": self.timestamp,
            "sizes": self.sizes,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "dry_run": self.dry_run,
            "recommended_size": self.recommended_size,
            "results": [r.to_dict() for r in self.results],
            "plan": [
                {
                    "chunk_size": entry.chunk_size,
                    "collection": entry.collection,
                    "cache_path": str(entry.cache_path),
                    "action": entry.action,
                }
                for entry in self.plan
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Chunk size sweep report saved to %s", path)

    def summary(self) -> str:
        if self.dry_run:
            lines = [f"Chunk Size Sweep (dry-run) [{self.timestamp}]"]
            for entry in self.plan:
                lines.append(
                    f"  size={entry.chunk_size} collection={entry.collection} "
                    f"cache={entry.cache_path} action={entry.action}"
                )
            return "\n".join(lines)
        if self.skipped:
            return f"Chunk size sweep skipped: {self.skip_reason}"
        lines = [
            f"Chunk Size Sweep [{self.timestamp}]",
            f"  Sizes evaluated: {', '.join(str(s) for s in self.sizes)}",
        ]
        if self.recommended_size is not None:
            lines.append(f"  Recommended chunk_size: {self.recommended_size}")
        return "\n".join(lines)

    def print_table(self, console: Console | None = None) -> None:
        out = console or Console()
        if self.dry_run:
            out.print("[cyan]Planned chunk size sweep[/cyan]")
            for entry in self.plan:
                out.print(
                    f"  [bold]{entry.chunk_size}[/bold] → {entry.collection} "
                    f"({entry.action}, cache: {entry.cache_path})"
                )
            return
        if self.skipped:
            out.print(f"[yellow]{self.summary()}[/yellow]")
            return

        table = Table(title="Chunk Size Comparison", show_header=True, header_style="bold cyan")
        table.add_column("Chunk Size", justify="right", style="white")
        table.add_column("Recall@5", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Relevance", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Weighted", justify="right")
        table.add_column("Status", justify="center")

        for result in self.results:
            status = "[red]ERROR[/red]" if result.error else "[green]OK[/green]"
            marker = " ★" if result.chunk_size == self.recommended_size else ""
            table.add_row(
                f"{result.chunk_size}{marker}",
                f"{result.mean_recall_at_5:.3f}",
                f"{result.mean_faithfulness:.3f}",
                f"{result.mean_relevance:.3f}",
                f"{result.mean_latency_ms:.1f}",
                f"{result.weighted_score:.3f}",
                status,
            )
        out.print(table)
        if self.recommended_size is not None:
            out.print(f"\n[bold green]Recommended chunk_size: {self.recommended_size}[/bold green]")


def load_sweep_sizes(config_path: Path | None = None) -> list[int]:
    """Load chunk sizes to sweep from configs/evals.yaml."""
    path = config_path or _EVALS_CONFIG_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        evals = data.get("evals") or {}
        sweep = evals.get("chunk_size_sweep") or {}
        raw_sizes = sweep.get("sizes")
        if isinstance(raw_sizes, list):
            sizes = [int(s) for s in raw_sizes if isinstance(s, int | float | str)]
            if sizes:
                return sizes
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Cannot load chunk size sweep config from %s: %s", path, exc)
    return list(_DEFAULT_SIZES)


def load_sweep_weights(config_path: Path | None = None) -> SweepWeights:
    """Load metric weights for the recommendation score."""
    path = config_path or _EVALS_CONFIG_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        evals = data.get("evals") or {}
        sweep = evals.get("chunk_size_sweep") or {}
        raw = sweep.get("weights")
        if isinstance(raw, dict):
            return SweepWeights(
                recall=float(raw.get("recall", _DEFAULT_WEIGHTS["recall"])),
                faithfulness=float(raw.get("faithfulness", _DEFAULT_WEIGHTS["faithfulness"])),
                relevance=float(raw.get("relevance", _DEFAULT_WEIGHTS["relevance"])),
                latency=float(raw.get("latency", _DEFAULT_WEIGHTS["latency"])),
            ).normalized()
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Cannot load sweep weights from %s: %s", path, exc)
    return SweepWeights(**_DEFAULT_WEIGHTS).normalized()


def collection_name_for_size(chunk_size: int, base_collection: str = "rag_documents") -> str:
    """Return an isolated Qdrant collection name for a sweep size."""
    return f"{base_collection}_cs{chunk_size}"


def chunk_cache_path(chunk_size: int, cache_dir: Path | None = None) -> Path:
    root = cache_dir or CHUNKS_DIR
    return root / str(chunk_size) / "chunks.json"


def bm25_cache_path(chunk_size: int, cache_dir: Path | None = None) -> Path:
    root = cache_dir or CHUNKS_DIR
    return root / str(chunk_size) / "bm25_index.json"


def build_chunk_size_overrides(
    chunk_size: int,
    *,
    base_collection: str = "rag_documents",
) -> dict[str, str]:
    """Env overrides for a single sweep size (chunking + isolated collection)."""
    return {
        "CHUNKING__CHUNK_SIZE": str(chunk_size),
        "QDRANT__COLLECTION": collection_name_for_size(chunk_size, base_collection),
    }


def save_chunk_cache(chunks: list[Chunk], chunk_size: int, cache_dir: Path | None = None) -> Path:
    """Persist raw chunks for *chunk_size* (embeddings stripped)."""
    path = chunk_cache_path(chunk_size, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        chunk.model_copy(update={"embedding": None, "sparse_vector": None}).model_dump()
        for chunk in chunks
    ]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d chunks to cache %s", len(chunks), path)
    return path


def load_chunk_cache(chunk_size: int, cache_dir: Path | None = None) -> list[Chunk] | None:
    """Load cached chunks for *chunk_size* or None when missing/invalid."""
    path = chunk_cache_path(chunk_size, cache_dir)
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("No chunk cache at %s: %s", path, exc)
        return None
    if not isinstance(raw, list):
        return None
    chunks: list[Chunk] = []
    for item in raw:
        if isinstance(item, dict):
            try:
                chunks.append(Chunk.model_validate(item))
            except ValueError as exc:
                logger.warning("Skipping invalid cached chunk: %s", exc)
    return chunks or None


def iter_source_files(source: Path) -> list[Path]:
    """Return supported document paths under a *source* (file or directory)."""
    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def chunk_documents_from_source(source: Path, chunk_size: int) -> list[Chunk]:
    """Chunk all documents under *source* at *chunk_size*."""
    from src.core.settings import settings
    from src.infrastructure.loaders import load_document
    from src.rag.chunking import get_chunker

    files = iter_source_files(source)
    if not files:
        raise ValueError(f"No supported documents found under {source}")

    with temporary_config(build_chunk_size_overrides(chunk_size)):
        cfg = settings.chunking
        chunker = get_chunker(
            cfg.strategy,
            use_contextual_headers=cfg.contextual_headers.enabled,
            chunk_size=chunk_size,
            overlap=cfg.overlap,
        )
        all_chunks: list[Chunk] = []
        for path in files:
            document = load_document(path)
            all_chunks.extend(chunker.chunk(document))
    return all_chunks


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Attach dense and sparse embeddings to *chunks*."""
    from src.infrastructure.embeddings import get_embedding_provider

    if not chunks:
        return []
    texts = [c.text for c in chunks]
    embedder = get_embedding_provider()
    dense_vecs, sparse_vecs = embedder.embed_both(texts)
    return [
        chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
        for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs, strict=True)
    ]


def clear_vector_index(store: VectorStoreRepository) -> None:
    """Drop the per-size Qdrant collection before a full re-index.

    Re-chunking assigns new chunk UUIDs while BM25 is fully replaced via: meth:`BM25Index.index`. Clearing Qdrant first keeps dense and lexical
    indexes aligned on repeat runs (``--force-rechunk``, cache overwrite, etc.).
    """  # noqa: E501
    drop = getattr(store, "drop_collection", None)
    if drop is None:
        return
    try:
        drop()
    except Exception as exc:
        logger.debug("Could not clear Qdrant collection before re-index: %s", exc)


def index_chunks_for_size(
    chunk_size: int,
    chunks: list[Chunk],
    *,
    cache_dir: Path | None = None,
    base_collection: str = "rag_documents",
) -> tuple[VectorStoreRepository, object, list[Chunk]]:
    """Embed and replace the per-size Qdrant collection + BM25 cache."""
    from src.infrastructure.vectordb.bm25 import BM25Index
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    embedded = embed_chunks(chunks)
    overrides = build_chunk_size_overrides(chunk_size, base_collection=base_collection)
    with temporary_config(overrides):
        store = QdrantVectorStore.from_settings()
        clear_vector_index(store)
        store.upsert(embedded)
        bm25 = BM25Index(index_path=bm25_cache_path(chunk_size, cache_dir))
        bm25.index(embedded)
        bm25.save()
    return store, bm25, embedded


def build_sweep_pipeline(
    *,
    vector_store: VectorStoreRepository,
    bm25_index: object | None = None,
) -> BenchmarkPipeline:
    """Construct a ChatPipeline for chunk-size sweep evaluation."""
    from src.rag.pipelines.chat_pipeline import ChatPipeline

    return ChatPipeline.from_settings(bm25_index=bm25_index, vector_store=vector_store)  # type: ignore[return-value]


def remap_relevant_chunks(
    pair: dict[str, object],
    indexed_ids: set[str],
    chunks_by_id: dict[str, Chunk],
) -> list[str]:
    """Resolve relevant chunk IDs for the current index.

    Uses golden IDs when present; otherwise matches chunks whose text overlaps
    the expected answer (common when chunk boundaries change between sizes).
    """
    relevant = pair.get("relevant_chunks")
    if isinstance(relevant, list):
        existing = [r for r in relevant if isinstance(r, str) and r in indexed_ids]
        if existing:
            return existing

    answer = pair.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return []

    needle = answer.strip().lower()
    matched: list[str] = []
    for chunk_id, chunk in chunks_by_id.items():
        if needle in chunk.text.lower():
            matched.append(chunk_id)
    return matched


def compute_weighted_score(
    result: ChunkSizeResult,
    weights: SweepWeights,
    *,
    latency_scores: dict[int, float],
) -> float:
    """Combine metrics into a single recommendation score."""
    w = weights.normalized()
    latency_component = latency_scores.get(result.chunk_size, 0.0)
    if result.error:
        return 0.0
    return (
        w.recall * result.mean_recall_at_5
        + w.faithfulness * result.mean_faithfulness
        + w.relevance * result.mean_relevance
        + w.latency * latency_component
    )


def recommend_size(results: list[ChunkSizeResult], weights: SweepWeights) -> int | None:
    """Return the chunk size with the highest weighted score."""
    successful = [r for r in results if not r.error]
    if not successful:
        return None
    latencies = [r.mean_latency_ms for r in successful]
    min_lat = min(latencies)
    max_lat = max(latencies)
    span = max(max_lat - min_lat, 1e-9)
    latency_scores = {
        r.chunk_size: 1.0 - ((r.mean_latency_ms - min_lat) / span) for r in successful
    }
    for result in results:
        result.weighted_score = compute_weighted_score(
            result, weights, latency_scores=latency_scores
        )
    best = max(successful, key=lambda r: r.weighted_score)
    return best.chunk_size


def build_sweep_plan(
    sizes: list[int],
    *,
    ingest_source: Path | None = None,
    cache_dir: Path | None = None,
    base_collection: str = "rag_documents",
) -> list[SweepPlanEntry]:
    """Describe planned work for each size (used by --dry-run)."""
    plan: list[SweepPlanEntry] = []
    for size in sizes:
        cache_path = chunk_cache_path(size, cache_dir)
        cached = load_chunk_cache(size, cache_dir) is not None
        if cached:
            action = "load cache + index"
        elif ingest_source is not None:
            action = "chunk source + cache + index"
        else:
            action = "missing cache (provide --ingest-source)"
        plan.append(
            SweepPlanEntry(
                chunk_size=size,
                collection=collection_name_for_size(size, base_collection),
                cache_path=cache_path,
                action=action,
            )
        )
    return plan


class ChunkSizeSweep:
    """Sweep chunk sizes and recommend the best for the current corpus."""

    def __init__(
        self,
        *,
        faithfulness: FaithfulnessMetric | None = None,
        relevance: RelevanceMetric | None = None,
        recall_k: int = 5,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.75,
        weights: SweepWeights | None = None,
    ) -> None:
        self._faith = faithfulness or FaithfulnessMetric(threshold=faithfulness_threshold)
        self._relev = relevance or RelevanceMetric(threshold=relevance_threshold)
        self._k = recall_k
        self._weights = weights or load_sweep_weights()

    async def run(
        self,
        qa_pairs: list[dict[str, object]],
        sizes: list[int],
        *,
        timestamp: str | None = None,
        ingest_source: Path | None = None,
        cache_dir: Path | None = None,
        base_collection: str = "rag_documents",
        dry_run: bool = False,
        force_rechunk: bool = False,
        pipeline_factory: Any | None = None,
    ) -> ChunkSizeSweepReport:
        ts = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        plan = build_sweep_plan(
            sizes,
            ingest_source=ingest_source,
            cache_dir=cache_dir,
            base_collection=base_collection,
        )

        if dry_run:
            return ChunkSizeSweepReport(
                timestamp=ts,
                sizes=sizes,
                results=[],
                dry_run=True,
                plan=plan,
            )

        if not has_real_qa_data(qa_pairs):
            return ChunkSizeSweepReport(
                timestamp=ts,
                sizes=sizes,
                results=[],
                skipped=True,
                skip_reason="Golden QA dataset contains only placeholders — populate via T-040.",
                plan=plan,
            )

        results: list[ChunkSizeResult] = []
        factory = pipeline_factory or build_sweep_pipeline

        for size in sizes:
            try:
                chunks = self._prepare_chunks(
                    size,
                    ingest_source=ingest_source,
                    cache_dir=cache_dir,
                    force_rechunk=force_rechunk,
                )
                store, bm25, embedded = index_chunks_for_size(
                    size,
                    chunks,
                    cache_dir=cache_dir,
                    base_collection=base_collection,
                )
                overrides = build_chunk_size_overrides(size, base_collection=base_collection)
                with temporary_config(overrides):
                    pipeline = factory(vector_store=store, bm25_index=bm25)
                    result = await self._evaluate_size(size, pipeline, qa_pairs, embedded)
            except Exception as exc:
                logger.exception("Chunk size %d sweep failed", size)
                result = ChunkSizeResult(
                    chunk_size=size,
                    total_samples=0,
                    mean_recall_at_5=0.0,
                    mean_faithfulness=0.0,
                    mean_relevance=0.0,
                    mean_latency_ms=0.0,
                    error=str(exc),
                )
            results.append(result)

        recommended = recommend_size(results, self._weights)
        return ChunkSizeSweepReport(
            timestamp=ts,
            sizes=sizes,
            results=results,
            recommended_size=recommended,
            plan=plan,
        )

    @staticmethod
    def _prepare_chunks(
        chunk_size: int,
        *,
        ingest_source: Path | None,
        cache_dir: Path | None,
        force_rechunk: bool,
    ) -> list[Chunk]:
        if not force_rechunk:
            cached = load_chunk_cache(chunk_size, cache_dir)
            if cached:
                return cached
        if ingest_source is None:
            raise ValueError(
                f"No chunk cache for size {chunk_size}; pass --ingest-source to build it"
            )
        chunks = chunk_documents_from_source(ingest_source, chunk_size)
        if not chunks:
            raise ValueError(f"Chunking produced no chunks for size {chunk_size}")
        save_chunk_cache(chunks, chunk_size, cache_dir)
        return chunks

    async def _evaluate_size(
        self,
        chunk_size: int,
        pipeline: BenchmarkPipeline,
        qa_pairs: list[dict[str, object]],
        indexed_chunks: list[Chunk],
    ) -> ChunkSizeResult:
        indexed_ids = {c.id for c in indexed_chunks}
        chunks_by_id = {c.id: c for c in indexed_chunks}
        accumulator = GenerationMetricAccumulator()

        for pair in qa_pairs:
            question = pair_str(pair.get("question"))
            expected = pair_str(pair.get("answer"))
            if not question:
                continue
            relevant_ids = remap_relevant_chunks(pair, indexed_ids, chunks_by_id)

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
                    "Pipeline failed for chunk_size=%d / %r: %s",
                    chunk_size,
                    question[:40],
                ),
            )
            accumulator.append(scores)

        means = accumulator.means()
        return ChunkSizeResult(
            chunk_size=chunk_size,
            total_samples=accumulator.total_samples,
            mean_recall_at_5=means.mean_recall_at_5,
            mean_faithfulness=means.mean_faithfulness,
            mean_relevance=means.mean_relevance,
            mean_latency_ms=means.mean_latency_ms,
        )


def run_chunk_size_sweep(
    sizes: list[int],
    qa_pairs: list[dict[str, object]],
    *,
    output_dir: Path | None = None,
    sweep: ChunkSizeSweep | None = None,
    ingest_source: Path | None = None,
    dry_run: bool = False,
) -> ChunkSizeSweepReport:
    """Synchronous entry point for scripts."""
    import asyncio

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    runner = sweep or ChunkSizeSweep()
    report = asyncio.run(
        runner.run(
            qa_pairs,
            sizes,
            timestamp=ts,
            ingest_source=ingest_source,
            dry_run=dry_run,
        )
    )
    if not report.skipped and not report.dry_run:
        out_dir = output_dir or EXPORTS_DIR
        report.save(out_dir / f"chunk_size_sweep_{ts}.json")
    return report
