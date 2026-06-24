"""Compare multiple embedding providers on the same golden QA dataset.

Each provider is evaluated against its own temporary Qdrant collection built by
re-embedding all BM25 chunks with that provider. Dense retrieval metrics are
reported (no BM25/hybrid fusion), so scores reflect embedding quality alone.

Usage:
    # Self-hosted only (no API key required)
    uv run python scripts/compare_embedding_providers.py --providers bge_m3

    # Compare local vs. API providers
    uv run python scripts/compare_embedding_providers.py \\
        --providers bge_m3 openai voyage \\
        --max-samples 50

    # Save results to a JSON file
    uv run python scripts/compare_embedding_providers.py \\
        --providers bge_m3 openai \\
        --output data/exports/embedding_comparison.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _benchmark_utils import add_eval_args, resolve_qa_pairs  # noqa: E402

from src.domain.entities.chunk import Chunk  # noqa: E402
from src.infrastructure.vectordb.qdrant import QdrantVectorStore  # noqa: E402
from src.rag.retrieval.dense_retriever import DenseRetriever  # noqa: E402

# ── Cost estimates (USD per 1K tokens) ────────────────────────────────────────
# last-updated: 2025-06-24. Check provider pricing pages for current rates:
#   OpenAI:  https://openai.com/api/pricing/
#   Voyage:  https://docs.voyageai.com/docs/pricing
#   Cohere:  https://cohere.com/pricing
#   Gemini:  https://ai.google.dev/gemini-api/docs/pricing

_INDEX_BATCH_SIZE = 32

_COST_PER_1K: dict[str, float] = {
    "bge_m3": 0.0,
    "nomic": 0.0,
    "qwen_embedding": 0.0,
    "openai": 0.13,    # text-embedding-3-large
    "voyage": 0.12,    # voyage-large-2
    "cohere": 0.10,    # embed-english-v3.0
    "gemini": 0.025,   # text-embedding-004
}


@dataclasses.dataclass
class ProviderResult:
    name: str
    recall_at_5: float
    ndcg_at_5: float
    latency_ms: float
    cost_per_1k: float
    n_queries: int
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and self.recall_at_5 > 0


def _compute_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def _compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    import math

    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, rid in enumerate(retrieved_ids[:k])
        if rid in relevant_set
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def _index_chunks_for_provider(
    provider: str,
    chunks: list[Chunk],
    settings: Settings,
) -> tuple[DenseRetriever, QdrantVectorStore]:
    """Embed *chunks* with *provider* and upsert into a temporary Qdrant collection."""
    from src.core.constants import API_EMBEDDING_PROVIDERS
    from src.infrastructure.embeddings import (
        create_embedding_provider,
        embedding_model_identifier,
        provider_dense_dim,
    )
    from src.infrastructure.embeddings.batch_reindex import (
        API_BATCH_SIZE,
        embed_and_upsert_batch,
        maybe_sleep_between_api_batches,
    )

    embedder = create_embedding_provider(provider, settings)
    dense_dim = provider_dense_dim(provider, settings)
    model_id = embedding_model_identifier(provider, settings)
    temp_collection = f"{settings.qdrant.collection}__compare__{provider}"

    vector_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection=temp_collection,
        api_key=settings.qdrant.api_key,
        dense_dim=dense_dim,
        embedding_model_name=model_id,
    )

    try:
        with contextlib.suppress(Exception):
            vector_store.drop_collection()

        is_api = provider in API_EMBEDDING_PROVIDERS
        batch_size = API_BATCH_SIZE if is_api else _INDEX_BATCH_SIZE

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embed_and_upsert_batch(embedder, vector_store, batch)
            maybe_sleep_between_api_batches(
                provider,
                batch_start=i,
                batch_size=batch_size,
                total_chunks=len(chunks),
            )

        retriever = DenseRetriever(embedder=embedder, vector_store=vector_store)
        return retriever, vector_store
    except Exception:
        with contextlib.suppress(Exception):
            vector_store.drop_collection()
        raise


async def _run_provider(
    provider: str,
    qa_pairs: list[dict[str, object]],
    chunks: list[Chunk],
    settings: Settings,
    k: int = 5,
) -> ProviderResult:
    try:
        retriever, vector_store = _index_chunks_for_provider(provider, chunks, settings)
    except Exception as exc:
        return ProviderResult(
            name=provider, recall_at_5=0.0, ndcg_at_5=0.0,
            latency_ms=0.0, cost_per_1k=_COST_PER_1K.get(provider, 0.0),
            n_queries=0, error=str(exc),
        )

    from src.core.exceptions import RetrievalError
    from src.domain.entities.query import Query

    recalls, ndcgs, latencies = [], [], []

    try:
        for pair in qa_pairs:
            question = pair.get("question")
            if not isinstance(question, str) or not question:
                continue
            relevant = list(pair.get("relevant_chunks", []))  # type: ignore[arg-type]

            t0 = time.monotonic()
            try:
                query = Query(text=question)
                results = retriever.retrieve(query, top_k=k)
                retrieved_ids = [r[0].id for r in results]
            except RetrievalError:
                retrieved_ids = []
            latencies.append((time.monotonic() - t0) * 1000)

            recalls.append(_compute_recall_at_k(retrieved_ids, relevant, k))
            ndcgs.append(_compute_ndcg_at_k(retrieved_ids, relevant, k))
    finally:
        with contextlib.suppress(Exception):
            vector_store.drop_collection()

    n = len(recalls)
    return ProviderResult(
        name=provider,
        recall_at_5=sum(recalls) / n if n else 0.0,
        ndcg_at_5=sum(ndcgs) / n if n else 0.0,
        latency_ms=sum(latencies) / n if n else 0.0,
        cost_per_1k=_COST_PER_1K.get(provider, 0.0),
        n_queries=n,
    )


def _print_table(results: list[ProviderResult], k: int) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(
        title=f"Embedding Provider Comparison (k={k})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Provider", style="white")
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column(f"NDCG@{k}", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Cost/1K tok", justify="right")
    table.add_column("Queries", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        if r.error:
            status = "[red]ERROR[/red]"
            row = [r.name, "–", "–", "–", f"${r.cost_per_1k:.3f}", "–", status]
        else:
            status = "[green]OK ✓[/green]" if r.passed else "[yellow]SKIP[/yellow]"
            cost = f"${r.cost_per_1k:.3f}" if r.cost_per_1k > 0 else "[dim]$0.000[/dim]"
            row = [
                r.name,
                f"{r.recall_at_5:.3f}",
                f"{r.ndcg_at_5:.3f}",
                f"{r.latency_ms:.0f}",
                cost,
                str(r.n_queries),
                status,
            ]
        table.add_row(*row)

    Console().print(table)
    for r in results:
        if r.error:
            print(f"  {r.name} error: {r.error}", file=sys.stderr)


def _save_results(results: list[ProviderResult], output: str) -> None:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = Path(output) if output else Path(f"data/exports/embedding_comparison_{ts}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": ts,
        "results": [dataclasses.asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")


async def run(args: argparse.Namespace) -> int:
    from src.core.settings import settings
    from src.infrastructure.vectordb.bm25 import BM25Index

    qa_pairs = resolve_qa_pairs(args.qa_dataset, args.max_samples)
    if qa_pairs is None:
        return 1

    bm25 = BM25Index.load_or_create()
    chunks = bm25.chunks
    if not chunks:
        print("BM25 index is empty — ingest documents first.", file=sys.stderr)
        return 1

    print(
        f"Comparing {len(args.providers)} provider(s) on {len(qa_pairs)} QA pairs "
        f"({len(chunks)} indexed chunks per provider)…\n"
    )

    results: list[ProviderResult] = []
    for provider in args.providers:
        print(f"  Running: {provider}")
        result = await _run_provider(
            provider, qa_pairs, chunks, settings, k=args.top_k
        )
        results.append(result)
        if result.error:
            print(f"    → skipped: {result.error}", file=sys.stderr)

    print()
    _print_table(results, k=args.top_k)

    # Save when the user explicitly named an output file (even if all providers
    # errored — they asked for the file, so write it).  For auto-generated paths,
    # only save when at least one provider produced valid results.
    if args.output or any(not r.error for r in results):
        _save_results(results, args.output or "")

    return 0 if all(not r.error for r in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare embedding providers on the same golden QA dataset"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        required=True,
        metavar="PROVIDER",
        help="One or more provider names (bge_m3 nomic openai voyage cohere gemini)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-K (default: 5)")
    parser.add_argument(
        "--output",
        default="",
        metavar="PATH",
        help="Save results JSON to this path "
        "(default: data/exports/embedding_comparison_{ts}.json)",
    )
    add_eval_args(parser)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
