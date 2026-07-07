"""Generate a synthetic QA dataset from ingested chunks.

Usage:
    make ingest SOURCE=data/raw/ # prerequisite — BM25 must be populated
    make evals

    uv run python scripts/run_evals.py
    uv run python scripts/run_evals.py --n-pairs 5 --max-chunks 50
    uv run python scripts/run_evals.py --output datasets/goldens/qa_dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.constants import DATASETS_DIR
from src.evals.golden_dataset import (
    MIN_QA_PAIRS,
    SyntheticDatasetBuilder,
    generate_until_min_pairs,
    qa_pairs_to_retrieval_rows,
    resolve_max_chunks,
    resolve_retrieval_output_path,
    save_retrieval_dataset,
)


def _default_output() -> Path:
    return DATASETS_DIR / "goldens" / "qa_dataset.json"


def _default_retrieval_output() -> Path:
    return DATASETS_DIR / "goldens" / "retrieval_dataset.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QA dataset")
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=3,
        help="QA pairs per chunk (increase if dedup removes too many)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap on chunk count (default: enough to reach --min-pairs)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=MIN_QA_PAIRS,
        help=f"Minimum QA pairs required before writing goldens (default: {MIN_QA_PAIRS})",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for question deduplication",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="QA golden output path (default: datasets/goldens/qa_dataset.json)",
    )
    parser.add_argument(
        "--retrieval-output",
        default=None,
        help=(
            "Retrieval golden output path "
            "(default: datasets/goldens/retrieval_dataset.json when --output is default; "
            "otherwise retrieval_dataset.json alongside custom --output)"
        ),
    )
    parser.add_argument(
        "--no-sync-retrieval",
        action="store_true",
        help="Skip writing datasets/goldens/retrieval_dataset.json",
    )
    args = parser.parse_args()

    from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
    from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
    from src.infrastructure.vectordb.bm25 import BM25Index

    bm25 = BM25Index.load_or_create()
    if bm25.size == 0:
        print("Error: BM25 index is empty. Run `make ingest SOURCE=...` first.", file=sys.stderr)
        sys.exit(1)

    chunks = bm25.chunks
    initial_limit = resolve_max_chunks(
        len(chunks),
        min_pairs=args.min_pairs,
        n_pairs_per_chunk=args.n_pairs,
        max_chunks=args.max_chunks,
        dedup_threshold=args.dedup_threshold,
    )

    print(
        f"Building QA pairs from {initial_limit} chunks "
        f"(n={args.n_pairs} per chunk, min={args.min_pairs})…"
    )

    llm = LlamaCppProvider.from_settings()
    embedder = BGEM3EmbeddingProvider.from_settings()

    builder = SyntheticDatasetBuilder(
        llm=llm,
        embedder=embedder,
        n_pairs_per_chunk=args.n_pairs,
        dedup_threshold=args.dedup_threshold,
    )
    pairs, chunk_limit = generate_until_min_pairs(
        builder,
        chunks,
        min_pairs=args.min_pairs,
        n_pairs_per_chunk=args.n_pairs,
        dedup_threshold=args.dedup_threshold,
        max_chunks=args.max_chunks,
    )

    if len(pairs) < args.min_pairs:
        print(
            f"Error: generated {len(pairs)} pairs but --min-pairs={args.min_pairs}. "
            "Increase --n-pairs, --max-chunks, or lower --dedup-threshold.",
            file=sys.stderr,
        )
        sys.exit(1)

    output = Path(args.output) if args.output else _default_output()
    builder.save(pairs, output)
    print(f"✓ {len(pairs)} QA pairs saved to {output}")

    if not args.no_sync_retrieval:
        retrieval_path = resolve_retrieval_output_path(
            output,
            retrieval_output=Path(args.retrieval_output) if args.retrieval_output else None,
            qa_golden_path=_default_output(),
            retrieval_golden_path=_default_retrieval_output(),
        )
        save_retrieval_dataset(qa_pairs_to_retrieval_rows(pairs), retrieval_path)
        print(f"✓ Retrieval golden synced to {retrieval_path}")

    if pairs:
        print("\nSample pair:")
        print(json.dumps(pairs[0].to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
