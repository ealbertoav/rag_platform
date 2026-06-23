"""Generate a synthetic QA dataset from ingested chunks.

Usage:
    uv run python scripts/run_evals.py
    uv run python scripts/run_evals.py --n-pairs 5 --max-chunks 50
    uv run python scripts/run_evals.py --output datasets/synthetic/my_dataset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QA dataset")
    parser.add_argument("--n-pairs", type=int, default=3, help="QA pairs per chunk")
    parser.add_argument("--max-chunks", type=int, default=None, help="Cap on chunk count")
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for question deduplication",
    )
    parser.add_argument(
        "--output",
        default="datasets/synthetic/generated_qa.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    from src.evals.golden_dataset import SyntheticDatasetBuilder
    from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
    from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
    from src.infrastructure.vectordb.bm25 import BM25Index

    bm25 = BM25Index.load_or_create()
    if bm25.size == 0:
        print("Error: BM25 index is empty. Ingest documents first.", file=sys.stderr)
        sys.exit(1)

    chunks = bm25.chunks
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    print(f"Building QA pairs from {len(chunks)} chunks (n={args.n_pairs} per chunk)…")

    llm = LlamaCppProvider.from_settings()
    embedder = BGEM3EmbeddingProvider.from_settings()

    builder = SyntheticDatasetBuilder(
        llm=llm,
        embedder=embedder,
        n_pairs_per_chunk=args.n_pairs,
        dedup_threshold=args.dedup_threshold,
    )
    pairs = builder.generate_from_chunks(chunks)

    output = Path(args.output)
    builder.save(pairs, output)
    print(f"✓ {len(pairs)} QA pairs saved to {output}")

    # Preview the first pair for human review.
    if pairs:
        print("\nSample pair:")
        print(json.dumps(pairs[0].to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
