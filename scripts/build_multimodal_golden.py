"""Generate a multimodal (table/figure) QA golden dataset from ingested chunks (T-280).

Restricted to chunks whose resolved modality is table or figure — requires
`parsing.table_chunks.enabled` / `parsing.figure_chunks.enabled` and a prior
`make ingest` with those chunk types present in the BM25 index.

Usage:
    make ingest SOURCE=data/raw/  # prerequisite — BM25 must hold table/figure chunks
    make multimodal-golden

    uv run python scripts/build_multimodal_golden.py
    uv run python scripts/build_multimodal_golden.py --n-pairs 5
    uv run python scripts/build_multimodal_golden.py --output datasets/goldens/multimodal.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.constants import DATASETS_DIR
from src.evals.golden_dataset import SyntheticDatasetBuilder
from src.evals.multimodal_golden_dataset import (
    build_multimodal_golden,
    filter_multimodal_chunks,
    save_jsonl,
)


def _default_output() -> Path:
    return DATASETS_DIR / "goldens" / "multimodal_qa_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a multimodal (table/figure) QA golden dataset"
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=3,
        help="QA pairs per table/figure chunk",
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
        help="Output path (default: datasets/goldens/multimodal_qa_dataset.jsonl)",
    )
    args = parser.parse_args()

    from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
    from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
    from src.infrastructure.vectordb.bm25 import BM25Index

    bm25 = BM25Index.load_or_create()
    if bm25.size == 0:
        print("Error: BM25 index is empty. Run `make ingest SOURCE=...` first.", file=sys.stderr)
        sys.exit(1)

    multimodal_chunks = filter_multimodal_chunks(bm25.iter_chunks())
    if not multimodal_chunks:
        print(
            "Error: no table/figure chunks found in the BM25 index. Enable "
            "`parsing.table_chunks`/`parsing.figure_chunks` and re-ingest first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Building multimodal QA pairs from {len(multimodal_chunks)} table/figure chunks "
        f"(n={args.n_pairs} per chunk)…"
    )

    llm = LlamaCppProvider.from_settings()
    embedder = BGEM3EmbeddingProvider.from_settings()
    builder = SyntheticDatasetBuilder(
        llm=llm,
        embedder=embedder,
        n_pairs_per_chunk=args.n_pairs,
        dedup_threshold=args.dedup_threshold,
    )

    pairs = build_multimodal_golden(builder, multimodal_chunks)
    if not pairs:
        print(
            "Error: no multimodal QA pairs generated (LLM failures or dedup removed everything).",
            file=sys.stderr,
        )
        sys.exit(1)

    output = Path(args.output) if args.output else _default_output()
    save_jsonl(pairs, output)
    print(f"✓ {len(pairs)} multimodal QA pairs saved to {output}")

    print("\nSample pair:")
    print(json.dumps(pairs[0].to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
