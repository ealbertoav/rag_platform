"""Benchmark table/figure retrieval recall against the multimodal golden (T-281).

Runs the live retrieval pipeline for every question in the multimodal QA
golden (T-280) and reports Recall@K separately for table- and figure-modality
questions, so regressions in structured-content retrieval are visible even
when overall Recall@K (T-041) looks healthy.

Usage:
    make ingest SOURCE=data/raw/   # prerequisite: table/figure chunks indexed
    make multimodal-golden          # prerequisite: multimodal_qa_dataset.jsonl

    uv run python scripts/benchmark_modality_recall.py
    uv run python scripts/benchmark_modality_recall.py --k 1 3 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.constants import DATASETS_DIR
from src.evals.multimodal_golden_dataset import load_jsonl
from src.evals.retrieval.modality_evaluator import ModalityRetrievalEvaluator
from src.evals.retrieval.modality_recall import load_modality_samples


def _default_dataset() -> Path:
    return DATASETS_DIR / "goldens" / "multimodal_qa_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark table/figure retrieval recall (T-281)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Multimodal QA golden path (default: datasets/goldens/multimodal_qa_dataset.jsonl)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="K values to report (default: 1 3 5 10)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    pairs = load_jsonl(dataset_path)
    if not pairs:
        print(
            f"Error: no multimodal QA pairs found at {dataset_path}. "
            "Run `make multimodal-golden` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    samples = load_modality_samples(pairs)
    if not any(s.relevant_ids for s in samples):
        print(
            "Error: no multimodal QA pairs have relevant_chunks to score.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.domain.entities.query import Query
    from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline

    print(f"Running retrieval for {len(samples)} table/figure questions…")
    pipeline = RetrievalPipeline.from_settings()
    for sample, pair in zip(samples, pairs, strict=True):
        question = pair.get("question")
        if not isinstance(question, str) or not question:
            continue
        result = pipeline.retrieve_sync(Query(text=question))
        sample.retrieved_ids = [chunk.id for chunk in result.chunks]

    evaluator = ModalityRetrievalEvaluator(k_values=args.k)
    metrics = evaluator.evaluate(samples)
    evaluator.print_table(metrics)


if __name__ == "__main__":
    main()
