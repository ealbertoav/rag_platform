"""End-to-end RAG benchmark: runs all QA pairs through the full pipeline.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --max-samples 20 --recall-threshold 0.4

Exit code 0 = all metrics above thresholds; 1 = at least one metric below threshold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def _load_qa(path: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("question")]
        return []
    except (OSError, ValueError):
        return []


async def run(args: argparse.Namespace) -> int:
    from src.core.constants import DATASETS_DIR, EXPORTS_DIR
    from src.evals.e2e.rag_benchmark import RAGBenchmark
    from src.rag.pipelines.chat_pipeline import ChatPipeline

    qa_path = Path(args.qa_dataset)
    if not qa_path.is_absolute():
        qa_path = DATASETS_DIR / "goldens" / args.qa_dataset

    qa_pairs = _load_qa(qa_path)
    if not qa_pairs:
        print(f"Error: no QA pairs found in {qa_path}", file=sys.stderr)
        return 1

    if args.max_samples:
        qa_pairs = qa_pairs[: args.max_samples]

    print(f"Running benchmark on {len(qa_pairs)} QA pairs…")

    pipeline = ChatPipeline.from_settings()
    benchmark = RAGBenchmark(
        recall_threshold=args.recall_threshold,
        faithfulness_threshold=args.faith_threshold,
        relevance_threshold=args.relev_threshold,
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report = await benchmark.run(pipeline, qa_pairs, timestamp=ts)

    print(report.summary())

    output = EXPORTS_DIR / f"benchmark_{ts}.json"
    report.save(output)
    print(f"\nFull results saved to {output}")

    return 0 if report.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG benchmark")
    parser.add_argument("--qa-dataset", default="qa_dataset.json",
                        help="Filename in datasets/goldens/ or absolute path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--recall-threshold", type=float, default=0.5)
    parser.add_argument("--faith-threshold", type=float, default=0.8)
    parser.add_argument("--relev-threshold", type=float, default=0.75)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
