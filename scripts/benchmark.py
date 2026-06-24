"""End-to-end RAG benchmark: runs all QA pairs through the full pipeline.

Usage:
    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --max-samples 20 --recall-threshold 0.4
    uv run python scripts/benchmark.py --llm-config configs/llm/qwen3-14b.yaml

Exit code 0 = all metrics above thresholds; 1 = at least one metric below threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from _benchmark_utils import add_eval_args, apply_llm_config, resolve_qa_pairs


async def run(args: argparse.Namespace) -> int:
    from src.core.constants import EXPORTS_DIR
    from src.evals.e2e.rag_benchmark import RAGBenchmark
    from src.rag.pipelines.chat_pipeline import ChatPipeline

    qa_pairs = resolve_qa_pairs(args.qa_dataset, args.max_samples)
    if qa_pairs is None:
        return 1

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
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Path to a per-model LLM config YAML (e.g. configs/llm/qwen3-14b.yaml). "
        "Overrides LLM settings before the pipeline loads.",
    )
    add_eval_args(parser)
    args = parser.parse_args()

    # Apply model overrides BEFORE any src.* imports, so the settings singleton
    # picks them up on first load.
    if args.llm_config:
        apply_llm_config(args.llm_config)

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
