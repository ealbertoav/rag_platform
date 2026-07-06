"""Compare RAG techniques side-by-side on the golden QA dataset (T-150).

Usage:
    uv run python scripts/benchmark_techniques.py
    uv run python scripts/benchmark_techniques.py \\
        --techniques baseline,multi_query,hyde,cch,reliable_rag,feedback_loop \\
        --max-samples 50

Exit code 0 when the run completes (including graceful skip on placeholder data); 1 on error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from _benchmark_utils import add_eval_args, apply_llm_config, resolve_qa_pairs


def _parse_techniques(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


async def run(args: argparse.Namespace) -> int:
    from src.core.constants import EXPORTS_DIR
    from src.evals.e2e.technique_benchmark import TechniqueBenchmark, has_real_qa_data

    qa_pairs = resolve_qa_pairs(args.qa_dataset, args.max_samples)
    if qa_pairs is None:
        return 1
    techniques = _parse_techniques(args.techniques)

    if not has_real_qa_data(qa_pairs):
        print(
            "Golden QA dataset contains only placeholders — skipping technique benchmark. "
            "Populate datasets/goldens/qa_dataset.json via T-040 first.",
            file=sys.stderr,
        )
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        from src.evals.e2e.technique_benchmark import TechniqueBenchmarkReport

        report = TechniqueBenchmarkReport(
            timestamp=ts,
            techniques=techniques,
            results=[],
            skipped=True,
            skip_reason="placeholder QA dataset",
        )
        report.print_table()
        output = EXPORTS_DIR / f"technique_benchmark_{ts}.json"
        report.save(output)
        print(f"\nSkip report saved to {output}")
        return 0

    print(f"Running technique benchmark on {len(qa_pairs)} QA pairs…")
    print(f"  Techniques: {', '.join(techniques)}")

    benchmark = TechniqueBenchmark(
        faithfulness_threshold=args.faith_threshold,
        relevance_threshold=args.relev_threshold,
    )
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report = await benchmark.run(qa_pairs, techniques, timestamp=ts)

    report.print_table()
    print(f"\n{report.summary()}")

    output = EXPORTS_DIR / f"technique_benchmark_{ts}.json"
    report.save(output)
    print(f"\nFull results saved to {output}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG techniques on the golden QA dataset")
    parser.add_argument(
        "--techniques",
        default="baseline,multi_query,hyde,cch,reliable_rag,feedback_loop",
        help="Comma-separated technique names (see configs/evals.yaml)",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Optional LLM config YAML override (applied before pipeline load)",
    )
    add_eval_args(parser)
    args = parser.parse_args()

    if args.llm_config:
        apply_llm_config(args.llm_config)

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
