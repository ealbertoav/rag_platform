"""Infrastructure latency baseline and regression benchmark (T-172).

Usage:
    uv run python scripts/benchmark_infra.py
    uv run python scripts/benchmark_infra.py --compare
    uv run python scripts/benchmark_infra.py --save-baseline
    uv run python scripts/benchmark_infra.py --scenarios bm25_100k,graph_retrieval

Scenario 5 (concurrent feedback) lives in tests/benchmarks/test_feedback_concurrency.py.

Exit code 0 when the run completes; 1 on error; 2 when --compare detects p95 regression
or a baselined scenario failed/skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from _benchmark_utils import apply_llm_config


def _parse_scenarios(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


async def run(args: argparse.Namespace) -> int:
    from src.core.constants import EXPORTS_DIR
    from src.evals.e2e.infra_benchmark import (
        InfraBenchmark,
        build_default_graph_retriever,
        build_default_pipeline,
        compare_to_baseline,
        load_infra_baseline,
        load_infra_thresholds,
        save_infra_baseline,
    )

    thresholds = load_infra_thresholds()
    scenarios = _parse_scenarios(args.scenarios)

    async def pipeline_factory():
        return await build_default_pipeline()

    benchmark = InfraBenchmark(
        thresholds=thresholds,
        pipeline_factory=pipeline_factory if _needs_pipeline(scenarios) else None,
        graph_retriever_factory=build_default_graph_retriever
        if "graph_retrieval" in scenarios
        else None,
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    print(f"Running infra benchmark scenarios: {', '.join(scenarios)}")
    report = await benchmark.run(scenarios, timestamp=ts)

    report.print_table()
    print(f"\n{report.summary()}")

    output = EXPORTS_DIR / f"infra_benchmark_{ts}.json"
    report.save(output)
    print(f"\nFull results saved to {output}")

    if args.save_baseline:
        baseline_path = save_infra_baseline(report, thresholds.baseline_path)
        print(f"Baseline updated at {baseline_path}")

    if args.compare:
        baseline = load_infra_baseline(thresholds.baseline_path)
        comparison = compare_to_baseline(
            report.scenario_map(),
            baseline,
            regression_p95_pct=thresholds.regression_p95_pct,
        )
        if comparison.failures:
            print("\nBaseline scenario failures:")
            for failure in comparison.failures:
                print(f"  ✗ {failure.message()}")
        if comparison.regressions:
            print(f"\nRegression warnings (> {thresholds.regression_p95_pct:.0f}% p95 increase):")
            for warning in comparison.regressions:
                print(f"  ⚠ {warning.message()}")
        if comparison.has_issues:
            return 2
        print("\nNo p95 regressions vs committed baseline.")

    return 0


def _needs_pipeline(scenarios: list[str]) -> bool:
    return bool({"streaming_chat", "concurrent_chats"} & set(scenarios))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure infrastructure latency baselines (streaming, concurrency, BM25, graph)"
    )
    parser.add_argument(
        "--scenarios",
        default="streaming_chat,concurrent_chats,bm25_100k,graph_retrieval",
        help="Comma-separated scenario names",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results to data/exports/infra_baseline.json (>10%% p95 increase warns)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write current results to the committed baseline file",
    )
    parser.add_argument(
        "--llm-config",
        default=None,
        help="Optional LLM config YAML override (applied before pipeline load)",
    )
    args = parser.parse_args()

    if args.llm_config:
        apply_llm_config(args.llm_config)

    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
