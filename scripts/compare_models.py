"""Compare multiple LLM configs against the same QA dataset.

Usage:
    uv run python scripts/compare_models.py \\
        --configs configs/llm/qwen3-30b.yaml configs/llm/qwen3-14b.yaml \\
        --max-samples 50

Runs the full RAG benchmark for each config and prints a Rich comparison table.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
from datetime import UTC, datetime

from _benchmark_utils import add_eval_args, apply_llm_config, resolve_qa_pairs
from rich.console import Console
from rich.table import Table


@dataclasses.dataclass
class ModelResult:
    name: str
    recall_at_5: float
    faithfulness: float
    relevance: float
    latency_s: float
    passed: bool
    error: str = ""


async def _run_one(
    config_path: str,
    qa_pairs: list[dict[str, object]],
    recall_threshold: float,
    faith_threshold: float,
    relev_threshold: float,
) -> ModelResult:
    import importlib
    import time

    # Reload settings so env var changes take effect for this config.
    import src.core.settings as _settings_mod

    label = apply_llm_config(config_path)
    importlib.reload(_settings_mod)

    from src.evals.e2e.rag_benchmark import RAGBenchmark
    from src.rag.pipelines.chat_pipeline import ChatPipeline

    try:
        t0 = time.monotonic()
        pipeline = ChatPipeline.from_settings()
        benchmark = RAGBenchmark(
            recall_threshold=recall_threshold,
            faithfulness_threshold=faith_threshold,
            relevance_threshold=relev_threshold,
        )
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        report = await benchmark.run(pipeline, qa_pairs, timestamp=ts)
        elapsed = time.monotonic() - t0

        return ModelResult(
            name=label,
            recall_at_5=report.mean_recall_at_5,
            faithfulness=report.mean_faithfulness,
            relevance=report.mean_relevance,
            latency_s=round(elapsed, 1),
            passed=report.passed,
        )
    except Exception as exc:
        return ModelResult(
            name=label,
            recall_at_5=0.0,
            faithfulness=0.0,
            relevance=0.0,
            latency_s=0.0,
            passed=False,
            error=str(exc),
        )


def _print_table(results: list[ModelResult]) -> None:
    table = Table(title="Model Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Model", style="white")
    table.add_column("Recall@5", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "[green]PASS ✓[/green]" if r.passed else "[red]FAIL ✗[/red]"
        if r.error:
            status = "[red]ERROR[/red]"
        table.add_row(
            r.name,
            f"{r.recall_at_5:.3f}",
            f"{r.faithfulness:.3f}",
            f"{r.relevance:.3f}",
            f"{r.latency_s:.1f}",
            status,
        )

    Console().print(table)
    if any(r.error for r in results):
        for r in results:
            if r.error:
                print(f"  {r.name} error: {r.error}", file=sys.stderr)


async def run(args: argparse.Namespace) -> int:
    qa_pairs = resolve_qa_pairs(args.qa_dataset, args.max_samples)
    if qa_pairs is None:
        return 1

    print(f"Comparing {len(args.configs)} model(s) on {len(qa_pairs)} QA pairs…\n")

    results: list[ModelResult] = []
    for cfg_path in args.configs:
        print(f"  Running: {cfg_path}")
        result = await _run_one(
            cfg_path,
            qa_pairs,
            args.recall_threshold,
            args.faith_threshold,
            args.relev_threshold,
        )
        results.append(result)

    print()
    _print_table(results)
    return 0 if all(r.passed for r in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple LLM configs on the same dataset")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="YAML",
        help="One or more LLM config YAMLs (e.g. configs/llm/qwen3-30b.yaml)",
    )
    add_eval_args(parser)
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
