"""Sweep chunk sizes and recommend the best for the current corpus (T-151).

Usage:
    uv run python scripts/benchmark_chunk_sizes.py --dry-run
    uv run python scripts/benchmark_chunk_sizes.py \\
        --ingest-source data/raw/ \\
        --max-samples 20

Exit code 0 when the run completes (including graceful skip / dry-run); 1 on error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from _benchmark_utils import add_eval_args, apply_llm_config, resolve_qa_pairs


def _parse_sizes(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    sizes = [int(s.strip()) for s in raw.split(",") if s.strip()]
    return sizes or None


async def run(args: argparse.Namespace) -> int:
    from src.core.constants import EXPORTS_DIR
    from src.evals.e2e.chunk_size_sweep import (
        ChunkSizeSweep,
        ChunkSizeSweepReport,
        load_sweep_sizes,
    )
    from src.evals.e2e.technique_benchmark import has_real_qa_data

    sizes = _parse_sizes(args.sizes) or load_sweep_sizes()
    ingest_source = Path(args.ingest_source) if args.ingest_source else None
    if ingest_source is not None and not ingest_source.exists():
        print(f"Error: ingest source {ingest_source} does not exist", file=sys.stderr)
        return 1

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    benchmark = ChunkSizeSweep(
        faithfulness_threshold=args.faith_threshold,
        relevance_threshold=args.relev_threshold,
    )

    if args.dry_run:
        report = await benchmark.run(
            [],
            sizes,
            timestamp=ts,
            ingest_source=ingest_source,
            dry_run=True,
            force_rechunk=args.force_rechunk,
        )
        report.print_table()
        print(f"\n{report.summary()}")
        output = EXPORTS_DIR / f"chunk_size_sweep_{ts}.json"
        report.save(output)
        print(f"\nDry-run plan saved to {output}")
        return 0

    qa_pairs = resolve_qa_pairs(args.qa_dataset, args.max_samples)
    if qa_pairs is None:
        return 1

    if not has_real_qa_data(qa_pairs):
        print(
            "Golden QA dataset contains only placeholders — skipping chunk size sweep. "
            "Populate datasets/goldens/qa_dataset.json via T-040 first.",
            file=sys.stderr,
        )
        report = ChunkSizeSweepReport(
            timestamp=ts,
            sizes=sizes,
            results=[],
            skipped=True,
            skip_reason="placeholder QA dataset",
        )
        report.print_table()
        output = EXPORTS_DIR / f"chunk_size_sweep_{ts}.json"
        report.save(output)
        print(f"\nSkip report saved to {output}")
        return 0

    print(f"Running chunk size sweep on {len(qa_pairs)} QA pairs…")
    print(f"  Sizes: {', '.join(str(s) for s in sizes)}")
    if ingest_source is not None:
        print(f"  Ingest source: {ingest_source}")

    report = await benchmark.run(
        qa_pairs,
        sizes,
        timestamp=ts,
        ingest_source=ingest_source,
        force_rechunk=args.force_rechunk,
    )

    report.print_table()
    print(f"\n{report.summary()}")

    output = EXPORTS_DIR / f"chunk_size_sweep_{ts}.json"
    report.save(output)
    print(f"\nFull results saved to {output}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep chunk sizes and recommend the best for the current corpus"
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated chunk sizes (default: configs/evals.yaml chunk_size_sweep.sizes)",
    )
    parser.add_argument(
        "--ingest-source",
        default=None,
        help="File or directory to chunk/index per size (required when cache is missing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned sweep steps without executing",
    )
    parser.add_argument(
        "--force-rechunk",
        action="store_true",
        help="Ignore cached chunks and re-chunk from --ingest-source",
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
