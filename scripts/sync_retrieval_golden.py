"""Sync retrieval_dataset.json from qa_dataset.json without LLM regeneration.

Usage:
    make sync-retrieval-goldens

    uv run python scripts/sync_retrieval_golden.py
    uv run python scripts/sync_retrieval_golden.py --qa datasets/goldens/qa_dataset.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.constants import DATASETS_DIR
from src.evals.golden_dataset import sync_retrieval_from_qa


def _default_qa_path() -> Path:
    return DATASETS_DIR / "goldens" / "qa_dataset.json"


def _default_retrieval_path() -> Path:
    return DATASETS_DIR / "goldens" / "retrieval_dataset.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync retrieval golden from evaluable QA rows",
    )
    parser.add_argument(
        "--qa",
        default=None,
        help="QA golden input path (default: datasets/goldens/qa_dataset.json)",
    )
    parser.add_argument(
        "--retrieval-output",
        default=None,
        help="Retrieval golden output path (default: datasets/goldens/retrieval_dataset.json)",
    )
    args = parser.parse_args()

    qa_path = Path(args.qa) if args.qa else _default_qa_path()
    retrieval_path = (
        Path(args.retrieval_output) if args.retrieval_output else _default_retrieval_path()
    )
    count = sync_retrieval_from_qa(qa_path, retrieval_path)
    print(f"✓ {count} retrieval rows synced to {retrieval_path}")


if __name__ == "__main__":
    main()
