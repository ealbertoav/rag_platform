"""Shared utilities for benchmark and compare_models scripts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_qa(path: Path) -> list[dict[str, object]]:
    """Load QA pairs from a *path*, filtering out any non-question entries."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("question")]
        return []
    except (OSError, ValueError):
        return []


def apply_llm_config(config_path: str) -> str:
    """Set LLM env vars from a per-model YAML and return the model label.

    Must be called BEFORE any "src.*" imports, so the settings singleton
    picks up the overrides on the first load.
    """
    import yaml  # type: ignore[import-untyped]

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    llm_cfg: dict[str, Any] = cfg.get("llm") or {}
    for k, v in llm_cfg.items():
        os.environ[f"LLM__{k.upper()}"] = str(v)
    return str(llm_cfg.get("model_path", Path(config_path).stem))


def resolve_qa_pairs(
    qa_dataset: str,
    max_samples: int | None,
    *,
    filter_placeholders: bool = True,
) -> list[dict[str, object]] | None:
    """Resolve, load, and optionally cap QA pairs from the standard datasets location.

    When *filter_placeholders* is True (default), placeholder rows are removed
    before applying *max_samples*, so the cap applies to real golden rows only.

    Returns "None" (and prints an error) when the file has no loadable pairs.
    """
    import sys

    from src.core.constants import DATASETS_DIR

    qa_path = Path(qa_dataset)
    if not qa_path.is_absolute():
        qa_path = DATASETS_DIR / "goldens" / qa_dataset

    pairs = load_qa(qa_path)
    if not pairs:
        print(f"Error: no QA pairs found in {qa_path}", file=sys.stderr)
        return None

    if filter_placeholders:
        from src.evals.e2e.technique_benchmark import prepare_qa_pairs

        pairs = prepare_qa_pairs(pairs, max_samples)
    elif max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]

    return pairs


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add the common QA-dataset and threshold arguments to *parser*."""
    parser.add_argument(
        "--qa-dataset",
        default="qa_dataset.json",
        help="Filename in datasets/goldens/ or absolute path",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--recall-threshold", type=float, default=0.5)
    parser.add_argument("--faith-threshold", type=float, default=0.8)
    parser.add_argument("--relev-threshold", type=float, default=0.75)
    parser.add_argument("--ctx-threshold", type=float, default=0.7)
    parser.add_argument("--halluc-threshold", type=float, default=0.1)
