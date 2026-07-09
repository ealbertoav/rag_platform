"""CI retrieval regression gate (T-152).

Validates committed golden datasets meet minimum sample counts and per-row
Recall@5 floors before merge. Skips gracefully when only placeholder data exists.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.core.constants import DATASETS_DIR
from src.evals.golden_dataset import (
    MIN_QA_PAIRS,
    count_real_qa_pairs,
    is_placeholder_retrieval_row,
    load_qa_dicts,
    retrieval_rows_match_qa,
)
from src.evals.retrieval.recall_at_k import oracle_recall_at_k

_DEFAULT_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"
_DEFAULT_RETRIEVAL_PATH = DATASETS_DIR / "goldens" / "retrieval_dataset.json"
_DEFAULT_BASELINE_PATH = DATASETS_DIR / "goldens" / "retrieval_baseline.json"


class GateStatus(StrEnum):
    SKIPPED = "skipped"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class RegressionGateResult:
    status: GateStatus
    message: str


def load_regression_baseline(path: Path | None = None) -> dict[str, object]:
    """Load committed regression thresholds; returns {} when missing or invalid."""
    baseline_path = path or _DEFAULT_BASELINE_PATH
    try:
        raw: object = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_real_retrieval_rows(path: Path | None = None) -> list[dict[str, object]]:
    """Return non-placeholder retrieval rows from the golden dataset file."""
    retrieval_path = path or _DEFAULT_RETRIEVAL_PATH
    if not retrieval_path.exists():
        return []
    try:
        raw: object = json.loads(retrieval_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict) and not is_placeholder_retrieval_row(row)]


def check_regression_gate(
    *,
    qa_path: Path | None = None,
    retrieval_path: Path | None = None,
    baseline_path: Path | None = None,
) -> RegressionGateResult:
    """Evaluate the retrieval regression gate without exiting the process."""
    qa = qa_path or _DEFAULT_QA_PATH
    retrieval = retrieval_path or _DEFAULT_RETRIEVAL_PATH

    if not retrieval.exists():
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="No retrieval dataset — skipping regression gate.",
        )

    real = load_real_retrieval_rows(retrieval)
    if not real:
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="Only placeholder data — skipping regression gate.",
        )

    baseline = load_regression_baseline(baseline_path)
    min_samples = baseline_int(baseline, "min_samples", MIN_QA_PAIRS)
    min_recall = baseline_float(baseline, "min_recall_at_5", 0.5)

    qa_count = count_real_qa_pairs(qa)
    if qa_count < min_samples or len(real) < min_samples:
        return RegressionGateResult(
            status=GateStatus.FAILED,
            message=(
                f"Regression gate FAILED: need >= {min_samples} real samples "
                + f"(qa={qa_count}, retrieval={len(real)})."
            ),
        )

    for row in real:
        raw_ids = row.get("relevant_chunk_ids", [])
        relevant = [r for r in (raw_ids if isinstance(raw_ids, list) else []) if isinstance(r, str)]
        if not relevant:
            row_id = str(row.get("id", "<unknown>"))
            return RegressionGateResult(
                status=GateStatus.FAILED,
                message=f"Regression gate FAILED: row {row_id} has no relevant_chunk_ids.",
            )

    qa_pairs = load_qa_dicts(qa)
    if not retrieval_rows_match_qa(qa_pairs, real):
        return RegressionGateResult(
            status=GateStatus.FAILED,
            message=(
                "Regression gate FAILED: retrieval_dataset.json is out of sync with "
                + "qa_dataset.json — run `make sync-retrieval-goldens` or `make evals`."
            ),
        )

    for row in real:
        raw_ids = row.get("relevant_chunk_ids", [])
        relevant = [r for r in (raw_ids if isinstance(raw_ids, list) else []) if isinstance(r, str)]
        score = oracle_recall_at_k(relevant, k=5)
        if score < min_recall:
            row_id = str(row.get("id", "<unknown>"))
            return RegressionGateResult(
                status=GateStatus.FAILED,
                message=(
                    f"Regression gate FAILED: Recall@5 {score:.3f} < {min_recall} for {row_id}."
                ),
            )

    return RegressionGateResult(
        status=GateStatus.PASSED,
        message=(
            f"Regression gate PASSED: {len(real)} retrieval samples, "
            + f"{qa_count} QA pairs, Recall@5 >= {min_recall}."
        ),
    )


def main() -> None:
    result = check_regression_gate()
    print(result.message)
    if result.status == GateStatus.FAILED:
        sys.exit(1)


def baseline_int(baseline: dict[str, object], key: str, default: int) -> int:
    value = baseline.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def baseline_float(baseline: dict[str, object], key: str, default: float) -> float:
    value = baseline.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
