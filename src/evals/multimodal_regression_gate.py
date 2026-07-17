"""CI multimodal (table/figure) regression gate (T-282).

Validates the T-280 multimodal golden dataset against per-modality minimum
sample counts and oracle Recall@5 floors (T-281's `table_recall_at_k` /
`figure_recall_at_k` inputs), mirroring T-152's `check_regression_gate`
pattern. Table/figure ingestion is opt-in (T-202/T-253), so most checkouts
never populate `datasets/goldens/multimodal_qa_dataset.jsonl` — the gate
skips gracefully rather than blocking CI when that data is absent, which is
what makes it "CI-optional".
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.core.constants import DATASETS_DIR, MODALITY_FIGURE, MODALITY_TABLE
from src.evals.multimodal_golden_dataset import load_jsonl
from src.evals.regression_gate import (
    GateStatus,
    RegressionGateResult,
    baseline_float,
    baseline_int,
    load_regression_baseline,
)
from src.evals.retrieval.modality_recall import load_modality_samples, samples_for_modality
from src.evals.retrieval.recall_at_k import oracle_recall_at_k

_DEFAULT_DATASET_PATH = DATASETS_DIR / "goldens" / "multimodal_qa_dataset.jsonl"
_DEFAULT_BASELINE_PATH = DATASETS_DIR / "goldens" / "multimodal_baseline.json"

_DEFAULT_MIN_SAMPLES = 1
_DEFAULT_MIN_RECALL = 0.5
_ORACLE_K = 5


def check_multimodal_regression_gate(
    *,
    dataset_path: Path | None = None,
    baseline_path: Path | None = None,
) -> RegressionGateResult:
    """Evaluate the multimodal regression gate without exiting the process."""
    dataset = dataset_path or _DEFAULT_DATASET_PATH
    rows = load_jsonl(dataset)
    if not rows:
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="No multimodal golden dataset — skipping multimodal regression gate.",
        )

    samples = load_modality_samples(rows)
    table_samples = samples_for_modality(samples, MODALITY_TABLE)
    figure_samples = samples_for_modality(samples, MODALITY_FIGURE)
    if not table_samples and not figure_samples:
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="No table/figure samples in multimodal golden — skipping.",
        )

    baseline = load_regression_baseline(baseline_path or _DEFAULT_BASELINE_PATH)

    for modality, modality_samples in (
        (MODALITY_TABLE, table_samples),
        (MODALITY_FIGURE, figure_samples),
    ):
        if not modality_samples:
            continue

        min_samples = baseline_int(baseline, f"min_{modality}_samples", _DEFAULT_MIN_SAMPLES)
        if len(modality_samples) < min_samples:
            return RegressionGateResult(
                status=GateStatus.FAILED,
                message=(
                    f"Regression gate FAILED: need >= {min_samples} {modality} samples "
                    + f"(found {len(modality_samples)})."
                ),
            )

        min_recall = baseline_float(baseline, f"min_{modality}_recall_at_5", _DEFAULT_MIN_RECALL)
        for index, sample in enumerate(modality_samples):
            if not sample.relevant_ids:
                return RegressionGateResult(
                    status=GateStatus.FAILED,
                    message=(
                        f"Regression gate FAILED: {modality} sample {index} "
                        + "has no relevant_chunks."
                    ),
                )
            score = oracle_recall_at_k(sample.relevant_ids, k=_ORACLE_K)
            if score < min_recall:
                return RegressionGateResult(
                    status=GateStatus.FAILED,
                    message=(
                        f"Regression gate FAILED: {modality} oracle Recall@{_ORACLE_K} "
                        + f"{score:.3f} < {min_recall} for sample {index}."
                    ),
                )

    return RegressionGateResult(
        status=GateStatus.PASSED,
        message=(
            f"Multimodal regression gate PASSED: {len(table_samples)} table samples, "
            + f"{len(figure_samples)} figure samples."
        ),
    )


def main() -> None:
    result = check_multimodal_regression_gate()
    print(result.message)
    if result.status == GateStatus.FAILED:
        sys.exit(1)
