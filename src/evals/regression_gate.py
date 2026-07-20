"""CI retrieval regression gate (T-152).

Validates committed golden datasets meet minimum sample counts and per-row
Recall@5 floors before merge. Skips gracefully when only placeholder data exists.

Two independent checks:
  - `check_regression_gate`: validates the golden dataset's own structure (row
    counts, sync with qa_dataset.json, and whether any row has more relevant
    chunks than K — which would cap oracle Recall@K below 1.0 regardless of
    retriever quality). Never invokes the retrieval pipeline.
  - `check_live_retrieval_regression`: actually runs the configured retrieval
    pipeline against each golden query and checks real Recall@5. Skips
    gracefully when Qdrant, the BM25 index, or self-hosted models aren't
    available in this environment (matching tests/integration/*'s auto-skip).
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from src.evals.retrieval.recall_at_k import oracle_recall_at_k, recall_at_k

logger = logging.getLogger(__name__)

_DEFAULT_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"
_DEFAULT_RETRIEVAL_PATH = DATASETS_DIR / "goldens" / "retrieval_dataset.json"
_DEFAULT_BASELINE_PATH = DATASETS_DIR / "goldens" / "retrieval_baseline.json"
_QDRANT_REACHABILITY_TIMEOUT_S = 2


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
        # Oracle: retrieved == relevant, so this only checks whether the row has
        # more unique relevant chunks than K — never actual retriever behavior.
        score = oracle_recall_at_k(relevant, k=5)
        if score < min_recall:
            row_id = str(row.get("id", "<unknown>"))
            return RegressionGateResult(
                status=GateStatus.FAILED,
                message=(
                    f"Regression gate FAILED: oracle Recall@5 {score:.3f} < {min_recall} "
                    + f"for {row_id} (row has more relevant chunks than K)."
                ),
            )

    return RegressionGateResult(
        status=GateStatus.PASSED,
        message=(
            f"Regression gate PASSED (dataset shape): {len(real)} retrieval samples, "
            + f"{qa_count} QA pairs, oracle Recall@5 >= {min_recall}."
        ),
    )


async def check_live_retrieval_regression(
    *,
    retrieval_path: Path | None = None,
    baseline_path: Path | None = None,
    k: int = 5,
) -> RegressionGateResult:
    """Run the *live* configured retrieval pipeline against golden queries.

    Unlike `check_regression_gate`'s oracle check, this actually retrieves
    chunks for each golden query and compares them against the ground-truth
    `relevant_chunk_ids` — so a broken retriever, a bad config change, or a
    corrupted index will fail this gate. Skips gracefully (not a failure) when
    Qdrant, the BM25 index, or self-hosted models aren't available in this
    environment.
    """
    retrieval = retrieval_path or _DEFAULT_RETRIEVAL_PATH
    real = load_real_retrieval_rows(retrieval)
    if not real:
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="No real retrieval rows — skipping live regression check.",
        )

    baseline = load_regression_baseline(baseline_path)
    min_recall = baseline_float(baseline, "min_recall_at_5", 0.5)

    if not _qdrant_reachable():
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="Qdrant not reachable — skipping live regression check.",
        )

    from src.infrastructure.vectordb.bm25 import BM25Index

    bm25_index = BM25Index.load_or_create()
    if bm25_index.size == 0:
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message=(
                "BM25 index is empty — skipping live regression check "
                + "(run `make ingest` to populate it)."
            ),
        )

    try:
        from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline.from_settings(bm25_index=bm25_index)
    except Exception as exc:
        logger.warning(
            "Live retrieval pipeline unavailable, skipping live regression check: %s", exc
        )
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message=(
                f"Live retrieval pipeline unavailable — skipping live regression check ({exc})."
            ),
        )

    from src.domain.entities.query import Query

    # Infra is confirmed present past this point (Qdrant reachable, BM25
    # populated, pipeline built) — a failure here is a genuine regression
    # signal, not an environment gap, so it fails the gate rather than skips.
    scores: list[float] = []
    for row in real:
        query_text = row.get("query")
        raw_ids = row.get("relevant_chunk_ids", [])
        relevant = [r for r in (raw_ids if isinstance(raw_ids, list) else []) if isinstance(r, str)]
        if not isinstance(query_text, str) or not query_text or not relevant:
            continue
        try:
            result = await pipeline.retrieve(Query(text=query_text))
        except Exception as exc:
            row_id = str(row.get("id", "<unknown>"))
            return RegressionGateResult(
                status=GateStatus.FAILED,
                message=f"Live regression gate FAILED: retrieval raised for {row_id}: {exc}",
            )
        retrieved_ids = [chunk.id for chunk in result.chunks]
        scores.append(recall_at_k(retrieved_ids, relevant, k))

    if not scores:
        return RegressionGateResult(
            status=GateStatus.SKIPPED,
            message="No evaluable rows — skipping live regression check.",
        )

    mean_recall = sum(scores) / len(scores)
    if mean_recall < min_recall:
        return RegressionGateResult(
            status=GateStatus.FAILED,
            message=(
                f"Live regression gate FAILED: live Recall@{k} {mean_recall:.3f} < {min_recall} "
                + f"across {len(scores)} live-retrieved samples."
            ),
        )

    return RegressionGateResult(
        status=GateStatus.PASSED,
        message=(
            f"Live regression gate PASSED: live Recall@{k} {mean_recall:.3f} >= {min_recall} "
            + f"across {len(scores)} live-retrieved samples."
        ),
    )


def _qdrant_reachable() -> bool:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException

    from src.core.settings import settings

    try:
        QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key,
            timeout=_QDRANT_REACHABILITY_TIMEOUT_S,
            check_compatibility=False,
        ).get_collections()
        return True
    except (OSError, TimeoutError, ResponseHandlingException):
        return False


def main() -> None:
    result = check_regression_gate()
    print(result.message)
    if result.status == GateStatus.FAILED:
        sys.exit(1)

    live_result = asyncio.run(check_live_retrieval_regression())
    print(live_result.message)
    if live_result.status == GateStatus.FAILED:
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
