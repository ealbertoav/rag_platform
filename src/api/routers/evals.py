from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_chat_pipeline
from src.rag.pipelines.chat_pipeline import ChatPipeline

router = APIRouter(prefix="/evals", tags=["evals"])


class EvalsRunResponse(BaseModel):
    status: str
    timestamp: str
    total_samples: int
    mean_recall_at_5: float
    mean_faithfulness: float
    mean_relevance: float
    mean_context_precision: float
    mean_hallucination: float
    passed: bool
    report_path: str
    message: str


@router.post("/run", response_model=EvalsRunResponse)
async def run_evals(
    pipeline: ChatPipeline = Depends(get_chat_pipeline),
) -> EvalsRunResponse:
    """Run the end-to-end RAG benchmark against the golden QA dataset.

    Loads "datasets/goldens/qa_dataset.json", runs every question through
    the full pipeline, computes Recall@5 / Faithfulness / Relevance /
    Context Precision / Hallucination, saves a timestamped report to
    a timestamped report to "data/exports/" and returns the summary.

    Returns 204 if the QA dataset is empty (no real samples found).
    """
    from src.core.constants import EXPORTS_DIR
    from src.domain.services.evaluation_service import EvaluationService

    svc = EvaluationService.from_settings(pipeline)
    report = await svc.run()

    if report.total_samples == 0:
        raise HTTPException(
            status_code=204,
            detail="QA dataset is empty — generate samples with `make evals` first.",
        )

    report_path = str(EXPORTS_DIR / f"benchmark_{report.timestamp}.json")
    status = "passed" if report.passed else "failed"

    return EvalsRunResponse(
        status=status,
        timestamp=report.timestamp,
        total_samples=report.total_samples,
        mean_recall_at_5=report.mean_recall_at_5,
        mean_faithfulness=report.mean_faithfulness,
        mean_relevance=report.mean_relevance,
        mean_context_precision=report.mean_context_precision,
        mean_hallucination=report.mean_hallucination,
        passed=report.passed,
        report_path=report_path,
        message=(
            "All metrics above threshold ✓"
            if report.passed
            else "One or more metrics below threshold — see report for details."
        ),
    )
