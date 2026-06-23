from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics", response_class=Response)
def metrics() -> Response:
    """Expose Prometheus metrics in text format for scraping.

    Compatible with Grafana / Prometheus scrape configs.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
