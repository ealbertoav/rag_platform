from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_ingestion_pipeline
from src.core.exceptions import IngestionError
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestPathRequest(BaseModel):
    source: str


class IngestResponse(BaseModel):
    status: str
    source: str
    chunk_count: int
    content_hash: str


@router.post("/path", response_model=IngestResponse)
async def ingest_path(
    body: IngestPathRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest a local file or directory by path."""
    source = Path(body.source)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {body.source}")
    try:
        if source.is_dir():
            results = pipeline.ingest_directory(source)
            ok = [r for r in results if r.error is None]
            total_chunks = sum(r.chunk_count for r in ok)
            return IngestResponse(
                status="ok",
                source=str(source),
                chunk_count=total_chunks,
                content_hash="",
            )
        result = pipeline.ingest_file(source)
        return IngestResponse(
            status="ok",
            source=result.source,
            chunk_count=result.chunk_count,
            content_hash=result.content_hash,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Ingest an uploaded file (saved to a temp path, then ingested)."""
    import tempfile

    suffix = Path(file.filename or "upload").suffix
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        result = pipeline.ingest_file(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return IngestResponse(
            status="ok",
            source=file.filename or tmp_path.name,
            chunk_count=result.chunk_count,
            content_hash=result.content_hash,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
