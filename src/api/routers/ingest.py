from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_ingestion_pipeline
from src.api.security import (
    read_upload_bounded,
    require_api_key,
    validate_ingest_path,
    validate_upload_filename,
)
from src.core.exceptions import IngestionError
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

router = APIRouter(
    prefix="/ingest",
    tags=["ingest"],
    dependencies=[Depends(require_api_key)],
)


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
    """Ingest a file or directory under configured allowed roots."""
    source = validate_ingest_path(Path(body.source))
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
    safe_name = validate_upload_filename(file.filename)
    suffix = Path(safe_name).suffix
    tmp_path: Path | None = None
    try:
        payload = await read_upload_bounded(file)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            upload_path = Path(tmp.name)
        tmp_path = upload_path
        result = pipeline.ingest_file(upload_path)
        return IngestResponse(
            status="ok",
            source=safe_name,
            chunk_count=result.chunk_count,
            content_hash=result.content_hash,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
