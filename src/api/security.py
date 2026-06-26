from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import Header, HTTPException, UploadFile, status

from src.core.constants import ROOT, SUPPORTED_EXTENSIONS
from src.core.settings import settings

_READ_CHUNK_SIZE = 64 * 1024


def _configured_api_key() -> str:
    return settings.api.api_key.get_secret_value()


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Require a matching API key when "api.api_key" is configured."""
    expected = _configured_api_key()
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def resolve_allowed_ingest_roots() -> list[Path]:
    """Return absolute, resolved ingest roots from settings."""
    roots: list[Path] = []
    for entry in settings.api.ingest_allowed_roots:
        path = Path(entry)
        if not path.is_absolute():
            path = ROOT / path
        roots.append(path.resolve())
    return roots


def validate_ingest_path(source: Path) -> Path:
    """Ensure a *source* resolves under a configured ingested root."""
    resolved = source.expanduser()
    resolved = (ROOT / resolved).resolve() if not resolved.is_absolute() else resolved.resolve()
    allowed_roots = resolve_allowed_ingest_roots()
    if not allowed_roots:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ingest path access is disabled (no allowed roots configured)",
        )
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ingest path must be under an allowed root directory",
        )
    return resolved


def validate_upload_filename(filename: str | None) -> str:
    """Return a safe basename with a supported extension."""
    name = Path(filename or "upload").name
    if not name or name in {".", ".."} or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload filename",
        )
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix or '(none)'}",
        )
    return name


async def read_upload_bounded(file: UploadFile, max_bytes: int | None = None) -> bytes:
    """Read an upload in chunks, rejecting payloads above the configured cap."""
    limit = max_bytes if max_bytes is not None else settings.api.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        block = await file.read(_READ_CHUNK_SIZE)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds maximum size of {limit} bytes",
            )
        chunks.append(block)
    return b"".join(chunks)
