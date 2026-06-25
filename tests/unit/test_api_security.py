"""Unit tests for API security helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api.security import (
    read_upload_bounded,
    resolve_allowed_ingest_roots,
    validate_ingest_path,
    validate_upload_filename,
)
from src.core.settings import settings


class TestValidateIngestPath:
    def test_allows_path_under_configured_root(
            self, monkeypatch:pytest.MonkeyPatch,
            tmp_path: Path)\
            :
        root = tmp_path / "data" / "raw"
        root.mkdir(parents=True)
        doc = root / "doc.md"
        doc.write_text("hello")
        monkeypatch.setattr(settings.api, "ingest_allowed_roots", [str(root)])
        assert validate_ingest_path(doc) == doc.resolve()

    def test_rejects_path_outside_root(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.api, "ingest_allowed_roots", ["data/raw"])
        with pytest.raises(HTTPException) as exc:
            validate_ingest_path(Path("/etc/passwd"))
        assert exc.value.status_code == 403


class TestValidateUploadFilename:
    def test_accepts_supported_extension(self):
        assert validate_upload_filename("report.pdf") == "report.pdf"

    def test_rejects_unsupported_extension(self):
        with pytest.raises(HTTPException) as exc:
            validate_upload_filename("payload.exe")
        assert exc.value.status_code == 415


class TestReadUploadBounded:
    @pytest.mark.asyncio
    async def test_reads_within_limit(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.api, "max_upload_bytes", 100)
        upload = MagicMock()
        upload.read = AsyncMock(side_effect=[b"hello", b""])
        assert await read_upload_bounded(upload) == b"hello"

    @pytest.mark.asyncio
    async def test_rejects_oversized_upload(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.api, "max_upload_bytes", 3)
        upload = MagicMock()
        upload.read = AsyncMock(side_effect=[b"1234", b""])
        with pytest.raises(HTTPException) as exc:
            await read_upload_bounded(upload)
        assert exc.value.status_code == 413


class TestResolveAllowedIngestRoots:
    def test_relative_roots_are_resolved_from_project_root(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(settings.api, "ingest_allowed_roots", ["data/raw"])
        roots = resolve_allowed_ingest_roots()
        assert roots[0].name == "raw"
        assert roots[0].parent.name == "data"
