"""T-223 — scanned-PDF OCR fallback unit tests."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import OCR_APPLIED_KEY
from src.core.exceptions import DocumentLoadError
from src.core.settings import Settings
from src.domain.entities.document import Document
from src.rag.ingestion.ocr_fallback import (
    apply_ocr_fallback,
    document_needs_ocr,
    text_is_low,
)
from tests.unit.ingestion_helpers import mock_ingestion_pipeline


def _doc(
    content: str = "",
    *,
    source: str = "/tmp/scan.pdf",
    metadata: dict | None = None,
) -> Document:
    return Document(source=source, content=content, metadata=metadata or {})


def _ocr_settings(
    *,
    enabled: bool = True,
    provider: str = "tesseract",
    min_chars: int = 50,
) -> Settings:
    return Settings(
        parsing={"ocr": {"enabled": enabled, "provider": provider, "min_chars": min_chars}}
    )


class TestTextIsLow:
    def test_empty_is_low(self) -> None:
        assert text_is_low("", 50) is True

    def test_whitespace_only_is_low(self) -> None:
        assert text_is_low("   \n\t  ", 50) is True

    def test_below_threshold_is_low(self) -> None:
        assert text_is_low("short", 50) is True

    def test_at_threshold_is_not_low(self) -> None:
        assert text_is_low("x" * 50, 50) is False

    def test_above_threshold_is_not_low(self) -> None:
        assert text_is_low("enough extractable text here " * 5, 50) is False

    def test_zero_min_chars_only_empty_is_low(self) -> None:
        assert text_is_low("", 0) is False
        assert text_is_low("a", 0) is False


class TestDocumentNeedsOcr:
    def test_empty_content_needs_ocr(self) -> None:
        assert document_needs_ocr(_doc(""), 50) is True

    def test_sufficient_content_skips_ocr(self) -> None:
        assert document_needs_ocr(_doc("x" * 80), 50) is False

    def test_all_low_text_pages_need_ocr(self) -> None:
        doc = _doc(
            "a\n\nb",
            metadata={"pages": ["", "  ", "tiny"]},
        )
        assert document_needs_ocr(doc, 50) is True

    def test_mixed_pages_skip_ocr(self) -> None:
        born_digital = "This page has plenty of extractable text from the PDF."
        doc = _doc(
            born_digital + "\n\n",
            metadata={"pages": [born_digital, ""]},
        )
        assert document_needs_ocr(doc, 50) is False

    def test_non_string_page_entries_treated_as_empty(self) -> None:
        doc = _doc("x", metadata={"pages": [None, 123, ""]})
        assert document_needs_ocr(doc, 50) is True

    def test_empty_pages_list_falls_back_to_content(self) -> None:
        doc = _doc("x" * 80, metadata={"pages": []})
        assert document_needs_ocr(doc, 50) is False

    def test_non_list_pages_falls_back_to_content(self) -> None:
        doc = _doc("", metadata={"pages": "not-a-list"})
        assert document_needs_ocr(doc, 50) is True


class TestApplyOcrFallback:
    def test_non_pdf_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("hello")
        doc = _doc("hello", source=str(path))
        provider = MagicMock()
        result = apply_ocr_fallback(doc, path, ocr_provider=provider)
        assert result is doc
        provider.ocr.assert_not_called()

    def test_disabled_ocr_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path))
        settings = _ocr_settings(enabled=False)
        with patch(
            "src.infrastructure.ocr.get_ocr_provider",
            return_value=None,
        ) as get_provider:
            result = apply_ocr_fallback(doc, path, app_settings=settings)
        assert result is doc
        get_provider.assert_called_once_with(settings)

    def test_sufficient_text_skips_provider(self, tmp_path: Path) -> None:
        path = tmp_path / "born.pdf"
        path.write_bytes(b"%PDF-1.4")
        content = "Born-digital PDF with plenty of extractable characters."
        doc = _doc(content, source=str(path), metadata={"pages": [content]})
        provider = MagicMock()
        result = apply_ocr_fallback(
            doc,
            path,
            app_settings=_ocr_settings(enabled=True),
            ocr_provider=provider,
        )
        assert result is doc
        provider.ocr.assert_not_called()

    def test_low_text_pdf_replaces_content(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path), metadata={"pages": ["", ""]})
        provider = MagicMock()
        provider.ocr.return_value = "  Recognized OCR text from scan.  "
        result = apply_ocr_fallback(
            doc,
            path,
            app_settings=_ocr_settings(enabled=True),
            ocr_provider=provider,
        )
        provider.ocr.assert_called_once_with(path)
        assert result.content == "Recognized OCR text from scan."
        assert result.metadata[OCR_APPLIED_KEY] is True
        assert result.id == doc.id

    def test_empty_ocr_result_keeps_original(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("tiny", source=str(path))
        provider = MagicMock()
        provider.ocr.return_value = "   "
        with caplog.at_level(logging.WARNING):
            result = apply_ocr_fallback(
                doc,
                path,
                app_settings=_ocr_settings(enabled=True),
                ocr_provider=provider,
            )
        assert result is doc
        assert "OCR returned empty text" in caplog.text

    def test_ocr_failure_keeps_original(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path))
        provider = MagicMock()
        provider.ocr.side_effect = DocumentLoadError("OCR boom")
        with caplog.at_level(logging.WARNING):
            result = apply_ocr_fallback(
                doc,
                path,
                app_settings=_ocr_settings(enabled=True),
                ocr_provider=provider,
            )
        assert result is doc
        assert "OCR failed" in caplog.text

    def test_uses_global_settings_when_not_passed(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path))
        provider = MagicMock()
        provider.ocr.return_value = "from global settings"
        live = _ocr_settings(enabled=True, min_chars=10)
        with (
            patch("src.core.settings.settings", live),
            patch("src.infrastructure.ocr.get_ocr_provider", return_value=provider) as get_provider,
        ):
            result = apply_ocr_fallback(doc, path)
        get_provider.assert_called_once_with(None)
        assert result.content == "from global settings"
        assert result.metadata[OCR_APPLIED_KEY] is True

    def test_min_chars_from_settings(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        # 40 chars — above default 50? No: below 100 with custom min_chars.
        content = "x" * 40
        doc = _doc(content, source=str(path))
        provider = MagicMock()
        provider.ocr.return_value = "ocr replacement"
        result = apply_ocr_fallback(
            doc,
            path,
            app_settings=_ocr_settings(enabled=True, min_chars=100),
            ocr_provider=provider,
        )
        assert result.content == "ocr replacement"


class TestIngestionPipelineOcrWiring:
    def test_ingest_file_applies_ocr_before_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        loaded = _doc("", source=str(path.resolve()), metadata={"pages": [""]})
        ocr_doc = loaded.model_copy(
            update={
                "content": "OCR recovered text for hashing",
                "metadata": {**loaded.metadata, OCR_APPLIED_KEY: True},
            }
        )
        pipeline, service, *_ = mock_ingestion_pipeline()
        with (
            patch(
                "src.rag.pipelines.ingestion_pipeline.load_document",
                return_value=loaded,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.apply_ocr_fallback",
                return_value=ocr_doc,
            ) as ocr_fn,
        ):
            result = pipeline.ingest_file(path)

        ocr_fn.assert_called_once()
        assert ocr_fn.call_args.args[0] is loaded
        assert ocr_fn.call_args.args[1] == path
        service.prepare.assert_called_once_with(ocr_doc)
        from src.rag.pipelines.ingestion_pipeline import content_hash

        assert result.content_hash == content_hash(
            str(path.resolve()),
            "OCR recovered text for hashing",
        )

    def test_ingest_file_noop_when_ocr_returns_same_doc(self, tmp_path: Path) -> None:
        path = tmp_path / "born.pdf"
        path.write_bytes(b"%PDF-1.4")
        loaded = _doc("enough text " * 20, source=str(path.resolve()))
        pipeline, service, *_ = mock_ingestion_pipeline()
        with (
            patch(
                "src.rag.pipelines.ingestion_pipeline.load_document",
                return_value=loaded,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.apply_ocr_fallback",
                return_value=loaded,
            ),
        ):
            result = pipeline.ingest_file(path)

        service.prepare.assert_called_once_with(loaded)
        assert result.skipped is False
