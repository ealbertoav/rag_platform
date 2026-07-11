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
    should_attempt_ocr,
    text_is_low,
)
from src.rag.pipelines.ingestion_pipeline import content_hash, source_file_hash
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

    def test_internal_whitespace_does_not_count_toward_threshold(self) -> None:
        # 1 non-whitespace char padded with spaces — strip()-based length would
        # wrongly treat this as above threshold.
        assert text_is_low("a" + (" " * 100), 50) is True

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

    def test_whitespace_padded_pages_need_ocr(self) -> None:
        doc = _doc(
            "a" + (" " * 80),
            metadata={"pages": ["a" + (" " * 80), "\t\n" + (" " * 60)]},
        )
        assert document_needs_ocr(doc, 50) is True

    def test_all_low_text_pages_need_ocr(self) -> None:
        doc = _doc(
            "a\n\nb",
            metadata={"pages": ["", "  ", "tiny"]},
        )
        assert document_needs_ocr(doc, 50) is True

    def test_mixed_pages_skip_ocr(self) -> None:
        # ≥50 non-whitespace chars so internal spaces do not inflate the count.
        born_digital = (
            "This page has plenty of extractable text from the born-digital PDF document."
        )
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


class TestShouldAttemptOcr:
    def test_non_pdf_false(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("hello")
        assert should_attempt_ocr(_doc("hello", source=str(path)), path) is False

    def test_disabled_false(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        assert (
            should_attempt_ocr(
                _doc("", source=str(path)),
                path,
                app_settings=_ocr_settings(enabled=False),
            )
            is False
        )

    def test_low_text_enabled_true(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        assert (
            should_attempt_ocr(
                _doc("", source=str(path)),
                path,
                app_settings=_ocr_settings(enabled=True),
            )
            is True
        )

    def test_sufficient_text_false(self, tmp_path: Path) -> None:
        path = tmp_path / "born.pdf"
        path.write_bytes(b"%PDF-1.4")
        content = "Born-digital PDF with plenty of extractable characters across the page content."
        assert (
            should_attempt_ocr(
                _doc(content, source=str(path), metadata={"pages": [content]}),
                path,
                app_settings=_ocr_settings(enabled=True),
            )
            is False
        )

    def test_uses_global_settings_when_not_passed(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        live = _ocr_settings(enabled=True, min_chars=10)
        with patch("src.core.settings.settings", live):
            assert should_attempt_ocr(_doc("", source=str(path)), path) is True


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
        with patch("src.infrastructure.ocr.get_ocr_provider") as get_provider:
            result = apply_ocr_fallback(doc, path, app_settings=settings)
        assert result is doc
        get_provider.assert_not_called()

    def test_sufficient_text_skips_provider(self, tmp_path: Path) -> None:
        path = tmp_path / "born.pdf"
        path.write_bytes(b"%PDF-1.4")
        content = "Born-digital PDF with plenty of extractable characters across the page content."
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

    def test_sufficient_text_skips_factory_resolution(self, tmp_path: Path) -> None:
        """Born-digital PDFs must not construct the OCR provider (Bugbot / T-223)."""
        path = tmp_path / "born.pdf"
        path.write_bytes(b"%PDF-1.4")
        content = "Born-digital PDF with plenty of extractable characters across the page content."
        doc = _doc(content, source=str(path), metadata={"pages": [content]})
        settings = _ocr_settings(enabled=True, provider="azure_di")
        with patch("src.infrastructure.ocr.get_ocr_provider") as get_provider:
            result = apply_ocr_fallback(doc, path, app_settings=settings)
        assert result is doc
        get_provider.assert_not_called()

    def test_misconfigured_provider_keeps_original_when_ocr_needed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from src.core.exceptions import ConfigurationError

        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path))
        settings = _ocr_settings(enabled=True, provider="azure_di")
        with (
            caplog.at_level(logging.WARNING),
            patch(
                "src.infrastructure.ocr.get_ocr_provider",
                side_effect=ConfigurationError("azure_di not implemented yet (T-222)"),
            ) as get_provider,
        ):
            result = apply_ocr_fallback(doc, path, app_settings=settings)
        assert result is doc
        get_provider.assert_called_once_with(settings)
        assert "OCR provider misconfigured" in caplog.text

    def test_factory_returns_none_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path))
        settings = _ocr_settings(enabled=True)
        with patch(
            "src.infrastructure.ocr.get_ocr_provider",
            return_value=None,
        ) as get_provider:
            result = apply_ocr_fallback(doc, path, app_settings=settings)
        assert result is doc
        get_provider.assert_called_once_with(settings)

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

    def test_whitespace_padded_low_text_triggers_ocr(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        padded = "x" + (" " * 80)
        doc = _doc(padded, source=str(path), metadata={"pages": [padded]})
        provider = MagicMock()
        provider.ocr.return_value = "recovered"
        result = apply_ocr_fallback(
            doc,
            path,
            app_settings=_ocr_settings(enabled=True, min_chars=50),
            ocr_provider=provider,
        )
        provider.ocr.assert_called_once_with(path)
        assert result.content == "recovered"

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

    def test_injected_provider_bypasses_enabled_flag(self, tmp_path: Path) -> None:
        """Explicit DI still runs OCR when the document needs it."""
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _doc("", source=str(path))
        provider = MagicMock()
        provider.ocr.return_value = "injected"
        result = apply_ocr_fallback(
            doc,
            path,
            app_settings=_ocr_settings(enabled=False),
            ocr_provider=provider,
        )
        provider.ocr.assert_called_once_with(path)
        assert result.content == "injected"


class TestSourceFileHash:
    def test_stable_for_same_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 identical")
        source = str(path.resolve())
        assert source_file_hash(source, path) == source_file_hash(source, path)

    def test_differs_from_text_content_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        assert source_file_hash(source, path) != content_hash(source, "")

    def test_changes_when_file_bytes_change(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 a")
        source = str(path.resolve())
        first = source_file_hash(source, path)
        path.write_bytes(b"%PDF-1.4 b")
        assert source_file_hash(source, path) != first


class TestIngestionPipelineOcrWiring:
    def test_ingest_file_applies_ocr_and_stores_file_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 scanned")
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
                "src.rag.pipelines.ingestion_pipeline.should_attempt_ocr",
                return_value=True,
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
        assert result.content_hash == source_file_hash(str(path.resolve()), path)

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
                "src.rag.pipelines.ingestion_pipeline.should_attempt_ocr",
                return_value=False,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.apply_ocr_fallback",
                return_value=loaded,
            ),
        ):
            result = pipeline.ingest_file(path)

        service.prepare.assert_called_once_with(loaded)
        assert result.skipped is False
        assert result.content_hash == content_hash(str(path.resolve()), loaded.content)

    def test_unchanged_scanned_pdf_skips_without_ocr(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 unchanged scan")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        file_hash = source_file_hash(source, path)
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=file_hash,
            chunk_count=3,
        )
        metadata.get_chunk_ids.return_value = ["c1", "c2", "c3"]
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        with (
            patch(
                "src.rag.pipelines.ingestion_pipeline.load_document",
                return_value=loaded,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.should_attempt_ocr",
                return_value=True,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.apply_ocr_fallback",
            ) as ocr_fn,
        ):
            result = pipeline.ingest_file(path)

        ocr_fn.assert_not_called()
        service.prepare.assert_not_called()
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()
        assert result.skipped is True
        assert result.content_hash == file_hash
        assert result.chunk_count == 3

    def test_unchanged_scanned_pdf_with_augmentor_ocrs_then_reindexes(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        ocr_doc = loaded.model_copy(
            update={
                "content": "OCR text for reindex",
                "metadata": {**loaded.metadata, OCR_APPLIED_KEY: True},
            }
        )
        file_hash = source_file_hash(source, path)
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=file_hash,
            chunk_count=1,
        )
        metadata.get_chunk_ids.return_value = ["c1"]
        augmentor = MagicMock()
        augmentor.augment.return_value = []
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(
            metadata=metadata,
            augmentor=augmentor,
        )
        with (
            patch(
                "src.rag.pipelines.ingestion_pipeline.load_document",
                return_value=loaded,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.should_attempt_ocr",
                return_value=True,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.apply_ocr_fallback",
                return_value=ocr_doc,
            ) as ocr_fn,
        ):
            result = pipeline.ingest_file(path)

        ocr_fn.assert_called_once()
        service.prepare.assert_called_once_with(ocr_doc)
        vector_store.upsert.assert_called_once()
        bm25.add.assert_called_once()
        assert result.skipped is False
        assert result.content_hash == file_hash

    def test_ocr_failure_stores_text_hash_not_file_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("tiny", source=source)
        pipeline, service, *_ = mock_ingestion_pipeline()
        with (
            patch(
                "src.rag.pipelines.ingestion_pipeline.load_document",
                return_value=loaded,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.should_attempt_ocr",
                return_value=True,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.apply_ocr_fallback",
                return_value=loaded,
            ),
        ):
            result = pipeline.ingest_file(path)

        service.prepare.assert_called_once_with(loaded)
        assert result.content_hash == content_hash(source, "tiny")
        assert result.content_hash != source_file_hash(source, path)
