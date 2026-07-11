"""T-223 — scanned-PDF OCR fallback unit tests."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import (
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_TABLE,
    OCR_APPLIED_KEY,
    TABLE_ID_KEY,
)
from src.core.exceptions import DocumentLoadError
from src.core.settings import Settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.ingestion.ocr_fallback import (
    apply_ocr_fallback,
    document_needs_ocr,
    should_attempt_ocr,
    text_is_low,
)
from src.rag.ingestion.table_chunker import table_chunk_id
from src.rag.pipelines.ingestion_pipeline import (
    IngestionPipeline,
    IngestionResult,
    content_hash,
    hash_after_ocr,
    is_unchanged_source,
    ocr_pending_hash,
    source_file_hash,
)
from tests.unit.ingestion_helpers import mock_ingestion_pipeline

_INGEST_MOD = "src.rag.pipelines.ingestion_pipeline"
_UNSET = object()
_SAMPLE_TABLE_MD = "| A | B |\n|---|---|\n| 1 | 2 |"
_MIN_CHARS_RAISED_CONTENT = "Extractable text that was enough under the old min_chars threshold."


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


def _metadata_record(
    content_hash_value: str,
    *,
    chunk_count: int,
    chunk_ids: list[str] | None = None,
) -> MagicMock:
    metadata = MagicMock()
    metadata.get_by_source.return_value = MagicMock(
        id="doc-1",
        content_hash=content_hash_value,
        chunk_count=chunk_count,
    )
    if chunk_ids is not None:
        metadata.get_chunk_ids.return_value = chunk_ids
    elif chunk_count == 0:
        metadata.get_chunk_ids.return_value = []
    else:
        metadata.get_chunk_ids.return_value = [f"c{i}" for i in range(1, chunk_count + 1)]
    return metadata


def _ocr_applied_doc(loaded: Document, content: str) -> Document:
    return loaded.model_copy(
        update={
            "content": content,
            "metadata": {**loaded.metadata, OCR_APPLIED_KEY: True},
        }
    )


def _ingest_with_ocr_patches(
    pipeline: IngestionPipeline,
    path: Path,
    loaded: Document,
    *,
    ocr_candidate: bool,
    ocr_return: Document | object = _UNSET,
) -> tuple[IngestionResult, MagicMock]:
    """Run ingest_file with load/OCR helpers patched; return (result, ocr mock)."""
    ocr_kwargs: dict = {}
    if ocr_return is not _UNSET:
        ocr_kwargs["return_value"] = ocr_return
    with (
        patch(f"{_INGEST_MOD}.load_document", return_value=loaded),
        patch(f"{_INGEST_MOD}.should_attempt_ocr", return_value=ocr_candidate),
        patch(f"{_INGEST_MOD}.apply_ocr_fallback", **ocr_kwargs) as ocr_fn,
    ):
        return pipeline.ingest_file(path), ocr_fn


def _assert_skipped_without_ocr(
    result: IngestionResult,
    ocr_fn: MagicMock,
    service: MagicMock,
    *,
    expected_hash: str,
    chunk_count: int | None = None,
) -> None:
    ocr_fn.assert_not_called()
    service.prepare.assert_not_called()
    assert result.skipped is True
    assert result.content_hash == expected_hash
    if chunk_count is not None:
        assert result.chunk_count == chunk_count


def _file_keyed_empty_scan_with_layout_table(
    tmp_path: Path,
) -> tuple[Path, Document, str, str, Chunk]:
    """Return (path, empty OCR doc with layout table, file_hash, table_id, table_chunk)."""
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4")
    source = str(path.resolve())
    loaded = _doc(
        "",
        source=source,
        metadata={
            "pages": [""],
            "tables": [{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE_MD}],
        },
    )
    file_hash = source_file_hash(source, path)
    table_id = table_chunk_id(source, "table-1")
    table_chunk = Chunk(
        id=table_id,
        document_id="doc-1",
        text=_SAMPLE_TABLE_MD,
        embedding=[0.1] * 4,
        sparse_vector={1: 0.9},
        metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_TABLE, TABLE_ID_KEY: "table-1"},
    )
    return path, loaded, file_hash, table_id, table_chunk


def _text_keyed_born_digital_pdf(
    tmp_path: Path,
    *,
    content: str = _MIN_CHARS_RAISED_CONTENT,
) -> tuple[Path, Document, str, str]:
    """Return (path, loaded doc, content, text_hash) for a text-keyed born-digital PDF."""
    path = tmp_path / "born.pdf"
    path.write_bytes(b"%PDF-1.4")
    source = str(path.resolve())
    loaded = _doc(content, source=source, metadata={"pages": [content]})
    return path, loaded, content, content_hash(source, content)


def _pipeline_with_table_chunker(
    *,
    metadata: MagicMock,
    table_chunk: Chunk,
    augmentor: MagicMock | None = None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    table_chunker = MagicMock()
    table_chunker.index.return_value = [table_chunk]
    pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(
        metadata=metadata,
        augmentor=augmentor,
    )
    pipeline._table_chunker = table_chunker  # noqa: SLF001
    return pipeline, service, vector_store, bm25, table_chunker


def _assert_table_backfill(
    *,
    service: MagicMock,
    table_chunker: MagicMock,
    vector_store: MagicMock,
    bm25: MagicMock,
    result: IngestionResult,
    expected_hash: str,
) -> None:
    service.prepare.assert_not_called()
    table_chunker.index.assert_called_once()
    vector_store.upsert.assert_called_once()
    bm25.add.assert_called_once()
    assert result.skipped is False
    assert result.content_hash == expected_hash


def _assert_reindexed_without_ocr(
    result: IngestionResult,
    ocr_fn: MagicMock,
    service: MagicMock,
    vector_store: MagicMock,
    *,
    content: str,
    expected_hash: str,
) -> None:
    ocr_fn.assert_not_called()
    service.prepare.assert_called_once()
    prepared_doc = service.prepare.call_args.args[0]
    assert prepared_doc.id == "doc-1"
    assert prepared_doc.content == content
    assert prepared_doc.metadata.get(OCR_APPLIED_KEY) is not True
    vector_store.upsert.assert_called_once()
    assert result.skipped is False
    assert result.content_hash == expected_hash


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
        # ≥50 non-whitespace chars, so internal spaces do not inflate the count.
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


class TestDedupHashStability:
    def test_file_keyed_match_ignores_ocr_candidate_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        doc = _doc("", source=source)
        file_hash = source_file_hash(source, path)
        assert is_unchanged_source(file_hash, source, path, doc, ocr_candidate=False) is True
        assert is_unchanged_source(file_hash, source, path, doc, ocr_candidate=True) is True

    def test_text_keyed_empty_ocr_candidate_allows_recovery(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        doc = _doc("", source=source)
        text_hash = content_hash(source, "")
        assert is_unchanged_source(text_hash, source, path, doc, ocr_candidate=False) is True
        assert is_unchanged_source(text_hash, source, path, doc, ocr_candidate=True) is False

    def test_text_keyed_nonempty_skips_when_min_chars_would_trigger_ocr(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "born.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        content = "x" * 40  # below a raised min_chars=100, but already indexed
        doc = _doc(content, source=source)
        text_hash = content_hash(source, content)
        assert is_unchanged_source(text_hash, source, path, doc, ocr_candidate=True) is True

    def test_pending_hash_never_matches_text_or_file(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        pending = ocr_pending_hash(source, "tiny")
        doc = _doc("tiny", source=source)
        assert pending != content_hash(source, "tiny")
        assert pending != source_file_hash(source, path)
        assert is_unchanged_source(pending, source, path, doc, ocr_candidate=True) is False

    def test_hash_after_ocr_pending_when_candidate_without_apply(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        doc = _doc("tiny", source=source)
        assert hash_after_ocr(doc, path, source, ocr_candidate=True) == ocr_pending_hash(
            source, "tiny"
        )

    def test_hash_after_ocr_file_hash_when_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        doc = _doc(
            "OCR text",
            source=source,
            metadata={OCR_APPLIED_KEY: True},
        )
        assert hash_after_ocr(doc, path, source, ocr_candidate=True) == source_file_hash(
            source, path
        )

    def test_non_pdf_ignores_file_hash_scheme(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("hello")
        source = str(path.resolve())
        doc = _doc("hello", source=source)
        assert (
            is_unchanged_source(
                content_hash(source, "hello"),
                source,
                path,
                doc,
                ocr_candidate=False,
            )
            is True
        )


class TestIngestionPipelineOcrWiring:
    def test_ingest_file_applies_ocr_and_stores_file_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 scanned")
        loaded = _doc("", source=str(path.resolve()), metadata={"pages": [""]})
        ocr_doc = _ocr_applied_doc(loaded, "OCR recovered text for hashing")
        pipeline, service, *_ = mock_ingestion_pipeline()
        result, ocr_fn = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=ocr_doc
        )

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
        result, _ = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=False, ocr_return=loaded
        )

        service.prepare.assert_called_once_with(loaded)
        assert result.skipped is False
        assert result.content_hash == content_hash(str(path.resolve()), loaded.content)

    def test_unchanged_scanned_pdf_skips_without_ocr(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 unchanged scan")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        file_hash = source_file_hash(source, path)
        metadata = _metadata_record(file_hash, chunk_count=3)
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=True)

        _assert_skipped_without_ocr(result, ocr_fn, service, expected_hash=file_hash, chunk_count=3)
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()

    def test_unchanged_scanned_pdf_with_augmentor_ocrs_then_reindexes(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        ocr_doc = _ocr_applied_doc(loaded, "OCR text for reindex")
        file_hash = source_file_hash(source, path)
        metadata = _metadata_record(file_hash, chunk_count=1)
        augmentor = MagicMock()
        augmentor.augment.return_value = []
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(
            metadata=metadata,
            augmentor=augmentor,
        )
        result, ocr_fn = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=ocr_doc
        )

        ocr_fn.assert_called_once()
        service.prepare.assert_called_once_with(ocr_doc)
        vector_store.upsert.assert_called_once()
        bm25.add.assert_called_once()
        assert result.skipped is False
        assert result.content_hash == file_hash

    def test_ocr_failure_stores_pending_hash_not_file_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("tiny", source=source)
        pipeline, service, *_ = mock_ingestion_pipeline()
        result, _ = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=loaded
        )

        service.prepare.assert_called_once_with(loaded)
        assert result.content_hash == ocr_pending_hash(source, "tiny")
        assert result.content_hash != content_hash(source, "tiny")
        assert result.content_hash != source_file_hash(source, path)

    def test_disabling_ocr_preserves_file_keyed_scan(self, tmp_path: Path) -> None:
        """OCR off after a successful OCR ingested must not reindex empty text."""
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 scanned")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        file_hash = source_file_hash(source, path)
        metadata = _metadata_record(file_hash, chunk_count=5)
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=False)

        _assert_skipped_without_ocr(result, ocr_fn, service, expected_hash=file_hash, chunk_count=5)
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.add.assert_not_called()

    def test_disabling_ocr_with_augmentor_still_skips_file_keyed_scan(self, tmp_path: Path) -> None:
        """Augmentor reindex must not wipe OCR chunks when OCR is disabled."""
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        file_hash = source_file_hash(source, path)
        metadata = _metadata_record(file_hash, chunk_count=2)
        augmentor = MagicMock()
        pipeline, service, vector_store, _ = mock_ingestion_pipeline(
            metadata=metadata,
            augmentor=augmentor,
        )
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=False)

        _assert_skipped_without_ocr(result, ocr_fn, service, expected_hash=file_hash)
        vector_store.delete.assert_not_called()
        augmentor.augment.assert_not_called()

    def test_raising_min_chars_does_not_force_ocr_on_text_keyed_pdf(self, tmp_path: Path) -> None:
        path, loaded, _, text_hash = _text_keyed_born_digital_pdf(tmp_path)
        metadata = _metadata_record(text_hash, chunk_count=4)
        pipeline, service, vector_store, _ = mock_ingestion_pipeline(metadata=metadata)
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=True)

        _assert_skipped_without_ocr(result, ocr_fn, service, expected_hash=text_hash, chunk_count=4)
        vector_store.upsert.assert_not_called()

    @pytest.mark.parametrize(
        "enricher_attr",
        ("augmentor", "hype_indexer", "hierarchical_indexer"),
    )
    def test_raising_min_chars_with_llm_enricher_reindexes_text_keyed_without_ocr(
        self,
        tmp_path: Path,
        enricher_attr: str,
    ) -> None:
        """Full reindex on skip must not OCR overwrite text-keyed born-digital PDFs."""
        path, loaded, content, text_hash = _text_keyed_born_digital_pdf(tmp_path)
        metadata = _metadata_record(text_hash, chunk_count=2, chunk_ids=["c1", "c2"])
        enricher = MagicMock()
        if enricher_attr == "augmentor":
            enricher.augment.return_value = []
            pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(
                metadata=metadata,
                augmentor=enricher,
            )
        elif enricher_attr == "hype_indexer":
            enricher.index.return_value = []
            pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
            pipeline._hype_indexer = enricher  # noqa: SLF001
        else:
            pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
            prepared = list(service.prepare.return_value)
            enricher.index.return_value = (prepared, [])
            pipeline._hierarchical_indexer = enricher  # noqa: SLF001

        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=True)

        _assert_reindexed_without_ocr(
            result,
            ocr_fn,
            service,
            vector_store,
            content=content,
            expected_hash=text_hash,
        )
        bm25.add.assert_called_once()
        metadata.upsert_document.assert_called()
        assert metadata.upsert_document.call_args.args[1] == text_hash

    def test_text_keyed_with_augmentor_reindexes_without_ocr_when_not_candidate(
        self, tmp_path: Path
    ) -> None:
        path, loaded, content, text_hash = _text_keyed_born_digital_pdf(
            tmp_path,
            content=(
                "Born-digital PDF with plenty of extractable characters across the page content."
            ),
        )
        metadata = _metadata_record(text_hash, chunk_count=1)
        augmentor = MagicMock()
        augmentor.augment.return_value = []
        pipeline, service, vector_store, _ = mock_ingestion_pipeline(
            metadata=metadata,
            augmentor=augmentor,
        )
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=False)

        _assert_reindexed_without_ocr(
            result,
            ocr_fn,
            service,
            vector_store,
            content=content,
            expected_hash=text_hash,
        )
        augmentor.augment.assert_called_once()

    def test_enabling_ocr_recovers_empty_text_keyed_scan(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        ocr_doc = _ocr_applied_doc(loaded, "Recovered by OCR after enabling")
        metadata = _metadata_record(content_hash(source, ""), chunk_count=0)
        pipeline, service, *_ = mock_ingestion_pipeline(metadata=metadata)
        result, ocr_fn = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=ocr_doc
        )

        ocr_fn.assert_called_once()
        service.prepare.assert_called_once_with(ocr_doc)
        assert result.skipped is False
        assert result.content_hash == source_file_hash(source, path)

    def test_ocr_pending_hash_retries_on_next_ingest(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("tiny", source=source)
        ocr_doc = _ocr_applied_doc(loaded, "OCR works now")
        metadata = _metadata_record(ocr_pending_hash(source, "tiny"), chunk_count=0)
        pipeline, service, *_ = mock_ingestion_pipeline(metadata=metadata)
        result, ocr_fn = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=ocr_doc
        )

        ocr_fn.assert_called_once()
        service.prepare.assert_called_once_with(ocr_doc)
        assert result.content_hash == source_file_hash(source, path)

    def test_augmentor_ocr_failure_preserves_file_keyed_chunks(self, tmp_path: Path) -> None:
        """Failed OCR on unchanged file-keyed scan must not purge or pending-hash."""
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        file_hash = source_file_hash(source, path)
        metadata = _metadata_record(file_hash, chunk_count=3, chunk_ids=["c1", "c2", "c3"])
        augmentor = MagicMock()
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(
            metadata=metadata,
            augmentor=augmentor,
        )
        result, ocr_fn = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=loaded
        )

        ocr_fn.assert_called_once()
        service.prepare.assert_not_called()
        vector_store.delete.assert_not_called()
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()
        augmentor.augment.assert_not_called()
        assert result.skipped is True
        assert result.content_hash == file_hash
        assert result.chunk_count == 3

    def test_augmentor_empty_ocr_result_preserves_file_keyed_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 scanned")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        # OCR "ran" but returned the same empty document (no OCR_APPLIED_KEY).
        file_hash = source_file_hash(source, path)
        metadata = _metadata_record(file_hash, chunk_count=2)
        hype = MagicMock()
        pipeline, service, vector_store, _ = mock_ingestion_pipeline(metadata=metadata)
        pipeline._hype_indexer = hype  # noqa: SLF001
        result, _ = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=loaded
        )

        service.prepare.assert_not_called()
        vector_store.delete.assert_not_called()
        hype.index.assert_not_called()
        assert result.skipped is True
        assert result.content_hash == file_hash

    def test_skip_backfills_tables_for_empty_ocr_candidate_with_layout(
        self, tmp_path: Path
    ) -> None:
        """File-keyed empty scans still backfill when layout tables[] have text."""
        path, loaded, file_hash, table_id, table_chunk = _file_keyed_empty_scan_with_layout_table(
            tmp_path
        )
        metadata = _metadata_record(file_hash, chunk_count=1, chunk_ids=["text-1"])
        pipeline, service, vector_store, bm25, table_chunker = _pipeline_with_table_chunker(
            metadata=metadata,
            table_chunk=table_chunk,
        )
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=True)

        ocr_fn.assert_not_called()
        _assert_table_backfill(
            service=service,
            table_chunker=table_chunker,
            vector_store=vector_store,
            bm25=bm25,
            result=result,
            expected_hash=file_hash,
        )
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert merged_ids == ["text-1", table_id]

    def test_skip_empty_ocr_without_layout_does_not_purge_tables(self, tmp_path: Path) -> None:
        """Empty OCR loads without layout tables must not treat tables as removed."""
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4")
        source = str(path.resolve())
        loaded = _doc("", source=source, metadata={"pages": [""]})
        file_hash = source_file_hash(source, path)
        stale_table = table_chunk_id(source, "table-1")
        metadata = _metadata_record(file_hash, chunk_count=2, chunk_ids=["text-1", stale_table])
        table_chunker = MagicMock()
        table_chunker.index.return_value = []
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        pipeline._table_chunker = table_chunker
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=True)

        ocr_fn.assert_not_called()
        service.prepare.assert_not_called()
        table_chunker.index.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        assert result.skipped is True
        assert result.content_hash == file_hash

    def test_augmentor_ocr_failure_still_backfills_layout_tables(self, tmp_path: Path) -> None:
        """When OCR fails on file-keyed reindex, preserve text and still sync tables."""
        path, loaded, file_hash, _, table_chunk = _file_keyed_empty_scan_with_layout_table(tmp_path)
        metadata = _metadata_record(file_hash, chunk_count=1, chunk_ids=["ocr-text-1"])
        augmentor = MagicMock()
        pipeline, service, vector_store, bm25, table_chunker = _pipeline_with_table_chunker(
            metadata=metadata,
            table_chunk=table_chunk,
            augmentor=augmentor,
        )
        result, ocr_fn = _ingest_with_ocr_patches(
            pipeline, path, loaded, ocr_candidate=True, ocr_return=loaded
        )

        ocr_fn.assert_called_once()
        augmentor.augment.assert_not_called()
        vector_store.delete.assert_not_called()
        _assert_table_backfill(
            service=service,
            table_chunker=table_chunker,
            vector_store=vector_store,
            bm25=bm25,
            result=result,
            expected_hash=file_hash,
        )

    def test_disabling_ocr_with_augmentor_backfills_layout_tables(self, tmp_path: Path) -> None:
        path, loaded, file_hash, _, table_chunk = _file_keyed_empty_scan_with_layout_table(tmp_path)
        metadata = _metadata_record(file_hash, chunk_count=1, chunk_ids=["ocr-text-1"])
        augmentor = MagicMock()
        pipeline, service, vector_store, bm25, table_chunker = _pipeline_with_table_chunker(
            metadata=metadata,
            table_chunk=table_chunk,
            augmentor=augmentor,
        )
        result, ocr_fn = _ingest_with_ocr_patches(pipeline, path, loaded, ocr_candidate=False)

        ocr_fn.assert_not_called()
        augmentor.augment.assert_not_called()
        _assert_table_backfill(
            service=service,
            table_chunker=table_chunker,
            vector_store=vector_store,
            bm25=bm25,
            result=result,
            expected_hash=file_hash,
        )
