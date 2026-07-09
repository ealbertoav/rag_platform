"""T-202 — Structured table chunks at ingesting."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import (
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_TABLE,
    TABLE_ID_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.ingestion.table_chunker import (
    TableChunker,
    build_table_chunks,
    extract_markdown_tables,
    is_table_chunk,
    table_chunk_id,
)
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline, IngestionResult, content_hash
from tests.unit.ingestion_helpers import (
    embedded_chunk,
    mock_ingestion_pipeline,
    mock_reingest_metadata,
)

_TABLE_CHUNKER = "src.rag.ingestion.table_chunker"
_INGESTION_PIPELINE = "src.rag.pipelines.ingestion_pipeline"


def _internal(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


resolve_table_text = cast(
    Callable[[dict[str, Any], str | None], str | None],
    _internal(_TABLE_CHUNKER, "_resolve_table_text"),
)

content_table_for_id = cast(
    Callable[[str, list[str]], str | None],
    _internal(_TABLE_CHUNKER, "_content_table_for_id"),
)

build_table_chunker = cast(
    Callable[..., object | None],
    _internal(_INGESTION_PIPELINE, "_build_table_chunker"),
)

_SAMPLE_TABLE = "| A | B |\n|---|---|\n| 1 | 2 |"
_SAMPLE_TABLE_2 = "| X | Y |\n|---|---|\n| 3 | 4 |"


def _doc(
    *,
    content: str = "Body text.",
    tables: list[dict[str, Any]] | None = None,
    source: str = "/tmp/report.pdf",
) -> Document:
    metadata: dict[str, Any] = {"loader": "docling", "filename": "report.pdf"}
    if tables is not None:
        metadata["tables"] = tables
    return Document(source=source, content=content, metadata=metadata)


def _table_chunk(text: str = _SAMPLE_TABLE, table_id: str = "table-1") -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=text,
        metadata={
            CHUNK_TYPE_KEY: CHUNK_TYPE_TABLE,
            TABLE_ID_KEY: table_id,
            CHUNK_SOURCE_KEY: "/tmp/report.pdf",
        },
    )


def _report_pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def _embedded_table_chunk(document: Document) -> Chunk:
    table = build_table_chunks(document)[0]
    return table.model_copy(update={"embedding": [0.1] * 4, "sparse_vector": {1: 0.9}})


def _unchanged_hash_metadata(
    path: Path,
    document: Document,
    *,
    chunk_ids: list[str],
) -> MagicMock:
    metadata = MagicMock()
    metadata.get_by_source.return_value = MagicMock(
        id="doc-1",
        content_hash=content_hash(str(path.resolve()), document.content),
        chunk_count=1,
    )
    metadata.get_chunk_ids.return_value = chunk_ids
    return metadata


def _run_skip_table_chunker_ingest(
    path: Path,
    document: Document,
    *,
    table_chunker: MagicMock,
    chunk_ids: list[str],
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock]:
    metadata = _unchanged_hash_metadata(path, document, chunk_ids=chunk_ids)
    pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
    pipeline._table_chunker = table_chunker  # noqa: SLF001
    with patch(
        "src.rag.pipelines.ingestion_pipeline.load_document",
        return_value=document,
    ):
        result = pipeline.ingest_file(path)
    return result, service, vector_store, bm25, metadata


def _doc_with_sample_table(source: str) -> Document:
    return _doc(
        source=source,
        tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
    )


def _run_skip_with_indexed_table_chunks(
    path: Path,
    document: Document,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock]:
    table = _embedded_table_chunk(document)
    table_chunker = MagicMock()
    table_chunker.index.return_value = [table]
    result, service, vector_store, bm25, _ = _run_skip_table_chunker_ingest(
        path,
        document,
        table_chunker=table_chunker,
        chunk_ids=["text-chunk-1", table.id],
    )
    return result, service, vector_store, bm25


def _assert_skip_without_reindex(
    result: IngestionResult,
    service: MagicMock,
    vector_store: MagicMock,
    bm25: MagicMock,
) -> None:
    assert result.skipped is True
    service.prepare.assert_not_called()
    vector_store.upsert.assert_not_called()
    bm25.add.assert_not_called()


class TestIsTableChunk:
    def test_true_for_table_type(self) -> None:
        assert is_table_chunk(_table_chunk()) is True

    def test_false_for_text_chunk(self) -> None:
        assert is_table_chunk(embedded_chunk()) is False


class TestExtractMarkdownTables:
    def test_extracts_single_table(self) -> None:
        content = f"Intro\n\n{_SAMPLE_TABLE}\n\nOutro"
        assert extract_markdown_tables(content) == [_SAMPLE_TABLE]

    def test_extracts_multiple_tables_in_order(self) -> None:
        content = f"{_SAMPLE_TABLE}\n\n{_SAMPLE_TABLE_2}"
        assert extract_markdown_tables(content) == [_SAMPLE_TABLE, _SAMPLE_TABLE_2]

    def test_returns_empty_when_no_tables(self) -> None:
        assert extract_markdown_tables("plain paragraph") == []

    def test_ignores_blank_matches(self) -> None:
        assert extract_markdown_tables("") == []


class TestContentTableForId:
    def test_maps_table_id_to_nth_content_table(self) -> None:
        tables = [_SAMPLE_TABLE, _SAMPLE_TABLE_2]
        assert content_table_for_id("table-1", tables) == _SAMPLE_TABLE
        assert content_table_for_id("table-2", tables) == _SAMPLE_TABLE_2

    def test_returns_none_for_unknown_or_non_numeric_ids(self) -> None:
        assert content_table_for_id("table-99", [_SAMPLE_TABLE]) is None
        assert content_table_for_id("custom-id", [_SAMPLE_TABLE]) is None


class TestResolveTableText:
    def test_prefers_text_key(self) -> None:
        entry = {"text": " from metadata "}
        assert resolve_table_text(entry, "| fallback |") == "from metadata"

    def test_uses_markdown_key_when_text_missing(self) -> None:
        entry = {"markdown": " md table "}
        assert resolve_table_text(entry, None) == "md table"

    def test_falls_back_to_content_table(self) -> None:
        assert resolve_table_text({}, _SAMPLE_TABLE) == _SAMPLE_TABLE

    def test_returns_none_when_all_missing(self) -> None:
        assert resolve_table_text({}, None) is None
        assert resolve_table_text({"text": "   "}, "  ") is None


class TestBuildTableChunks:
    def test_table_chunk_ids_are_deterministic(self) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        first = build_table_chunks(document)
        second = build_table_chunks(document)
        assert len(first) == 1
        assert first[0].id == second[0].id
        assert first[0].id == table_chunk_id(document.source, "table-1")

    def test_table_chunk_ids_stable_across_document_reload(self) -> None:
        source = "/tmp/report.pdf"
        tables = [{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}]
        first_load = _doc(source=source, tables=tables)
        reloaded = _doc(source=source, tables=tables)
        assert first_load.id != reloaded.id
        assert build_table_chunks(first_load)[0].id == build_table_chunks(reloaded)[0].id

    def test_returns_empty_without_tables_metadata(self) -> None:
        assert build_table_chunks(_doc()) == []

    def test_returns_empty_for_empty_tables_list(self) -> None:
        assert build_table_chunks(_doc(tables=[])) == []

    def test_builds_chunk_from_metadata_text(self) -> None:
        document = _doc(
            tables=[
                {
                    TABLE_ID_KEY: "table-1",
                    "text": _SAMPLE_TABLE,
                    CHUNK_PAGE_KEY: 2,
                    BBOX_KEY: [1.0, 2.0, 3.0, 4.0],
                }
            ]
        )
        chunks = build_table_chunks(document)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == _SAMPLE_TABLE
        assert chunk.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_TABLE
        assert chunk.metadata[TABLE_ID_KEY] == "table-1"
        assert chunk.metadata[CHUNK_PAGE_KEY] == 2
        assert chunk.metadata[BBOX_KEY] == [1.0, 2.0, 3.0, 4.0]
        assert chunk.metadata[CHUNK_SOURCE_KEY] == document.source
        assert chunk.metadata["loader"] == "docling"
        assert "tables" not in chunk.metadata

    def test_falls_back_to_markdown_in_document_content(self) -> None:
        document = _doc(
            content=f"Intro\n\n{_SAMPLE_TABLE}",
            tables=[{TABLE_ID_KEY: "table-1"}],
        )
        chunks = build_table_chunks(document)
        assert len(chunks) == 1
        assert chunks[0].text == _SAMPLE_TABLE

    def test_fallback_uses_table_id_not_metadata_index(self) -> None:
        base = _doc(
            content=f"{_SAMPLE_TABLE}\n\n{_SAMPLE_TABLE_2}",
            tables=[{TABLE_ID_KEY: "table-2"}],
        )
        metadata = dict(base.metadata)
        metadata["tables"] = ["bad", metadata["tables"][0]]
        document = base.model_copy(update={"metadata": metadata})
        chunks = build_table_chunks(document)
        assert len(chunks) == 1
        assert chunks[0].metadata[TABLE_ID_KEY] == "table-2"
        assert chunks[0].text == _SAMPLE_TABLE_2

    def test_skips_non_dict_entries(self, caplog: pytest.LogCaptureFixture) -> None:
        base = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        metadata = dict(base.metadata)
        metadata["tables"] = ["bad", metadata["tables"][0]]
        document = base.model_copy(update={"metadata": metadata})
        with caplog.at_level(logging.DEBUG, logger=_TABLE_CHUNKER):
            chunks = build_table_chunks(document)
        assert len(chunks) == 1
        assert "Skipping non-dict" in caplog.text

    def test_skips_entries_without_table_id(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(tables=[{"text": _SAMPLE_TABLE}])
        with caplog.at_level(logging.DEBUG, logger=_TABLE_CHUNKER):
            chunks = build_table_chunks(document)
        assert chunks == []
        assert TABLE_ID_KEY in caplog.text

    def test_skips_entries_without_text(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-9"}])
        with caplog.at_level(logging.WARNING, logger=_TABLE_CHUNKER):
            chunks = build_table_chunks(document)
        assert chunks == []
        assert "No text for table table-9" in caplog.text


class TestTableChunker:
    @staticmethod
    def _chunker(chunks: list[Chunk] | None = None) -> TableChunker:
        embedder = MagicMock()
        embedder.embed_both.return_value = (
            [[0.1] * 4 for _ in (chunks or [_table_chunk()])],
            [{1: 0.9} for _ in (chunks or [_table_chunk()])],
        )
        return TableChunker(embedder=embedder)  # type: ignore[arg-type]

    def test_index_returns_embedded_chunks(self) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        chunker = self._chunker()
        result = chunker.index(document)
        assert len(result) == 1
        assert result[0].embedding is not None
        assert result[0].sparse_vector is not None
        chunker._embedder.embed_both.assert_called_once()  # type: ignore[attr-defined]

    def test_index_returns_empty_when_no_tables(self) -> None:
        chunker = self._chunker()
        assert chunker.index(_doc()) == []
        chunker._embedder.embed_both.assert_not_called()  # type: ignore[attr-defined]

    def test_index_returns_empty_on_embedding_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        embedder = MagicMock()
        embedder.embed_both.side_effect = RuntimeError("embed failed")
        chunker = TableChunker(embedder=embedder)  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger=_TABLE_CHUNKER):
            assert chunker.index(document) == []
        assert "Embedding table chunks failed" in caplog.text


class TestBuildTableChunker:
    def test_returns_none_when_disabled(self) -> None:
        assert build_table_chunker(MagicMock(), type("Cfg", (), {"enabled": False})()) is None

    def test_returns_chunker_when_enabled(self) -> None:
        embedder = MagicMock(spec=EmbeddingRepository)
        chunker = build_table_chunker(embedder, type("Cfg", (), {"enabled": True})())
        assert isinstance(chunker, TableChunker)


class TestIngestionPipelineTableChunks:
    def test_table_chunks_indexed_in_qdrant_and_bm25(self, tmp_path: Path) -> None:
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF-1.4")
        base = embedded_chunk(0)
        table = _table_chunk()
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline([base])
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=_doc(source=str(path.resolve())),
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 1
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 2
        assert any(is_table_chunk(c) for c in upserted)
        bm25_added = bm25.add.call_args.args[0]
        assert any(is_table_chunk(c) for c in bm25_added)

    def test_table_only_document_indexes_without_text_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "tables-only.pdf"
        path.write_bytes(b"%PDF-1.4")
        table = _table_chunk()
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(prepared_chunks=[])
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=_doc(source=str(path.resolve())),
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 0
        service.prepare.assert_called_once()
        table_chunker.index.assert_called_once()
        vector_store.upsert.assert_called_once()
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert is_table_chunk(upserted[0])
        bm25.add.assert_called_once()

    def test_table_only_reingest_purges_old_chunks_and_indexes_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "tables-only.pdf"
        path.write_bytes(b"%PDF-1.4")
        table = _table_chunk()
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        metadata = mock_reingest_metadata()
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[],
            metadata=metadata,
        )
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=_doc(source=str(path.resolve())),
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 0
        vector_store.upsert.assert_called_once()
        vector_store.delete.assert_called_once_with(["old-chunk-1"])
        bm25.remove_by_ids.assert_called_once_with(["old-chunk-1"])
        metadata.upsert_document.assert_called_once()

    def test_table_chunker_none_skips_indexing(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("hello")
        pipeline, _, vector_store, _ = mock_ingestion_pipeline()
        result = pipeline.ingest_file(path)
        assert result.chunk_count == 1
        assert len(vector_store.upsert.call_args.args[0]) == 1

    def test_from_settings_wires_table_chunker_when_enabled(self) -> None:
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.rag.chunking.get_chunker"),
            patch("src.infrastructure.embeddings.get_embedding_provider") as get_embedder,
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.infrastructure.metadata.sqlite_store.SQLiteMetadataStore.from_settings"),
        ):
            mock_settings.chunking = MagicMock(
                strategy="recursive",
                chunk_size=512,
                overlap=64,
                contextual_headers=MagicMock(enabled=False),
                augmentation=MagicMock(enabled=False),
                hierarchical=MagicMock(enabled=False),
            )
            mock_settings.metadata = MagicMock(enabled=False)
            mock_settings.neo4j = MagicMock(enabled=False)
            mock_settings.retrieval = MagicMock(hype=MagicMock(enabled=False))
            mock_settings.parsing = MagicMock(table_chunks=MagicMock(enabled=True))
            get_embedder.return_value = MagicMock()
            pipeline = IngestionPipeline.from_settings()

        assert isinstance(pipeline._table_chunker, TableChunker)  # noqa: SLF001

    def test_skip_backfills_missing_table_chunks_on_unchanged_hash(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(
            source=str(path.resolve()),
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        table = _embedded_table_chunk(document)
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        result, service, vector_store, bm25, metadata = _run_skip_table_chunker_ingest(
            path,
            document,
            table_chunker=table_chunker,
            chunk_ids=["text-chunk-1"],
        )

        assert result.skipped is False
        service.prepare.assert_not_called()
        table_chunker.index.assert_called_once()
        indexed_document = table_chunker.index.call_args.args[0]
        assert indexed_document.id == "doc-1"
        vector_store.upsert.assert_called_once()
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert is_table_chunk(upserted[0])
        bm25.add.assert_called_once()
        metadata.upsert_document.assert_called_once()
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert merged_ids == ["text-chunk-1", table.id]

    def test_skip_unchanged_without_table_chunker(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()))
        metadata = _unchanged_hash_metadata(path, document, chunk_ids=["text-chunk-1"])
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        pipeline._table_chunker = None  # noqa: SLF001
        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=document,
        ):
            result = pipeline.ingest_file(path)

        assert result.skipped is True
        _assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_skips_when_table_chunks_already_indexed(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        result, service, vector_store, bm25 = _run_skip_with_indexed_table_chunks(
            path,
            _doc_with_sample_table(str(path.resolve())),
        )
        _assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_skips_when_table_chunks_already_indexed_after_reload(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        first_load = _doc_with_sample_table(source)
        reloaded = _doc_with_sample_table(source)
        assert first_load.id != reloaded.id
        result, service, vector_store, bm25 = _run_skip_with_indexed_table_chunks(
            path,
            reloaded,
        )
        _assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_backfill_noop_when_table_chunker_returns_empty(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()))
        table_chunker = MagicMock()
        table_chunker.index.return_value = []
        result, service, vector_store, bm25, _ = _run_skip_table_chunker_ingest(
            path,
            document,
            table_chunker=table_chunker,
            chunk_ids=["text-chunk-1"],
        )

        assert result.skipped is True
        _assert_skip_without_reindex(result, service, vector_store, bm25)
