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
)
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline
from tests.unit.ingestion_helpers import embedded_chunk, mock_ingestion_pipeline

_TABLE_CHUNKER = "src.rag.ingestion.table_chunker"
_INGESTION_PIPELINE = "src.rag.pipelines.ingestion_pipeline"


def _internal(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


resolve_table_text = cast(
    Callable[[dict[str, Any], str | None], str | None],
    _internal(_TABLE_CHUNKER, "_resolve_table_text"),
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
