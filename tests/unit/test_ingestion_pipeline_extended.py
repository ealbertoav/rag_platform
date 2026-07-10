"""Extended ingestion pipeline tests for Phase 11 coverage."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.domain.repositories.metadata_repository import DocumentRecord
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline
from tests.unit.ingestion_helpers import (
    embedded_chunk,
    mock_ingestion_pipeline,
    mock_reingest_metadata,
    write_reingest_doc,
)


class TestDiscoverViaIngestDirectory:
    def test_unsupported_files_only_returns_empty(self, tmp_path: Path):
        (tmp_path / "data.bin").write_bytes(b"\x00")
        pipeline, *_ = mock_ingestion_pipeline()
        assert pipeline.ingest_directory(tmp_path) == []

    def test_supported_file_is_discovered(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text("# Hi")
        pipeline, *_ = mock_ingestion_pipeline()
        results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 1


class TestIngestionMetadataAndReingest:
    def test_reingest_changed_hash_removes_old_chunks(self, tmp_path: Path):
        path = write_reingest_doc(tmp_path, content="version one")
        metadata = mock_reingest_metadata()
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        path.write_text("version two")
        result = pipeline.ingest_file(path)
        assert result.skipped is False
        assert result.source == str(path.resolve())
        vector_store.delete.assert_called_once_with(["old-chunk-1"])
        bm25.remove_by_ids.assert_called_once_with(["old-chunk-1"])
        metadata.upsert_document.assert_called_once()

    def test_reingest_empty_chunks_purges_stale_bm25(self, tmp_path: Path):
        metadata = mock_reingest_metadata()
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[], metadata=metadata
        )
        result = pipeline.ingest_file(write_reingest_doc(tmp_path))
        assert result.chunk_count == 0
        vector_store.delete.assert_called_once_with(["old-chunk-1"])
        bm25.remove_by_ids.assert_called_once_with(["old-chunk-1"])
        bm25.add.assert_not_called()
        vector_store.upsert.assert_not_called()

    def test_reingest_prepare_failure_preserves_stale_chunks(self, tmp_path: Path):
        metadata = mock_reingest_metadata()
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        service.prepare.side_effect = RuntimeError("chunker failed")
        with pytest.raises(RuntimeError, match="chunker failed"):
            pipeline.ingest_file(write_reingest_doc(tmp_path))
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()

    def test_reingest_upsert_failure_preserves_stale_chunks(self, tmp_path: Path):
        metadata = mock_reingest_metadata()
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        vector_store.upsert.side_effect = RuntimeError("qdrant down")
        with pytest.raises(RuntimeError, match="qdrant down"):
            pipeline.ingest_file(write_reingest_doc(tmp_path))
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()

    def test_empty_chunks_records_metadata(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        metadata = MagicMock()
        metadata.get_by_source.return_value = None
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[], metadata=metadata
        )
        result = pipeline.ingest_file(path)
        assert result.chunk_count == 0
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()
        metadata.upsert_document.assert_called_once()

    def test_successful_ingest_records_metadata(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        metadata = MagicMock()
        metadata.get_by_source.return_value = None
        chunks = [embedded_chunk(0), embedded_chunk(1)]
        pipeline, _, _, _ = mock_ingestion_pipeline(prepared_chunks=chunks, metadata=metadata)
        pipeline.ingest_file(path)
        args, kwargs = metadata.upsert_document.call_args
        assert args[2] == [c.id for c in chunks]
        assert kwargs["chunk_count"] == len(chunks)

    def test_list_documents_without_metadata_returns_empty(self):
        pipeline, *_ = mock_ingestion_pipeline(metadata=None)
        assert pipeline.list_documents() == []

    def test_list_documents_delegates_to_metadata(self):
        metadata = MagicMock()
        record = DocumentRecord(
            id="d1",
            source_path="/tmp/a.md",
            content_hash="abc",
            chunk_count=2,
            ingested_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        metadata.list_documents.return_value = [record]
        pipeline, *_ = mock_ingestion_pipeline(metadata=metadata)
        assert pipeline.list_documents() == [record]


class TestGraphIndexing:
    def test_index_graph_calls_indexer(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        graph = MagicMock()
        pipeline, _, _, _ = mock_ingestion_pipeline(graph_indexer=graph)
        pipeline.ingest_file(path)
        graph.index_chunks.assert_called_once()

    def test_remove_document_chunks_without_metadata_is_noop(self):
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(metadata=None)
        pipeline._remove_document_chunks("doc-id")  # noqa: SLF001
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()

    def test_remove_document_chunks_deletes_from_vector_store_and_bm25(self):
        metadata = MagicMock()
        metadata.get_chunk_ids.return_value = ["old-chunk-1", "old-chunk-2"]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        pipeline._remove_document_chunks("doc-meta-1")  # noqa: SLF001
        metadata.get_chunk_ids.assert_called_once_with("doc-meta-1")
        vector_store.delete.assert_called_once_with(["old-chunk-1", "old-chunk-2"])
        bm25.remove_by_ids.assert_called_once_with(["old-chunk-1", "old-chunk-2"])

    def test_purge_superseded_chunks_skips_retained_ids(self):
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline()
        pipeline._purge_superseded_chunks(  # noqa: SLF001
            ["keep-me", "remove-me"],
            retained_chunk_ids={"keep-me"},
        )
        vector_store.delete.assert_called_once_with(["remove-me"])
        bm25.remove_by_ids.assert_called_once_with(["remove-me"])

    def test_purge_superseded_chunks_noop_when_all_retained(self):
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline()
        pipeline._purge_superseded_chunks(  # noqa: SLF001
            ["stable-table-id"],
            retained_chunk_ids={"stable-table-id"},
        )
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()

    def test_purge_superseded_chunks_noop_for_empty_input(self):
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline()
        pipeline._purge_superseded_chunks([])  # noqa: SLF001
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()

    def test_index_graph_failure_does_not_abort_ingest(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        graph = MagicMock()
        graph.index_chunks.side_effect = RuntimeError("neo4j down")
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(graph_indexer=graph)
        result = pipeline.ingest_file(path)
        assert result.chunk_count == 1
        vector_store.upsert.assert_called_once()
        bm25.add.assert_called_once()


class TestBuildGraphIndexerViaFromSettings:
    def test_graph_indexer_none_when_extract_disabled(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.rag.chunking.get_chunker"),
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.infrastructure.metadata.sqlite_store.SQLiteMetadataStore.from_settings"),
        ):
            mock_settings.chunking = MagicMock(strategy="recursive", chunk_size=512, overlap=64)
            mock_settings.metadata = MagicMock(enabled=False)
            mock_settings.neo4j = MagicMock(enabled=True, extract_entities_on_ingest=False)
            pipeline = IngestionPipeline.from_settings()
            assert pipeline._graph_indexer is None  # noqa: SLF001

    def test_graph_indexer_created_when_enabled(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.rag.chunking.get_chunker"),
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.infrastructure.metadata.sqlite_store.SQLiteMetadataStore.from_settings"),
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch("src.infrastructure.vectordb.neo4j_graph.Neo4jGraphRepository.from_settings"),
        ):
            mock_settings.chunking = MagicMock(strategy="recursive", chunk_size=512, overlap=64)
            mock_settings.metadata = MagicMock(enabled=False)
            mock_settings.neo4j = MagicMock(enabled=True, extract_entities_on_ingest=True)
            mock_llm.return_value = MagicMock()
            from src.rag.ingestion.graph_indexer import GraphIndexer

            pipeline = IngestionPipeline.from_settings()
            assert isinstance(pipeline._graph_indexer, GraphIndexer)  # noqa: SLF001

    def test_graph_indexer_none_on_llm_failure(self, caplog):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.rag.chunking.get_chunker"),
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.infrastructure.metadata.sqlite_store.SQLiteMetadataStore.from_settings"),
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings",
                side_effect=RuntimeError("no model"),
            ),
            caplog.at_level(logging.WARNING, logger="src.rag.pipelines.ingestion_pipeline"),
        ):
            mock_settings.chunking = MagicMock(strategy="recursive", chunk_size=512, overlap=64)
            mock_settings.metadata = MagicMock(enabled=False)
            mock_settings.neo4j = MagicMock(enabled=True, extract_entities_on_ingest=True)
            pipeline = IngestionPipeline.from_settings()
            assert pipeline._graph_indexer is None  # noqa: SLF001
            assert "Graph indexer unavailable" in caplog.text


class TestIngestionFromSettingsNeo4j:
    def test_from_settings_with_neo4j_enabled(self):
        graph_mock = MagicMock()
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.rag.chunking.get_chunker"),
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.infrastructure.metadata.sqlite_store.SQLiteMetadataStore.from_settings"),
            patch(
                "src.rag.pipelines.ingestion_pipeline._build_graph_indexer",
                return_value=graph_mock,
            ) as mock_graph,
        ):
            mock_settings.chunking = MagicMock(strategy="recursive", chunk_size=512, overlap=64)
            mock_settings.metadata = MagicMock(enabled=True)
            mock_settings.neo4j = MagicMock(enabled=True)
            pipeline = IngestionPipeline.from_settings()
            mock_graph.assert_called_once()
            assert pipeline._graph_indexer is graph_mock  # noqa: SLF001
