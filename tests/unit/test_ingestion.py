"""T-015 unit tests — IngestionService and IngestionPipeline (all deps mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import IngestionError
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.services.ingestion_service import IngestionService
from src.infrastructure.vectordb.bm25 import BM25Index
from src.infrastructure.vectordb.bm25_disk import DiskBM25Index
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline, IngestionResult, content_hash
from tests.unit.ingestion_helpers import (
    disk_bm25_index,
    embedded_chunk,
    index_reingest_corpus,
    ingest_with_hierarchical_and_hype_indexers,
    memory_bm25_index,
    mock_ingestion_pipeline,
    mock_reingest_metadata,
    real_bm25_reingest_pipeline,
    reingest_corpus_chunks,
    reingest_fresh_chunk,
    vector_store_with_upsert_failure,
    write_reingest_doc,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _doc(content: str = "hello world " * 20, source: str = "test.md") -> Document:
    return Document(source=source, content=content)


def _chunk(i: int = 0) -> Chunk:
    return Chunk(document_id="doc-1", text=f"chunk {i}")


# ── content hash (tested through IngestionResult) ─────────────────────────────


class TestContentHashBehaviour:
    def test_hash_is_16_char_hex(self):
        assert len(content_hash("/tmp/doc.md", "some content")) == 16
        assert all(c in "0123456789abcdef" for c in content_hash("/tmp/doc.md", "some content"))

    def test_same_content_same_source_same_hash(self):
        text = "identical content"
        source = "/tmp/a.md"
        assert content_hash(source, text) == content_hash(source, text)

    def test_same_content_different_source_different_hash(self):
        text = "identical content"
        assert content_hash("/tmp/a.md", text) != content_hash("/tmp/b.md", text)

    def test_different_content_different_hash(self):
        assert content_hash("/tmp/a.md", "content A") != content_hash("/tmp/a.md", "content B")

    def test_hash_on_ingest_result(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("some content")
        pipeline, *_ = mock_ingestion_pipeline()
        result = pipeline.ingest_file(path)
        assert result.content_hash == content_hash(str(path.resolve()), "some content")


# ── IngestionService ───────────────────────────────────────────────────────────


class TestIngestionService:
    @staticmethod
    def _service(chunks: list[Chunk] | None = None) -> IngestionService:
        chunker = MagicMock()
        chunker.chunk.return_value = chunks if chunks is not None else [_chunk(0), _chunk(1)]
        embedder = MagicMock()
        embedder.embed_both.return_value = (
            [[0.1] * 4, [0.2] * 4],
            [{1: 0.9}, {2: 0.8}],
        )
        return IngestionService(chunker=chunker, embedder=embedder)  # type: ignore[arg-type]

    def test_returns_list_of_chunks(self):
        result = self._service().prepare(_doc())
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunk_count_matches(self):
        svc = self._service([_chunk(i) for i in range(3)])
        svc._embedder.embed_both.return_value = (  # type: ignore[attr-defined]
            [[0.0] * 4] * 3,
            [{i: 0.9} for i in range(3)],
        )
        assert len(svc.prepare(_doc())) == 3

    def test_embeddings_attached(self):
        chunks = self._service().prepare(_doc())
        assert all(c.embedding is not None for c in chunks)
        assert all(c.sparse_vector is not None for c in chunks)

    def test_embed_both_called_once(self):
        svc = self._service()
        svc.prepare(_doc())
        svc._embedder.embed_both.assert_called_once()  # type: ignore[attr-defined]

    def test_empty_document_returns_empty(self):
        svc = self._service(chunks=[])
        assert svc.prepare(_doc(content="")) == []

    def test_embedding_failure_returns_empty_and_logs(self, caplog):
        svc = self._service()
        svc._embedder.embed_both.side_effect = RuntimeError("GPU OOM")  # type: ignore[attr-defined]
        import logging

        with caplog.at_level(logging.ERROR):
            result = svc.prepare(_doc())
        assert result == []
        assert "Embedding failed" in caplog.text


# ── IngestionPipeline ──────────────────────────────────────────────────────────


class TestIngestionPipelineFile:
    def test_returns_ingestion_result(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Hello\n\nSome content here.")
        pipeline, *_ = mock_ingestion_pipeline()
        result = pipeline.ingest_file(path)
        assert isinstance(result, IngestionResult)

    def test_chunk_count_in_result(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Hello\n\nSome content here.")
        pipeline, *_ = mock_ingestion_pipeline([embedded_chunk(0), embedded_chunk(1)])
        result = pipeline.ingest_file(path)
        assert result.chunk_count == 2

    def test_content_hash_set(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("some content")
        pipeline, *_ = mock_ingestion_pipeline()
        result = pipeline.ingest_file(path)
        assert len(result.content_hash) == 16

    def test_vector_store_upsert_called(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        pipeline, _, vector_store, _ = mock_ingestion_pipeline()
        pipeline.ingest_file(path)
        vector_store.upsert.assert_called_once()

    def test_bm25_add_called(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        pipeline, _, _, bm25 = mock_ingestion_pipeline()
        pipeline.ingest_file(path)
        bm25.add.assert_called_once()

    def test_augmentor_indexes_source_and_question_chunks(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        source = embedded_chunk(0)
        question = embedded_chunk(1)
        augmentor = MagicMock()
        augmentor.augment.return_value = [question]
        metadata = MagicMock()
        pipeline, _, vector_store, _ = mock_ingestion_pipeline(
            [source], augmentor=augmentor, metadata=metadata
        )
        result = pipeline.ingest_file(path)
        augmentor.augment.assert_called_once_with([source])
        indexed = vector_store.upsert.call_args.args[0]
        assert len(indexed) == 2
        assert indexed[0].id == source.id
        assert indexed[1].id == question.id
        assert result.chunk_count == 1  # source chunks only in result
        metadata.upsert_document.assert_called_once()
        _, kwargs = metadata.upsert_document.call_args
        assert kwargs["chunk_count"] == 1

    def test_hierarchical_and_hype_indexers_extend_upsert(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        upserted, summary, hype = ingest_with_hierarchical_and_hype_indexers(path)
        assert any(c.id == summary.id for c in upserted)
        assert any(c.id == hype.id for c in upserted)

    def test_no_upsert_when_no_chunks(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(prepared_chunks=[])
        result = pipeline.ingest_file(path)
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()
        assert result.chunk_count == 0

    def test_reingest_empty_chunks_purges_real_bm25(self, tmp_path: Path):
        from src.infrastructure.vectordb.bm25 import BM25Index

        _, _, stale = reingest_corpus_chunks()
        bm25 = BM25Index()
        bm25.index([stale])
        pipeline, _, vector_store, _ = real_bm25_reingest_pipeline(
            bm25,
            prepared_chunks=[],
            metadata=mock_reingest_metadata(),
        )
        result = pipeline.ingest_file(write_reingest_doc(tmp_path))
        assert result.chunk_count == 0
        vector_store.delete.assert_called_once_with(["old-chunk-1"])
        assert bm25.get_by_id("old-chunk-1") is None
        assert bm25.search("kubernetes", top_k=1) == []

    @pytest.mark.parametrize(
        "bm25_factory",
        [memory_bm25_index, disk_bm25_index],
        ids=["memory", "disk"],
    )
    def test_reingest_upsert_failure_keeps_indexes_consistent(self, tmp_path: Path, bm25_factory):
        bm25 = bm25_factory(tmp_path)
        index_reingest_corpus(bm25)
        vector_store = vector_store_with_upsert_failure()
        pipeline, _, _, _ = real_bm25_reingest_pipeline(
            bm25,
            prepared_chunks=[reingest_fresh_chunk()],
            vector_store=vector_store,
        )
        with pytest.raises(RuntimeError, match="qdrant down"):
            pipeline.ingest_file(write_reingest_doc(tmp_path))
        vector_store.delete.assert_not_called()
        assert bm25.get_by_id("old-chunk-1") is not None
        assert bm25.size == 3

    def test_reingest_success_purges_both_indexes_together(self, tmp_path: Path):
        from src.infrastructure.vectordb.bm25 import BM25Index

        fresh = Chunk(
            id="new-chunk-1",
            document_id="doc-1",
            text="fresh vector database content",
            embedding=[0.1] * 4,
        )
        bm25 = BM25Index()
        index_reingest_corpus(bm25)
        pipeline, _, vector_store, _ = real_bm25_reingest_pipeline(
            bm25,
            prepared_chunks=[fresh],
        )
        result = pipeline.ingest_file(write_reingest_doc(tmp_path))
        assert result.chunk_count == 1
        vector_store.upsert.assert_called_once()
        vector_store.delete.assert_called_once_with(["old-chunk-1"])
        assert bm25.get_by_id("old-chunk-1") is None
        assert bm25.get_by_id("new-chunk-1") is not None
        assert bm25.size == 3
        assert not any(chunk.id == "old-chunk-1" for chunk, _ in bm25.search("kubernetes", top_k=5))
        assert any(
            chunk.id == "new-chunk-1" for chunk, _ in bm25.search("vector database", top_k=5)
        )

    def test_load_error_raises_ingestion_error(self, tmp_path: Path):
        path = tmp_path / "ghost.pdf"
        pipeline, *_ = mock_ingestion_pipeline()
        with pytest.raises(IngestionError):
            pipeline.ingest_file(path)

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = tmp_path / "file.xyz"
        path.write_text("content")
        pipeline, *_ = mock_ingestion_pipeline()
        with pytest.raises(IngestionError):
            pipeline.ingest_file(path)

    def test_layout_parser_misconfiguration_raises_ingestion_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.core.settings import Settings
        from src.infrastructure.parsers import clear_layout_parser_cache

        clear_layout_parser_cache()
        monkeypatch.setattr(
            "src.infrastructure.loaders._settings",
            lambda: Settings(parsing={"layout_parser": {"enabled": True, "provider": "unknown"}}),
        )
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF-1.4")
        pipeline, *_ = mock_ingestion_pipeline()
        with pytest.raises(IngestionError, match="Cannot load report.pdf"):
            pipeline.ingest_file(path)


class TestIngestionPipelineDirectory:
    def test_returns_results_for_each_file(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}\n\nContent {i}.")
        pipeline, *_ = mock_ingestion_pipeline()
        results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 3

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        pipeline, *_ = mock_ingestion_pipeline()
        assert pipeline.ingest_directory(tmp_path) == []

    def test_failed_file_recorded_not_raised(self, tmp_path: Path):
        (tmp_path / "good.md").write_text("# Good\n\nContent.")
        (tmp_path / "bad.xyz").write_text("unsupported")
        pipeline, _, _, _ = mock_ingestion_pipeline()
        # Inject a failure for the good file by making service raise
        pipeline._service.prepare.side_effect = [RuntimeError("boom"), [embedded_chunk()]]
        results = pipeline.ingest_directory(tmp_path)
        errors = [r for r in results if r.error is not None]
        assert len(errors) >= 1

    def test_ignores_unsupported_extensions(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        pipeline, *_ = mock_ingestion_pipeline()
        results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 1

    def test_save_indexes_calls_bm25_save(self):
        pipeline, _, _, bm25 = mock_ingestion_pipeline()
        pipeline.save_indexes()
        bm25.save.assert_called_once()

    def test_ingest_file_rebuilds_bm25_once(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Doc\n\nContent.")
        bm25 = BM25Index()
        service = MagicMock()
        service.prepare.return_value = [embedded_chunk()]
        pipeline = IngestionPipeline(
            service=service,
            vector_store=MagicMock(),
            bm25=bm25,
        )
        with patch.object(bm25, "_rebuild", wraps=bm25._rebuild) as mock_rebuild:
            pipeline.ingest_file(path)
        assert mock_rebuild.call_count == 1

    def test_ingest_file_disk_flushes_bm25_once(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Doc\n\nContent.")
        bm25 = DiskBM25Index(tmp_path / "bm25_disk", segment_size=2)
        service = MagicMock()
        service.prepare.return_value = [embedded_chunk()]
        pipeline = IngestionPipeline(
            service=service,
            vector_store=MagicMock(),
            bm25=bm25,
        )
        with patch.object(bm25, "_flush_to_disk", wraps=bm25._flush_to_disk) as mock_flush:
            pipeline.ingest_file(path)
        assert mock_flush.call_count == 1

    def test_ingest_directory_rebuilds_bm25_once(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}\n\nContent {i}.")
        bm25 = BM25Index()
        service = MagicMock()
        service.prepare.return_value = [embedded_chunk()]
        pipeline = IngestionPipeline(
            service=service,
            vector_store=MagicMock(),
            bm25=bm25,
        )
        with patch.object(bm25, "_rebuild", wraps=bm25._rebuild) as mock_rebuild:
            results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 3
        assert mock_rebuild.call_count == 1

    def test_ingest_directory_searchable_during_batch(self, tmp_path: Path):
        """Shared BM25 index stays searchable while directory ingest defers rebuilds."""
        import threading

        (tmp_path / "doc0.md").write_text("# Doc 0\n\nkubernetes scheduling.")
        (tmp_path / "doc1.md").write_text("# Doc 1\n\nvector database indexing.")
        bm25 = BM25Index()
        bm25.index([Chunk(document_id="baseline", text="baseline corpus for bm25 idf")])
        service = MagicMock()
        service.prepare.side_effect = [
            [Chunk(document_id="doc-0", text="kubernetes scheduling.", embedding=[0.1] * 4)],
            [Chunk(document_id="doc-1", text="vector database indexing.", embedding=[0.2] * 4)],
        ]
        pipeline = IngestionPipeline(
            service=service,
            vector_store=MagicMock(),
            bm25=bm25,
        )
        search_ready = threading.Event()
        added = threading.Event()
        found = threading.Event()
        original_ingest_file = pipeline.ingest_file

        def ingest_file_with_sync(path: Path) -> IngestionResult:
            if path.name == "doc1.md":
                search_ready.wait(timeout=5)
            result = original_ingest_file(path)
            if path.name == "doc1.md":
                added.set()
            return result

        pipeline.ingest_file = ingest_file_with_sync  # type: ignore[method-assign]

        def search_while_ingesting() -> None:
            search_ready.set()
            added.wait(timeout=5)
            if bm25.search("vector database", top_k=1):
                found.set()

        search_thread = threading.Thread(target=search_while_ingesting)
        search_thread.start()
        pipeline.ingest_directory(tmp_path)
        search_thread.join(timeout=5)
        assert found.is_set()
        assert bm25.search("kubernetes scheduling", top_k=1)

    def test_skips_unchanged_file_when_metadata_matches(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("stable content")
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=content_hash(str(path.resolve()), "stable content"),
            chunk_count=2,
        )
        metadata.get_chunk_ids.return_value = ["c1", "c2"]
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        result = pipeline.ingest_file(path)
        assert result.skipped is True
        service.prepare.assert_not_called()
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()
        metadata.upsert_document.assert_called_once()
        _, kwargs = metadata.upsert_document.call_args
        assert kwargs["chunk_count"] == 2

    def test_skip_preserves_source_chunk_count_with_augmented_ids(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("stable content")
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=content_hash(str(path.resolve()), "stable content"),
            chunk_count=1,
        )
        metadata.get_chunk_ids.return_value = ["c1", "synthetic-q1", "synthetic-q2"]
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        result = pipeline.ingest_file(path)
        assert result.skipped is True
        assert result.chunk_count == 1
        service.prepare.assert_not_called()
        vector_store.upsert.assert_not_called()
        _, kwargs = metadata.upsert_document.call_args
        assert kwargs["chunk_count"] == 1


class TestIngestionPipelineFromSettings:
    def test_from_settings_builds_pipeline(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.rag.chunking.get_chunker") as mock_chunker,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
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
            )
            mock_settings.metadata = MagicMock(enabled=True)
            mock_settings.neo4j = MagicMock(enabled=False)
            pipeline = IngestionPipeline.from_settings()

        assert isinstance(pipeline, IngestionPipeline)
        mock_chunker.assert_called_once_with(
            "recursive",
            use_contextual_headers=False,
            chunk_size=512,
            overlap=64,
        )


# ── embed_both default ─────────────────────────────────────────────────────────


class TestEmbedBothDefault:
    def test_default_calls_embed_and_embed_sparse(self):
        from src.domain.repositories.embedding_repository import EmbeddingRepository

        class _Minimal(EmbeddingRepository):
            def embed(self, texts):
                return [[0.1] for _ in texts]

            def embed_sparse(self, texts):
                return [{1: 0.5} for _ in texts]

        repo = _Minimal()
        dense, sparse = repo.embed_both(["hello", "world"])
        assert len(dense) == 2
        assert len(sparse) == 2
