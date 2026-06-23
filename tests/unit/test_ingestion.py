"""T-015 unit tests — IngestionService and IngestionPipeline (all deps mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.exceptions import IngestionError
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.services.ingestion_service import IngestionService
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline, IngestionResult

# ── helpers ────────────────────────────────────────────────────────────────────


def _doc(content: str = "hello world " * 20, source: str = "test.md") -> Document:
    return Document(source=source, content=content)


def _chunk(i: int = 0) -> Chunk:
    return Chunk(document_id="doc-1", text=f"chunk {i}")


def _embedded_chunk(i: int = 0) -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=f"chunk {i}",
        embedding=[float(i)] * 4,
        sparse_vector={i + 1: 0.9},
    )


# ── content hash (tested through IngestionResult) ─────────────────────────────


class TestContentHashBehaviour:
    def test_hash_is_16_char_hex(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("some content")
        pipeline, *_ = _pipeline()
        result = pipeline.ingest_file(path)
        assert len(result.content_hash) == 16
        assert all(c in "0123456789abcdef" for c in result.content_hash)

    def test_same_content_same_hash(self, tmp_path: Path):
        a, b = tmp_path / "a.md", tmp_path / "b.md"
        a.write_text("identical content")
        b.write_text("identical content")
        pipeline, *_ = _pipeline()
        assert pipeline.ingest_file(a).content_hash == pipeline.ingest_file(b).content_hash

    def test_different_content_different_hash(self, tmp_path: Path):
        a, b = tmp_path / "a.md", tmp_path / "b.md"
        a.write_text("content A")
        b.write_text("content B")
        pipeline, *_ = _pipeline()
        assert pipeline.ingest_file(a).content_hash != pipeline.ingest_file(b).content_hash


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


def _pipeline(
    prepared_chunks: list[Chunk] | None = None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock]:
    service = MagicMock()
    service.prepare.return_value = (
        prepared_chunks if prepared_chunks is not None else [_embedded_chunk()]
    )
    vector_store = MagicMock()
    bm25 = MagicMock()
    pipeline = IngestionPipeline(service=service, vector_store=vector_store, bm25=bm25)
    return pipeline, service, vector_store, bm25


class TestIngestionPipelineFile:
    def test_returns_ingestion_result(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Hello\n\nSome content here.")
        pipeline, *_ = _pipeline()
        result = pipeline.ingest_file(path)
        assert isinstance(result, IngestionResult)

    def test_chunk_count_in_result(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("# Hello\n\nSome content here.")
        pipeline, *_ = _pipeline([_embedded_chunk(0), _embedded_chunk(1)])
        result = pipeline.ingest_file(path)
        assert result.chunk_count == 2

    def test_content_hash_set(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("some content")
        pipeline, *_ = _pipeline()
        result = pipeline.ingest_file(path)
        assert len(result.content_hash) == 16

    def test_vector_store_upsert_called(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        pipeline, _, vector_store, _ = _pipeline()
        pipeline.ingest_file(path)
        vector_store.upsert.assert_called_once()

    def test_bm25_add_called(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        pipeline, _, _, bm25 = _pipeline()
        pipeline.ingest_file(path)
        bm25.add.assert_called_once()

    def test_no_upsert_when_no_chunks(self, tmp_path: Path):
        path = tmp_path / "doc.md"
        path.write_text("content")
        pipeline, _, vector_store, bm25 = _pipeline(prepared_chunks=[])
        result = pipeline.ingest_file(path)
        vector_store.upsert.assert_not_called()
        bm25.add.assert_not_called()
        assert result.chunk_count == 0

    def test_load_error_raises_ingestion_error(self, tmp_path: Path):
        path = tmp_path / "ghost.pdf"
        pipeline, *_ = _pipeline()
        with pytest.raises(IngestionError):
            pipeline.ingest_file(path)

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = tmp_path / "file.xyz"
        path.write_text("content")
        pipeline, *_ = _pipeline()
        with pytest.raises(IngestionError):
            pipeline.ingest_file(path)


class TestIngestionPipelineDirectory:
    def test_returns_results_for_each_file(self, tmp_path: Path):
        for i in range(3):
            (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}\n\nContent {i}.")
        pipeline, *_ = _pipeline()
        results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 3

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        pipeline, *_ = _pipeline()
        assert pipeline.ingest_directory(tmp_path) == []

    def test_failed_file_recorded_not_raised(self, tmp_path: Path):
        (tmp_path / "good.md").write_text("# Good\n\nContent.")
        (tmp_path / "bad.xyz").write_text("unsupported")
        pipeline, _, _, _ = _pipeline()
        # Inject a failure for the good file by making service raise
        pipeline._service.prepare.side_effect = [RuntimeError("boom"), [_embedded_chunk()]]
        results = pipeline.ingest_directory(tmp_path)
        errors = [r for r in results if r.error is not None]
        assert len(errors) >= 1

    def test_ignores_unsupported_extensions(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        pipeline, *_ = _pipeline()
        results = pipeline.ingest_directory(tmp_path)
        assert len(results) == 1

    def test_save_indexes_calls_bm25_save(self):
        pipeline, _, _, bm25 = _pipeline()
        pipeline.save_indexes()
        bm25.save.assert_called_once()


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
