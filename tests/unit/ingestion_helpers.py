"""Shared helpers for ingestion pipeline unit tests."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline, IngestionResult, content_hash

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25 import BM25Index
    from src.infrastructure.vectordb.bm25_disk import DiskBM25Index


def embedded_chunk(i: int = 0) -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=f"chunk {i}",
        embedding=[float(i)] * 4,
        sparse_vector={i + 1: 0.9},
    )


def write_reingest_doc(tmp_path: Path, *, content: str = "version two") -> Path:
    path = tmp_path / "doc.md"
    path.write_text(content)
    return path


def unchanged_skip_setup(
    tmp_path: Path,
    *,
    content: str = "stable content",
    chunk_count: int = 1,
    chunk_ids: list[str] | None = None,
) -> tuple[Path, MagicMock]:
    """Return an on-disk doc and metadata that matches its content hash."""
    path = tmp_path / "doc.md"
    path.write_text(content)
    metadata = MagicMock()
    metadata.get_by_source.return_value = MagicMock(
        id="doc-1",
        content_hash=content_hash(str(path.resolve()), content),
        chunk_count=chunk_count,
    )
    metadata.get_chunk_ids.return_value = chunk_ids or ["c1"]
    return path, metadata


def unchanged_hash_metadata(
    path: Path,
    document: Document,
    *,
    chunk_ids: list[str],
) -> MagicMock:
    """Metadata mock whose stored hash matches *document* content at *path*."""
    metadata = MagicMock()
    metadata.get_by_source.return_value = MagicMock(
        id="doc-1",
        content_hash=content_hash(str(path.resolve()), document.content),
        chunk_count=1,
    )
    metadata.get_chunk_ids.return_value = chunk_ids
    return metadata


def run_structured_chunker_ingest(
    path: Path,
    document: Document,
    *,
    chunker: MagicMock,
    chunker_attr: str,
    pipeline: IngestionPipeline,
) -> IngestionResult:
    """Ingest *path* with a mocked structured chunker attribute on *pipeline*."""
    setattr(pipeline, chunker_attr, chunker)
    with patch(
        "src.rag.pipelines.ingestion_pipeline.load_document",
        return_value=document,
    ):
        return pipeline.ingest_file(path)


def run_skip_structured_chunker_ingest(
    path: Path,
    document: Document,
    *,
    chunker: MagicMock,
    chunker_attr: str,
    chunk_ids: list[str],
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Skip-path ingest when content hash is unchanged, with an optional chunker."""
    metadata = unchanged_hash_metadata(path, document, chunk_ids=chunk_ids)
    pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
    result = run_structured_chunker_ingest(
        path,
        document,
        chunker=chunker,
        chunker_attr=chunker_attr,
        pipeline=pipeline,
    )
    return result, service, vector_store, bm25, metadata


def skip_backfills_missing_structured_chunks_on_unchanged_hash(
    path: Path,
    document: Document,
    *,
    indexed_chunk: Chunk,
    chunker_attr: str,
    is_structured_chunk: Callable[[Chunk], bool],
) -> None:
    """Unchanged-hash skip backfills a missing structured chunk into both indexes."""
    chunker = MagicMock()
    chunker.index.return_value = [indexed_chunk]
    result, service, vector_store, bm25, metadata = run_skip_structured_chunker_ingest(
        path,
        document,
        chunker=chunker,
        chunker_attr=chunker_attr,
        chunk_ids=["text-chunk-1"],
    )

    assert result.skipped is False
    service.prepare.assert_not_called()
    chunker.index.assert_called_once()
    indexed_document = chunker.index.call_args.args[0]
    assert indexed_document.id == "doc-1"
    vector_store.upsert.assert_called_once()
    upserted = vector_store.upsert.call_args.args[0]
    assert len(upserted) == 1
    assert is_structured_chunk(upserted[0])
    bm25.add.assert_called_once()
    metadata.upsert_document.assert_called_once()
    _, _, merged_ids = metadata.upsert_document.call_args.args
    assert merged_ids == ["text-chunk-1", indexed_chunk.id]


def skip_purges_stale_structured_chunks_when_layout_changes(
    path: Path,
    document: Document,
    *,
    indexed_chunk: Chunk,
    stale_chunk_id: str,
    chunker_attr: str,
) -> None:
    """Unchanged-hash skip replaces a stale structured chunk with a new layout id."""
    chunker = MagicMock()
    chunker.index.return_value = [indexed_chunk]
    result, service, vector_store, bm25, metadata = run_skip_structured_chunker_ingest(
        path,
        document,
        chunker=chunker,
        chunker_attr=chunker_attr,
        chunk_ids=["text-chunk-1", stale_chunk_id],
    )

    assert result.skipped is False
    service.prepare.assert_not_called()
    vector_store.upsert.assert_called_once()
    vector_store.delete.assert_called_once_with([stale_chunk_id])
    bm25.remove_by_ids.assert_called_once_with([stale_chunk_id])
    _, _, merged_ids = metadata.upsert_document.call_args.args
    assert merged_ids == ["text-chunk-1", indexed_chunk.id]


def run_skip_with_indexed_structured_chunks(
    path: Path,
    document: Document,
    *,
    indexed_chunk: Chunk,
    chunker_attr: str,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock]:
    """Skip-path ingest when a structured chunk is already present in metadata/BM25."""
    chunker = MagicMock()
    chunker.index.return_value = [indexed_chunk]
    metadata = unchanged_hash_metadata(
        path,
        document,
        chunk_ids=["text-chunk-1", indexed_chunk.id],
    )
    pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
    bm25.get_by_id.side_effect = lambda chunk_id: (
        indexed_chunk.model_copy(update={"embedding": None, "sparse_vector": None})
        if chunk_id == indexed_chunk.id
        else None
    )
    result = run_structured_chunker_ingest(
        path,
        document,
        chunker=chunker,
        chunker_attr=chunker_attr,
        pipeline=pipeline,
    )
    return result, service, vector_store, bm25


def run_reingest_with_empty_structured_chunker(
    path: Path,
    document: Document,
    *,
    chunker_attr: str,
    old_chunk_ids: list[str],
    prepared_chunks: list[Chunk] | None = None,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock]:
    """Re-ingest when the structured chunker returns no embedded chunks.

    Defaults to refreshing text chunks while keeping ``old_chunk_ids`` in metadata.
    Returns ``(result, metadata, vector_store, bm25)``.
    """
    chunks = (
        prepared_chunks
        if prepared_chunks is not None
        else [embedded_chunk(0).model_copy(update={"id": "new-text-chunk-1"})]
    )
    metadata = mock_reingest_metadata(chunk_ids=old_chunk_ids)
    chunker = mock_chunker_with_empty_index()
    pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
        prepared_chunks=chunks,
        metadata=metadata,
    )
    result = run_structured_chunker_ingest(
        path,
        document,
        chunker=chunker,
        chunker_attr=chunker_attr,
        pipeline=pipeline,
    )
    return result, metadata, vector_store, bm25


def full_reindex_on_skip_preserves_stable_structured_chunk_ids(
    path: Path,
    document: Document,
    *,
    indexed_chunk: Chunk,
    chunker_attr: str,
) -> None:
    """Hash-unchanged skip forced to full reindex keeps a stable structured chunk."""
    metadata = unchanged_hash_metadata(
        path,
        document,
        chunk_ids=["old-text-chunk-1", indexed_chunk.id],
    )
    augmentor = MagicMock()
    augmentor.augment.return_value = []
    chunker = MagicMock()
    chunker.index.return_value = [indexed_chunk]
    pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(
        prepared_chunks=[embedded_chunk(0).model_copy(update={"id": "new-text-chunk-1"})],
        metadata=metadata,
        augmentor=augmentor,
    )
    result = run_structured_chunker_ingest(
        path,
        document,
        chunker=chunker,
        chunker_attr=chunker_attr,
        pipeline=pipeline,
    )

    assert result.skipped is False
    service.prepare.assert_called_once()
    assert_purges_only_stale_text_chunk(vector_store, bm25)


def assert_purges_only_stale_text_chunk(
    vector_store: MagicMock,
    bm25: MagicMock,
    *,
    stale_text_chunk_id: str = "old-text-chunk-1",
) -> None:
    vector_store.delete.assert_called_once_with([stale_text_chunk_id])
    bm25.remove_by_ids.assert_called_once_with([stale_text_chunk_id])


def assert_skip_without_reindex(
    result: IngestionResult,
    service: MagicMock,
    vector_store: MagicMock,
    bm25: MagicMock,
) -> None:
    assert result.skipped is True
    service.prepare.assert_not_called()
    vector_store.upsert.assert_not_called()
    bm25.add.assert_not_called()


def skip_unchanged_without_structured_chunker(
    path: Path,
    document: Document,
    *,
    chunker_attr: str,
) -> None:
    """Unchanged-hash skip with the structured chunker explicitly disabled."""
    metadata = unchanged_hash_metadata(path, document, chunk_ids=["text-chunk-1"])
    pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
    setattr(pipeline, chunker_attr, None)
    with patch(
        "src.rag.pipelines.ingestion_pipeline.load_document",
        return_value=document,
    ):
        result = pipeline.ingest_file(path)

    assert result.skipped is True
    assert_skip_without_reindex(result, service, vector_store, bm25)


def mock_chunker_with_empty_index() -> MagicMock:
    chunker = MagicMock()
    chunker.index.return_value = []
    return chunker


def run_skip_purge_only_structured_chunker_ingest(
    path: Path,
    document: Document,
    *,
    chunker_attr: str,
    stale_chunk_id: str,
    logger_name: str,
    caplog: pytest.LogCaptureFixture,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Skip-path ingest that only purges a stale structured chunk ID."""
    chunker = mock_chunker_with_empty_index()
    with caplog.at_level(logging.INFO, logger=logger_name):
        result, service, vector_store, bm25, metadata = run_skip_structured_chunker_ingest(
            path,
            document,
            chunker=chunker,
            chunker_attr=chunker_attr,
            chunk_ids=["text-chunk-1", stale_chunk_id],
        )
    return result, service, vector_store, bm25, metadata, chunker


def assert_skip_purged_only_structured_chunks(
    result: IngestionResult,
    service: MagicMock,
    chunker: MagicMock,
    vector_store: MagicMock,
    bm25: MagicMock,
    metadata: MagicMock,
    *,
    stale_chunk_id: str,
    merged_ids: list[str],
    index_called: bool,
    kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert result.skipped is False
    service.prepare.assert_not_called()
    if index_called:
        chunker.index.assert_called_once()
    else:
        chunker.index.assert_not_called()
    vector_store.upsert.assert_not_called()
    vector_store.delete.assert_called_once_with([stale_chunk_id])
    bm25.remove_by_ids.assert_called_once_with([stale_chunk_id])
    metadata.upsert_document.assert_called_once()
    _, _, actual_merged_ids = metadata.upsert_document.call_args.args
    assert actual_merged_ids == merged_ids
    assert f"Purged 1 stale {kind} chunk(s)" in caplog.text


def reingest_corpus_chunks() -> tuple[Chunk, Chunk, Chunk]:
    """Return (baseline, filler, stale) chunks for BM25 re-ingest tests."""
    return (
        Chunk(id="baseline-1", document_id="doc-0", text="baseline corpus for bm25 idf"),
        Chunk(id="filler-1", document_id="doc-9", text="unrelated filler document text"),
        Chunk(id="old-chunk-1", document_id="doc-1", text="stale databases terms"),
    )


def reingest_fresh_chunk() -> Chunk:
    return Chunk(document_id="doc-1", text="fresh content", embedding=[0.1] * 4)


def index_reingest_corpus(bm25: BM25Index | DiskBM25Index) -> None:
    baseline, filler, stale = reingest_corpus_chunks()
    bm25.index([baseline, filler, stale])


def vector_store_with_upsert_failure(*, error: str = "qdrant down") -> MagicMock:
    store = MagicMock()
    store.upsert.side_effect = RuntimeError(error)
    return store


def memory_bm25_index(_tmp_path: Path) -> BM25Index:
    from src.infrastructure.vectordb.bm25 import BM25Index

    return BM25Index()


def disk_bm25_index(tmp_path: Path) -> DiskBM25Index:
    from src.infrastructure.vectordb.bm25_disk import DiskBM25Index

    return DiskBM25Index(tmp_path / "bm25_disk", segment_size=2)


def mock_reingest_metadata(*, chunk_ids: list[str] | None = None) -> MagicMock:
    metadata = MagicMock()
    metadata.get_by_source.return_value = MagicMock(
        id="doc-meta-1",
        content_hash="old-hash",
        chunk_count=1,
    )
    metadata.get_chunk_ids.return_value = chunk_ids or ["old-chunk-1"]
    return metadata


def real_bm25_reingest_pipeline(
    bm25: BM25Index | DiskBM25Index,
    *,
    prepared_chunks: list[Chunk],
    vector_store: MagicMock | None = None,
    metadata: MagicMock | None = None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock]:
    service = MagicMock()
    service.prepare.return_value = prepared_chunks
    store = vector_store or MagicMock()
    meta = metadata or mock_reingest_metadata()
    pipeline = IngestionPipeline(
        service=service,
        vector_store=store,
        bm25=bm25,
        metadata=meta,
    )
    return pipeline, service, store, meta


def reingest_preserves_stable_structured_chunks_in_real_bm25(
    path: Path,
    document: Document,
    *,
    structured_chunk: Chunk,
    chunker_attr: str,
    stale_structured_text: str,
) -> None:
    """Assert stable structured chunk IDs survive a real-BM25 re-ingest."""
    from src.infrastructure.vectordb.bm25 import BM25Index

    stable_id = structured_chunk.id
    stale_text = Chunk(
        id="old-text-chunk-1",
        document_id="doc-1",
        text="stale paragraph about databases",
        embedding=[0.1] * 4,
    )
    stale_structured = structured_chunk.model_copy(
        update={"text": stale_structured_text, "embedding": [0.2] * 4}
    )
    fresh_text = embedded_chunk(0).model_copy(
        update={"id": "new-text-chunk-1", "text": "fresh paragraph about databases"}
    )

    bm25 = BM25Index()
    bm25.index([stale_text, stale_structured])
    metadata = mock_reingest_metadata(chunk_ids=[stale_text.id, stable_id])
    chunker = MagicMock()
    chunker.index.return_value = [structured_chunk]
    pipeline, _, vector_store, _ = real_bm25_reingest_pipeline(
        bm25,
        prepared_chunks=[fresh_text],
        metadata=metadata,
    )
    setattr(pipeline, chunker_attr, chunker)

    with patch(
        "src.rag.pipelines.ingestion_pipeline.load_document",
        return_value=document,
    ):
        pipeline.ingest_file(path)

    vector_store.delete.assert_called_once_with([stale_text.id])
    assert bm25.get_by_id(stale_text.id) is None
    assert bm25.get_by_id(stable_id) is not None
    assert bm25.get_by_id(fresh_text.id) is not None


def mock_ingestion_pipeline(
    prepared_chunks: list[Chunk] | None = None,
    metadata: MagicMock | None = None,
    augmentor: MagicMock | None = None,
    graph_indexer: MagicMock | None = None,
) -> tuple[IngestionPipeline, MagicMock, MagicMock, MagicMock]:
    service = MagicMock()
    service.prepare.return_value = (
        prepared_chunks if prepared_chunks is not None else [embedded_chunk()]
    )
    vector_store = MagicMock()
    bm25 = MagicMock()
    pipeline = IngestionPipeline(
        service=service,
        vector_store=vector_store,
        bm25=bm25,
        metadata=metadata,
        augmentor=augmentor,
        graph_indexer=graph_indexer,
    )
    return pipeline, service, vector_store, bm25


def ingest_without_structured_chunker_indexes_text_only(tmp_path: Path) -> None:
    """Ingest a plain markdown file when no structured chunker is configured."""
    path = tmp_path / "doc.md"
    path.write_text("hello")
    pipeline, _, vector_store, _ = mock_ingestion_pipeline()
    result = pipeline.ingest_file(path)
    assert result.chunk_count == 1
    assert len(vector_store.upsert.call_args.args[0]) == 1


def ingestion_pipeline_from_settings(*, parsing: MagicMock) -> IngestionPipeline:
    """Build ``IngestionPipeline.from_settings`` with optional indexers disabled."""
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
        mock_settings.parsing = parsing
        get_embedder.return_value = MagicMock()
        return IngestionPipeline.from_settings()


def ingest_with_hierarchical_and_hype_indexers(path: Path) -> tuple[list[Chunk], Chunk, Chunk]:
    """Ingest *path* with mocked hierarchical and HyPE indexers.

    Returns ``(upserted_chunks, summary_chunk, hype_chunk)``.
    """
    base = embedded_chunk(0)
    summary = embedded_chunk(1)
    hype = embedded_chunk(2)
    hierarchical = MagicMock()
    hierarchical.index.return_value = ([base], [summary])
    hype_indexer = MagicMock()
    hype_indexer.index.return_value = [hype]
    pipeline, _, vector_store, _ = mock_ingestion_pipeline([base])
    pipeline._hierarchical_indexer = hierarchical
    pipeline._hype_indexer = hype_indexer
    pipeline.ingest_file(path)
    return vector_store.upsert.call_args.args[0], summary, hype
