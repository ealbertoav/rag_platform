"""Shared helpers for ingestion pipeline unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from src.domain.entities.chunk import Chunk
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

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


def reingest_corpus_chunks() -> tuple[Chunk, Chunk, Chunk]:
    """Return (baseline, filler, stale) chunks for BM25 re-ingest tests."""
    return (
        Chunk(id="baseline-1", document_id="doc-0", text="baseline corpus for bm25 idf"),
        Chunk(id="filler-1", document_id="doc-9", text="unrelated filler document text"),
        Chunk(id="old-chunk-1", document_id="doc-1", text="stale kubernetes terms"),
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
