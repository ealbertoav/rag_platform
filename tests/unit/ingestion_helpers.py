"""Shared helpers for ingestion pipeline unit tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.domain.entities.chunk import Chunk
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline


def embedded_chunk(i: int = 0) -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=f"chunk {i}",
        embedding=[float(i)] * 4,
        sparse_vector={i + 1: 0.9},
    )


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
