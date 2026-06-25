"""Shared helpers for ingestion pipeline unit tests."""

from __future__ import annotations

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
