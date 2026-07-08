"""GraphIndexer unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.domain.entities.chunk import Chunk
from src.infrastructure.vectordb.neo4j_graph import GraphRelation
from src.rag.ingestion.graph_indexer import GraphIndexer


def _chunk(i: int = 0) -> Chunk:
    return Chunk(document_id="doc-1", text=f"chunk text {i}")


class TestGraphIndexer:
    def test_indexes_relations_for_each_chunk(self):
        extractor = MagicMock()
        extractor.extract_relations.return_value = [
            GraphRelation(subject="A", relation="rel", object="B")
        ]
        graph = MagicMock()
        graph.upsert = AsyncMock()
        indexer = GraphIndexer(extractor=extractor, graph=graph)
        chunks = [_chunk(0), _chunk(1)]
        indexer.index_chunks(chunks, document_id="doc-1")
        assert graph.upsert.await_count == 2

    def test_skips_chunks_with_no_relations(self):
        extractor = MagicMock()
        extractor.extract_relations.return_value = []
        graph = MagicMock()
        graph.upsert = AsyncMock()
        indexer = GraphIndexer(extractor=extractor, graph=graph)
        indexer.index_chunks([_chunk()], document_id="doc-1")
        graph.upsert.assert_not_awaited()

    def test_continues_when_upsert_fails(self):
        extractor = MagicMock()
        extractor.extract_relations.return_value = [
            GraphRelation(subject="A", relation="rel", object="B")
        ]
        graph = MagicMock()
        graph.upsert = AsyncMock(side_effect=RuntimeError("neo4j down"))
        indexer = GraphIndexer(extractor=extractor, graph=graph)
        indexer.index_chunks([_chunk(), _chunk(1)], document_id="doc-1")
        assert graph.upsert.await_count == 2
