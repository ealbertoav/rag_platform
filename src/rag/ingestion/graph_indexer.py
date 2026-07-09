from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.async_bridge import run_async

if TYPE_CHECKING:
    from src.domain.entities.chunk import Chunk
    from src.infrastructure.vectordb.neo4j_graph import Neo4jGraphRepository
    from src.rag.retrieval.graph_retriever import EntityExtractor

logger = logging.getLogger(__name__)


class GraphIndexer:
    """Extract entities from chunks at ingested time and persist to Neo4j."""

    def __init__(
        self,
        extractor: EntityExtractor,
        graph: Neo4jGraphRepository,
    ) -> None:
        self._extractor: EntityExtractor = extractor
        self._graph: Neo4jGraphRepository = graph

    def index_chunks(self, chunks: list[Chunk], document_id: str) -> None:
        run_async(self._index_chunks_async(chunks, document_id))

    async def _index_chunks_async(self, chunks: list[Chunk], document_id: str) -> None:
        for chunk in chunks:
            relations = self._extractor.extract_relations(chunk.text)
            if not relations:
                continue
            try:
                await self._graph.upsert(relations, chunk_id=chunk.id, document_id=document_id)
            except Exception as exc:
                logger.warning("Failed to index graph relations for chunk %s: %s", chunk.id, exc)
