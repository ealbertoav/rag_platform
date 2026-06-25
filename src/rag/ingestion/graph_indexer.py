from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
        self._extractor = extractor
        self._graph = graph

    def index_chunks(self, chunks: list[Chunk], document_id: str) -> None:
        for chunk in chunks:
            relations = self._extractor.extract_relations(chunk.text)
            if not relations:
                continue
            try:
                self._graph.upsert(relations, chunk_id=chunk.id, document_id=document_id)
            except Exception as exc:
                logger.warning("Failed to index graph relations for chunk %s: %s", chunk.id, exc)
