from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, cast

from src.core.async_bridge import run_async
from src.core.exceptions import RetrievalError
from src.domain.entities.retrieval_filter import RetrievalFilter

if TYPE_CHECKING:
    from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GraphRelation:
    """A knowledge-graph triple extracted from a document chunk."""

    subject: str
    relation: str
    object: str
    subject_type: str = "Entity"
    object_type: str = "Entity"


class Neo4jGraphRepository:
    """Stores and queries a knowledge graph backed by Neo4j.

    Schema
    ------
    Nodes: (:Entity {name, type})
    Edges: (:Entity)-[:RELATES_TO {relation}]->(:Entity)
            (:Entity)-[:MENTIONED_IN {document_id}]->(:Chunk {id})

    Uses the async Neo4j driver, so graph retrieval does not block the event loop.
    Requires "pip install neo4j" (or "uv sync --extra graph").
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        *,
        max_connection_pool_size: int = 100,
    ) -> None:
        self.uri: str = uri
        self.user: str = user
        self.password: str = password
        self.max_connection_pool_size: Any = max_connection_pool_size
        self._driver: AsyncDriver | None = None

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> Neo4jGraphRepository:
        from src.core.settings import settings

        cfg = settings.neo4j
        raw_password = cfg.password
        if hasattr(raw_password, "get_secret_value"):
            password = raw_password.get_secret_value()
        else:
            password = str(raw_password)
        return cls(
            uri=cfg.uri,
            user=cfg.user,
            password=password,
            max_connection_pool_size=cfg.max_connection_pool_size,
        )

    # ── Public (async) ─────────────────────────────────────────────────────────

    async def upsert(
        self,
        relations: list[GraphRelation],
        chunk_id: str,
        document_id: str = "",
    ) -> None:
        """Persist *relations* and link them to the originating *chunk_id*."""
        if not relations:
            return
        driver = self._get_driver()
        try:
            async with driver.session() as session:
                for rel in relations:
                    await session.execute_write(_upsert_relation, rel, chunk_id, document_id)
        except Exception as exc:
            raise RetrievalError("Neo4j upsert failed", cause=exc) from exc

    async def search_by_entities(
        self,
        entity_names: list[str],
        top_k: int,
        *,
        filters: RetrievalFilter | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs for chunks mentioning *entity_names*.

        Score = number of query entities found in the chunk (normalized).
        When *filters* include document IDs, scope is applied in Cypher before
        ranking and "LIMIT" so in-scope chunks are not dropped by a global cutoff.
        """
        if not entity_names:
            return []
        document_ids = list(filters.document_ids) if filters and filters.document_ids else None
        driver = self._get_driver()
        try:
            async with driver.session() as session:
                result = await session.execute_read(
                    _search_chunks, entity_names, top_k, document_ids
                )
            return cast(list[tuple[str, float]], result)
        except Exception as exc:
            raise RetrievalError("Neo4j search failed", cause=exc) from exc

    async def close(self) -> None:
        driver = self._driver
        if driver is not None:
            await driver.close()
            self._driver = None

    # ── Sync wrappers (ingestion / CLI callers) ────────────────────────────────

    def upsert_sync(
        self,
        relations: list[GraphRelation],
        chunk_id: str,
        document_id: str = "",
    ) -> None:
        """Synchronous wrapper for ingestion paths that are not async."""
        run_async(self.upsert(relations, chunk_id, document_id))

    def close_sync(self) -> None:
        """Synchronous wrapper for shutdown hooks."""
        run_async(self.close())

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_driver(self) -> AsyncDriver:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import AsyncGraphDatabase

            driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
            )
            self._driver = cast("AsyncDriver", driver)
            logger.info("Neo4j async driver connected to %s", self.uri)
            return self._driver
        except (ImportError, Exception) as exc:
            raise RetrievalError(f"Cannot connect to Neo4j at {self.uri!r}", cause=exc) from exc


# ── Cypher helpers ─────────────────────────────────────────────────────────────


async def _upsert_relation(tx: Any, rel: GraphRelation, chunk_id: str, document_id: str) -> None:
    await tx.run(
        """
        MERGE (s:Entity {name: $subject})
          ON CREATE SET s.type = $subject_type
        MERGE (o:Entity {name: $object})
          ON CREATE SET o.type = $object_type
        MERGE (s)-[r:RELATES_TO {relation: $relation}]->(o)
        MERGE (c:Chunk {id: $chunk_id})
          ON CREATE SET c.document_id = $document_id
        MERGE (s)-[:MENTIONED_IN]->(c)
        MERGE (o)-[:MENTIONED_IN]->(c)
        """,
        subject=rel.subject,
        subject_type=rel.subject_type,
        object=rel.object,
        object_type=rel.object_type,
        relation=rel.relation,
        chunk_id=chunk_id,
        document_id=document_id,
    )


async def _search_chunks(
    tx: Any,
    entity_names: list[str],
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[tuple[str, float]]:
    if document_ids:
        result = await tx.run(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.name IN $names AND c.document_id IN $document_ids
            WITH c.id AS chunk_id, count(DISTINCT e) AS hits
            RETURN chunk_id, toFloat(hits) / $n_entities AS score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            names=entity_names,
            n_entities=len(entity_names),
            top_k=top_k,
            document_ids=document_ids,
        )
    else:
        result = await tx.run(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.name IN $names
            WITH c.id AS chunk_id, count(DISTINCT e) AS hits
            RETURN chunk_id, toFloat(hits) / $n_entities AS score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            names=entity_names,
            n_entities=len(entity_names),
            top_k=top_k,
        )
    rows: list[tuple[str, float]] = []
    async for row in result:
        rows.append((row["chunk_id"], float(row["score"])))
    return rows
