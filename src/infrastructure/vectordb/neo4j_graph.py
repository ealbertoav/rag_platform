from __future__ import annotations

import dataclasses
import logging
from typing import Any, cast

from src.core.exceptions import RetrievalError

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

    Requires "pip install neo4j" (or "uv sync --extra graph").
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: object | None = None

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> Neo4jGraphRepository:
        from src.core.settings import settings

        cfg = getattr(settings, "neo4j", None)
        if cfg is None:
            return cls()
        return cls(
            uri=str(getattr(cfg, "uri", "bolt://localhost:7687")),
            user=str(getattr(cfg, "user", "neo4j")),
            password=str(getattr(cfg, "password", "neo4j")),
        )

    # ── Public ─────────────────────────────────────────────────────────────────

    def upsert(self, relations: list[GraphRelation], chunk_id: str, document_id: str = "") -> None:
        """Persist *relations* and link them to the originating *chunk_id*."""
        if not relations:
            return
        driver = self._get_driver()
        try:
            with driver.session() as session:  # type: ignore[attr-defined]
                for rel in relations:
                    session.execute_write(_upsert_relation, rel, chunk_id, document_id)
        except Exception as exc:
            raise RetrievalError("Neo4j upsert failed", cause=exc) from exc

    def search_by_entities(self, entity_names: list[str], top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs for chunks mentioning *entity_names*.

        Score = number of query entities found in the chunk (normalized).
        """
        if not entity_names:
            return []
        driver = self._get_driver()
        try:
            with driver.session() as session:  # type: ignore[attr-defined]
                result = session.execute_read(_search_chunks, entity_names, top_k)
            return cast(list[tuple[str, float]], result)
        except Exception as exc:
            raise RetrievalError("Neo4j search failed", cause=exc) from exc

    def close(self) -> None:
        driver = self._driver
        if driver is not None:
            driver.close()  # type: ignore[attr-defined]
            self._driver = None

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_driver(self) -> object:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]

            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver = driver
            logger.info("Neo4j driver connected to %s", self.uri)
            return driver
        except (ImportError, Exception) as exc:
            raise RetrievalError(f"Cannot connect to Neo4j at {self.uri!r}", cause=exc) from exc


# ── Cypher helpers ─────────────────────────────────────────────────────────────


def _upsert_relation(tx: Any, rel: GraphRelation, chunk_id: str, document_id: str) -> None:
    tx.run(
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


def _search_chunks(tx: Any, entity_names: list[str], top_k: int) -> list[tuple[str, float]]:
    result = tx.run(
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
    return [(row["chunk_id"], float(row["score"])) for row in result]
