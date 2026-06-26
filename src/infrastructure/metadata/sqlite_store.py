from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.core.constants import METADATA_DB_PATH
from src.domain.repositories.metadata_repository import (
    DocumentRecord,
    MetadataRepository,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (document_id, chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    chunks_added INTEGER NOT NULL DEFAULT 0,
    chunks_skipped INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteMetadataStore(MetadataRepository):
    """SQLite-backed metadata store for ingestion deduplication and auditing."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or METADATA_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def from_settings(cls) -> SQLiteMetadataStore:
        from src.core.settings import settings

        return cls(db_path=Path(settings.metadata.db_path))

    def get_by_source(self, source_path: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, source_path, content_hash, chunk_count, ingested_at, updated_at "
                "FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=row[0],
            source_path=row[1],
            content_hash=row[2],
            chunk_count=row[3],
            ingested_at=_parse_ts(row[4]),
            updated_at=_parse_ts(row[5]),
        )

    def get_chunk_ids(self, document_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM document_chunks WHERE document_id = ? ORDER BY chunk_id",
                (document_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def upsert_document(
        self,
        source_path: str,
        content_hash: str,
        chunk_ids: list[str],
        *,
        chunk_count: int | None = None,
        duration_ms: float = 0.0,
        skipped: bool = False,
        error: str | None = None,
    ) -> DocumentRecord:
        now = _now()
        now_iso = now.isoformat()
        status = "skipped" if skipped else ("error" if error else "success")
        reported_chunk_count = chunk_count if chunk_count is not None else len(chunk_ids)

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, ingested_at FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()

            if existing is None:
                doc_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO documents "
                    "(id, source_path, content_hash, chunk_count, ingested_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, source_path, content_hash, reported_chunk_count, now_iso, now_iso),
                )
                ingested_at = now
            else:
                doc_id = existing[0]
                ingested_at = _parse_ts(existing[1])
                conn.execute(
                    "UPDATE documents SET content_hash = ?, chunk_count = ?, updated_at = ? "
                    "WHERE id = ?",
                    (content_hash, reported_chunk_count, now_iso, doc_id),
                )
                conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))

            for chunk_id in chunk_ids:
                conn.execute(
                    "INSERT INTO document_chunks (document_id, chunk_id) VALUES (?, ?)",
                    (doc_id, chunk_id),
                )

            run_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO ingestion_runs ("
                "id, document_id, status, chunks_added, chunks_skipped, "
                "duration_ms, error, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    doc_id,
                    status,
                    0 if skipped else reported_chunk_count,
                    reported_chunk_count if skipped else 0,
                    duration_ms,
                    error,
                    now_iso,
                ),
            )

        return DocumentRecord(
            id=doc_id,
            source_path=source_path,
            content_hash=content_hash,
            chunk_count=reported_chunk_count,
            ingested_at=ingested_at,
            updated_at=now,
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source_path, content_hash, chunk_count, ingested_at, updated_at "
                "FROM documents ORDER BY updated_at DESC"
            ).fetchall()
        return [
            DocumentRecord(
                id=row[0],
                source_path=row[1],
                content_hash=row[2],
                chunk_count=row[3],
                ingested_at=_parse_ts(row[4]),
                updated_at=_parse_ts(row[5]),
            )
            for row in rows
        ]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection and always close it on exit.

        Note: ``with sqlite3.Connection`` only commits/rolls back — it does
        *not* close the connection, which triggers ResourceWarning under pytest.
        """
        conn = sqlite3.connect(self._path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.debug("SQLite metadata schema ready at %s", self._path)
