"""T-117 — SQLite metadata store tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.metadata.sqlite_store import SQLiteMetadataStore


class TestSQLiteMetadataStore:
    def test_upsert_and_get_by_source(self, tmp_path: Path):
        db = SQLiteMetadataStore(db_path=tmp_path / "meta.db")
        record = db.upsert_document("/tmp/doc.md", "abc123", ["c1", "c2"], duration_ms=12.0)
        assert record.source_path == "/tmp/doc.md"
        assert record.content_hash == "abc123"
        assert record.chunk_count == 2

        loaded = db.get_by_source("/tmp/doc.md")
        assert loaded is not None
        assert loaded.id == record.id
        assert db.get_chunk_ids(record.id) == ["c1", "c2"]

    def test_list_documents(self, tmp_path: Path):
        db = SQLiteMetadataStore(db_path=tmp_path / "meta.db")
        db.upsert_document("/tmp/a.md", "hash-a", ["c1"])
        db.upsert_document("/tmp/b.md", "hash-b", ["c2"])
        docs = db.list_documents()
        assert len(docs) == 2
        paths = {d.source_path for d in docs}
        assert paths == {"/tmp/a.md", "/tmp/b.md"}

    def test_skipped_run_recorded(self, tmp_path: Path):
        db = SQLiteMetadataStore(db_path=tmp_path / "meta.db")
        record = db.upsert_document(
            "/tmp/doc.md",
            "abc123",
            ["c1"],
            duration_ms=1.0,
            skipped=True,
        )
        assert record.chunk_count == 1

    def test_update_replaces_chunk_ids(self, tmp_path: Path):
        db = SQLiteMetadataStore(db_path=tmp_path / "meta.db")
        first = db.upsert_document("/tmp/doc.md", "hash-v1", ["c1", "c2"])
        second = db.upsert_document("/tmp/doc.md", "hash-v2", ["c3"])
        assert first.id == second.id
        assert db.get_chunk_ids(first.id) == ["c3"]

    def test_get_by_source_returns_none_for_missing(self, tmp_path: Path):
        db = SQLiteMetadataStore(db_path=tmp_path / "meta.db")
        assert db.get_by_source("/missing.md") is None

    def test_from_settings_uses_config_path(self, tmp_path: Path):
        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.metadata = MagicMock(db_path=str(tmp_path / "custom.db"))
            db = SQLiteMetadataStore.from_settings()
        assert db._path == tmp_path / "custom.db"

    def test_connect_rolls_back_on_error(self, tmp_path: Path):
        db = SQLiteMetadataStore(db_path=tmp_path / "meta.db")
        with pytest.raises(sqlite3.OperationalError):
            with db._connect() as conn:
                conn.execute("SELECT * FROM no_such_table_xyz")
        record = db.upsert_document("/tmp/x.md", "h1", ["c1"])
        assert record.source_path == "/tmp/x.md"
