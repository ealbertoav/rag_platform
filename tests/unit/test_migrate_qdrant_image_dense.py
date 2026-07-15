"""Unit tests for scripts/migrate_qdrant_image_dense.py (T-252)."""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.migrate_qdrant_image_dense import run_migration


def _store(**overrides: object) -> MagicMock:
    store = MagicMock()
    store.image_dense_dim = overrides.get("image_dense_dim", 512)
    store.collection_exists.return_value = overrides.get("collection_exists", True)
    store.has_named_vector.return_value = overrides.get("has_named_vector", False)
    store.export_all_points.return_value = overrides.get("export_all_points", [])
    return store


class TestRunMigration:
    def test_noop_when_provider_has_no_image_space(self, capsys):
        store = _store(image_dense_dim=None)
        run_migration(store, provider="bge_m3", collection="rag_documents", dry_run=False)
        assert "no image embedding space" in capsys.readouterr().out
        store.export_all_points.assert_not_called()
        store.recreate_collection.assert_not_called()

    def test_noop_when_collection_missing(self, capsys):
        store = _store(collection_exists=False)
        run_migration(store, provider="clip", collection="rag_documents", dry_run=False)
        assert "does not exist yet" in capsys.readouterr().out
        store.recreate_collection.assert_not_called()

    def test_noop_when_already_migrated(self, capsys):
        store = _store(has_named_vector=True)
        run_migration(store, provider="clip", collection="rag_documents", dry_run=False)
        assert "already has image_dense" in capsys.readouterr().out
        store.export_all_points.assert_not_called()
        store.recreate_collection.assert_not_called()

    def test_dry_run_reports_without_writing(self, capsys):
        chunks = [MagicMock(), MagicMock()]
        store = _store(export_all_points=chunks)
        run_migration(store, provider="clip", collection="rag_documents", dry_run=True)
        out = capsys.readouterr().out
        assert "Found 2 point(s)" in out
        assert "Dry-run" in out
        store.recreate_collection.assert_not_called()
        store.upsert.assert_not_called()

    def test_migrates_and_restores_points(self, capsys):
        chunks = [MagicMock(), MagicMock()]
        store = _store(export_all_points=chunks)
        run_migration(store, provider="clip", collection="rag_documents", dry_run=False)
        store.recreate_collection.assert_called_once()
        store.upsert.assert_called_once_with(chunks)
        assert "Recreated collection" in capsys.readouterr().out

    def test_migrates_empty_collection_skips_upsert(self, capsys):
        store = _store(export_all_points=[])
        run_migration(store, provider="voyage", collection="rag_documents", dry_run=False)
        store.recreate_collection.assert_called_once()
        store.upsert.assert_not_called()
