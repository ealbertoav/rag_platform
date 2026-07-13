"""T-230 — LocalAssetStore unit tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.constants import ASSETS_DIR, ROOT
from src.rag.ingestion.local_asset_store import (
    LocalAssetStore,
    document_asset_key,
    resolve_store_root,
    sanitize_segment,
)


class TestDocumentAssetKey:
    def test_stable_for_same_source(self) -> None:
        assert document_asset_key("/docs/report.pdf") == document_asset_key("/docs/report.pdf")

    def test_differs_for_different_sources(self) -> None:
        assert document_asset_key("/docs/a.pdf") != document_asset_key("/docs/b.pdf")

    def test_includes_safe_stem(self) -> None:
        key = document_asset_key("/weird/My Report (final)!!.pdf")
        assert key.startswith("My_Report_final-")
        assert " " not in key
        assert "(" not in key

    def test_empty_stem_fallback(self) -> None:
        key = document_asset_key("/tmp/.hidden")
        assert key.startswith("document-") or key.startswith("hidden-")


class TestSanitizeSegment:
    def test_replaces_unsafe_characters(self) -> None:
        assert sanitize_segment("fig ure/1!") == "fig_ure_1"

    def test_empty_becomes_asset(self) -> None:
        assert sanitize_segment("!!!") == "asset"
        assert sanitize_segment("") == "asset"


class TestResolveStoreRoot:
    def test_relative_to_project_root(self) -> None:
        resolved = resolve_store_root("data/assets")
        assert resolved == (ROOT / "data" / "assets").resolve()

    def test_absolute_unchanged(self, tmp_path: Path) -> None:
        assert resolve_store_root(tmp_path) == tmp_path.resolve()


class TestLocalAssetStore:
    def test_default_root_is_assets_dir(self) -> None:
        store = LocalAssetStore()
        assert store.root == ASSETS_DIR.resolve()

    def test_custom_root(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path / "figures")
        assert store.root == (tmp_path / "figures").resolve()

    def test_path_for_normalizes_extension(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.path_for("doc-key", "figure-1", extension=".PNG")
        assert path == tmp_path.resolve() / "doc-key" / "figure-1.png"

    def test_path_for_empty_extension_defaults_png(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.path_for("doc-key", "figure-1", extension="")
        assert path.suffix == ".png"

    def test_save_writes_bytes(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.save("doc-key", "figure-1", b"png-bytes", extension="png")
        assert path.is_file()
        assert path.read_bytes() == b"png-bytes"

    def test_save_rejects_empty_bytes(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            store.save("doc-key", "figure-1", b"")

    def test_save_sanitizes_segments(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.save("doc key!!", "fig/1", b"x", extension="jpg")
        assert path.parent.name == "doc_key"
        assert path.name == "fig_1.jpg"

    def test_save_replacing_bytes_removes_caption_sidecar(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.save("doc-key", "figure-1", b"original", extension="png")
        sidecar = path.with_name("figure-1.caption.txt")
        sidecar.write_text("stale caption", encoding="utf-8")

        store.save("doc-key", "figure-1", b"replaced", extension="png")

        assert path.read_bytes() == b"replaced"
        assert not sidecar.exists()

    def test_save_identical_bytes_keeps_caption_sidecar(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.save("doc-key", "figure-1", b"same-bytes", extension="png")
        sidecar = path.with_name("figure-1.caption.txt")
        sidecar.write_text("keep me", encoding="utf-8")

        store.save("doc-key", "figure-1", b"same-bytes", extension="png")

        assert sidecar.read_text(encoding="utf-8") == "keep me"

    def test_save_unreadable_existing_still_writes(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.save("doc-key", "figure-1", b"original", extension="png")
        sidecar = path.with_name("figure-1.caption.txt")
        sidecar.write_text("maybe stale", encoding="utf-8")

        with patch.object(Path, "read_bytes", side_effect=OSError("busy")):
            result = store.save("doc-key", "figure-1", b"new-bytes", extension="png")

        assert result.read_bytes() == b"new-bytes"
        # Could not compare previous bytes → leave sidecar; captioner hash-binds
        assert sidecar.is_file()

    def test_save_sidecar_unlink_failure_still_writes(self, tmp_path: Path) -> None:
        store = LocalAssetStore(tmp_path)
        path = store.save("doc-key", "figure-1", b"original", extension="png")
        sidecar = path.with_name("figure-1.caption.txt")
        sidecar.write_text("stale", encoding="utf-8")

        with patch.object(Path, "unlink", side_effect=OSError("denied")):
            result = store.save("doc-key", "figure-1", b"replaced", extension="png")

        assert result.read_bytes() == b"replaced"
        assert sidecar.is_file()
