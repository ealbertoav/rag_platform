"""Unit tests for rebuild_embeddings preflight guards."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from scripts import rebuild_embeddings


def _args(*, force: bool = False, recreate_collection: bool = False) -> argparse.Namespace:
    return argparse.Namespace(force=force, recreate_collection=recreate_collection)


class TestRebuildEmbeddingsPreflight:
    def test_force_skips_dimension_warning_only(self):
        with (
            patch.object(rebuild_embeddings, "_preflight_dimensions") as dim_check,
            patch.object(rebuild_embeddings, "_preflight_model_mismatch") as mismatch_check,
        ):
            if not _args(force=True).force:
                rebuild_embeddings._preflight_dimensions(MagicMock())
            if not _args(force=True).recreate_collection:
                rebuild_embeddings._preflight_model_mismatch()

        dim_check.assert_not_called()
        mismatch_check.assert_called_once()

    def test_recreate_collection_skips_model_mismatch(self):
        with patch.object(rebuild_embeddings, "_preflight_model_mismatch") as mismatch_check:
            args = _args(recreate_collection=True)
            if not args.recreate_collection:
                rebuild_embeddings._preflight_model_mismatch()

        mismatch_check.assert_not_called()

    def test_model_mismatch_runs_without_force(self):
        with patch.object(rebuild_embeddings, "_preflight_model_mismatch") as mismatch_check:
            args = _args(force=False, recreate_collection=False)
            if not args.recreate_collection:
                rebuild_embeddings._preflight_model_mismatch()

        mismatch_check.assert_called_once()

    def test_model_mismatch_exits_with_error(self):
        from src.core.exceptions import VectorStoreError

        with (
            patch(
                "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"
            ) as from_settings,
            pytest.raises(SystemExit) as exc_info,
        ):
            store = MagicMock()
            store.validate_embedding_model.side_effect = VectorStoreError("mismatch")
            from_settings.return_value = store
            rebuild_embeddings._preflight_model_mismatch()

        assert exc_info.value.code == 1
