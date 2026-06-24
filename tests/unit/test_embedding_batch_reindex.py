"""Unit tests for batch re-embedding helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.domain.entities.chunk import Chunk
from src.infrastructure.embeddings.batch_reindex import (
    embed_and_upsert_batch,
    maybe_sleep_between_api_batches,
)


def _chunk(i: int = 0) -> Chunk:
    return Chunk(
        id=f"chunk-{i:04d}",
        document_id="doc-1",
        text=f"chunk text {i}",
        metadata={"source": "test.pdf"},
    )


class TestEmbedAndUpsertBatch:
    def test_embeds_batch_and_upserts_vectors(self):
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1, 0.2]], [{}])
        vector_store = MagicMock()
        batch = [_chunk()]

        embed_and_upsert_batch(embedder, vector_store, batch)

        embedder.embed_both.assert_called_once_with(["chunk text 0"])
        vector_store.upsert.assert_called_once()
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert upserted[0].embedding == [0.1, 0.2]
        assert upserted[0].sparse_vector == {}


class TestMaybeSleepBetweenApiBatches:
    def test_skips_sleep_for_self_hosted_provider(self):
        with patch("src.infrastructure.embeddings.batch_reindex.time.sleep") as sleep:
            maybe_sleep_between_api_batches(
                "bge_m3",
                batch_start=0,
                batch_size=32,
                total_chunks=64,
            )
        sleep.assert_not_called()

    def test_sleeps_between_api_batches(self):
        with patch("src.infrastructure.embeddings.batch_reindex.time.sleep") as sleep:
            maybe_sleep_between_api_batches(
                "openai",
                batch_start=0,
                batch_size=32,
                total_chunks=64,
            )
        sleep.assert_called_once()

    def test_skips_sleep_after_final_batch(self):
        with patch("src.infrastructure.embeddings.batch_reindex.time.sleep") as sleep:
            maybe_sleep_between_api_batches(
                "openai",
                batch_start=32,
                batch_size=32,
                total_chunks=64,
            )
        sleep.assert_not_called()
