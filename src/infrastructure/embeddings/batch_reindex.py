"""Shared helpers for batch re-embedding and Qdrant upsert."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.core.constants import API_EMBEDDING_PROVIDERS
from src.domain.entities.chunk import Chunk

if TYPE_CHECKING:
    from src.domain.repositories.embedding_repository import EmbeddingRepository
    from src.domain.repositories.vector_store_repository import VectorStoreRepository

API_BATCH_SIZE = 32
API_BATCH_SLEEP = 0.1


def embed_and_upsert_batch(
    embedder: EmbeddingRepository,
    vector_store: VectorStoreRepository,
    batch: list[Chunk],
) -> None:
    """Embed *batch* and upsert the resulting vectors into *vector_store*."""
    dense_vecs, sparse_vecs = embedder.embed_both([c.text for c in batch])
    embedded = [
        chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs, strict=True)
    ]
    vector_store.upsert(embedded)


def maybe_sleep_between_api_batches(
    provider: str,
    *,
    batch_start: int,
    batch_size: int,
    total_chunks: int,
    sleep_seconds: float = API_BATCH_SLEEP,
) -> None:
    """Sleep between API embedding batches to reduce rate-limit errors."""
    if provider in API_EMBEDDING_PROVIDERS and batch_start + batch_size < total_chunks:
        time.sleep(sleep_seconds)
