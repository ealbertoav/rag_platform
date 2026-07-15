"""Add the image_dense named vector to an existing Qdrant collection (T-252).

Qdrant does not support adding a named vector to a collection in place, so
this script rebuilds the collection under the extended schema: it scrolls
every point out (payload + dense/sparse vectors), drops the collection, then
re-upserts everything through QdrantVectorStore.upsert(), which recreates the
collection using the current embedding config — now including image_dense
when the active provider (clip / voyage) has an image embedding space.

No-op when:
  - the active provider has no image embedding space (nothing to add)
  - the collection does not exist yet (created correctly on first upsert)
  - the collection already has an image_dense vector (already migrated)

This only extends the *schema* — existing chunks keep image_embedding=None
until re-ingested with a multimodal provider (T-253 wires that up).

Usage:
    uv run python scripts/migrate_qdrant_image_dense.py
    uv run python scripts/migrate_qdrant_image_dense.py --dry-run
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from src.core.constants import IMAGE_DENSE_VECTOR_NAME

if TYPE_CHECKING:
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore


def run_migration(
    store: QdrantVectorStore,
    *,
    provider: str,
    collection: str,
    dry_run: bool,
) -> None:
    """Add image_dense to *collection* if the active *provider* supports images."""
    if store.image_dense_dim is None:
        print(f"Provider '{provider}' has no image embedding space — nothing to migrate.")
        return

    if not store.collection_exists():
        print(
            f"Collection '{collection}' does not exist yet — it will be created "
            "with image_dense automatically on first upsert."
        )
        return

    if store.has_named_vector(IMAGE_DENSE_VECTOR_NAME):
        print(f"Collection '{collection}' already has image_dense — nothing to do.")
        return

    chunks = store.export_all_points()
    print(f"Found {len(chunks)} point(s) to migrate.")
    if dry_run:
        print("Dry-run — no changes written.")
        return

    store.recreate_collection()
    if chunks:
        store.upsert(chunks)
    print(
        f"Recreated collection '{collection}' with image_dense "
        f"({store.image_dense_dim}-dim) and restored {len(chunks)} point(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the image_dense named vector to the Qdrant collection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without changing the collection",
    )
    args = parser.parse_args()

    from src.core.settings import settings
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    run_migration(
        QdrantVectorStore.from_settings(),
        provider=settings.embeddings.provider,
        collection=settings.qdrant.collection,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
