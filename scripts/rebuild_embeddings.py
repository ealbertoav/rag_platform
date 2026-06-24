"""Re-embed all chunks from the BM25 index and upsert them into Qdrant.

Use this when you:
  - Switch to a different embedding model
  - Change embedding dimensions in configs/embeddings.yaml
  - Need to recover a corrupted Qdrant collection while the BM25
    index (with chunk text) is still intact

Usage:
    uv run python scripts/rebuild_embeddings.py
    uv run python scripts/rebuild_embeddings.py --batch-size 16 --dry-run
"""
from __future__ import annotations

import argparse
import sys

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed all chunks and sync Qdrant")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of chunks to embed per batch (default: 32)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and count chunks without writing to Qdrant",
    )
    parser.add_argument(
        "--recreate-collection", action="store_true",
        help="Drop and recreate the Qdrant collection before upserting",
    )
    args = parser.parse_args()

    from src.core.settings import settings
    from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
    from src.infrastructure.vectordb.bm25 import BM25Index
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    # ── Load chunks from BM25 index ────────────────────────────────────────────
    bm25 = BM25Index.load_or_create()
    chunks = bm25.chunks

    if not chunks:
        print("BM25 index is empty — ingest documents first.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(chunks)} chunks in BM25 index.")

    if args.dry_run:
        print("Dry-run mode — no changes written.")
        return

    # ── Initialise infrastructure ───────────────────────────────────────────────
    embedder = BGEM3EmbeddingProvider.from_settings()
    vector_store = QdrantVectorStore.from_settings()

    if args.recreate_collection:
        print(f"Dropping collection '{settings.qdrant.collection}'…")
        try:
            vector_store._client.delete_collection(settings.qdrant.collection)  # type: ignore[attr-defined]
            vector_store._collection_ready = False  # type: ignore[attr-defined]
            print("Collection dropped.")
        except Exception as exc:
            print(f"Warning: could not drop collection: {exc}", file=sys.stderr)

    # ── Re-embed in batches ─────────────────────────────────────────────────────
    errors = 0
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Re-embedding", total=len(chunks))

        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i : i + args.batch_size]
            progress.update(task, description=f"[cyan]Batch {i // args.batch_size + 1}")
            try:
                dense_vecs, sparse_vecs = embedder.embed_both([c.text for c in batch])
                embedded = [
                    chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
                    for chunk, dense, sparse in zip(batch, dense_vecs, sparse_vecs, strict=True)
                ]
                vector_store.upsert(embedded)
            except Exception as exc:
                print(f"\nBatch {i}–{i + len(batch)} failed: {exc}", file=sys.stderr)
                errors += 1
            finally:
                progress.advance(task, len(batch))

    ok = len(chunks) - errors * args.batch_size
    print(f"\n✓ Re-embedded {ok}/{len(chunks)} chunks → Qdrant '{settings.qdrant.collection}'")
    if errors:
        print(f"  {errors} batch(es) failed — check logs above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
