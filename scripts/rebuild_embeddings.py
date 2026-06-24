"""Re-embed all chunks from the BM25 index and upsert them into Qdrant.

Use this when you:
  - Switch to a different embedding model (self-hosted or API-based)
  - Change embedding dimensions in configs/embeddings.yaml
  - Need to recover a corrupted Qdrant collection while the BM25
    index (with chunk text) is still intact

Usage:
    uv run python scripts/rebuild_embeddings.py
    uv run python scripts/rebuild_embeddings.py --batch-size 16 --dry-run
    uv run python scripts/rebuild_embeddings.py --recreate-collection
    uv run python scripts/rebuild_embeddings.py --recreate-collection --force
"""

from __future__ import annotations

import argparse
import sys

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.core.constants import API_EMBEDDING_PROVIDERS, SELF_HOSTED_EMBEDDING_DEFAULT_DIMS
from src.infrastructure.embeddings.batch_reindex import (
    API_BATCH_SIZE,
    embed_and_upsert_batch,
    maybe_sleep_between_api_batches,
)

_DEFAULT_BATCH_SIZE = 32


def _check_api_key(settings: object) -> None:
    """Abort early if an API provider is selected, but its key is missing.

    Always called — even with --force — so users get a clear error rather
    than an opaque failure from the embedding API itself.
    """
    from src.core.settings import Settings

    s: Settings = settings  # type: ignore[assignment]
    provider = s.embeddings.provider

    if provider not in API_EMBEDDING_PROVIDERS:
        print(f"Provider: {provider} (self-hosted)")
        return

    key_map = {
        "openai": s.embeddings.openai.api_key.get_secret_value(),
        "voyage": s.embeddings.voyage.api_key.get_secret_value(),
        "cohere": s.embeddings.cohere.api_key.get_secret_value(),
        "gemini": s.embeddings.gemini.api_key.get_secret_value(),
    }
    # Guard against new providers added to API_EMBEDDING_PROVIDERS but not to key_map.
    if provider not in key_map:
        print(
            f"[error] Provider '{provider}' is listed as API-based but has no key_map entry.\n"
            f"        Update _check_api_key() in rebuild_embeddings.py to add it.",
            file=sys.stderr,
        )
        sys.exit(1)
    api_key = key_map[provider]
    if not api_key:
        env_var = f"EMBEDDINGS__{provider.upper()}__API_KEY"
        print(
            f"[error] Provider '{provider}' requires an API key.\n"
            f"        Set {env_var} in your environment or .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Provider: {provider} (API-based, key present)")


def _preflight_dimensions(settings: object) -> None:
    """Warn when dense_dim does not match the provider default."""
    from src.core.settings import Settings

    s: Settings = settings  # type: ignore[assignment]
    provider = s.embeddings.provider

    expected_dims = {
        **SELF_HOSTED_EMBEDDING_DEFAULT_DIMS,
        "openai": s.embeddings.openai.dimensions,
        "voyage": s.embeddings.voyage.dimensions,
        "cohere": s.embeddings.cohere.dimensions,
        "gemini": s.embeddings.gemini.dimensions,
    }
    if provider in expected_dims:
        expected = expected_dims[provider]
        configured = s.embeddings.dense_dim
        if expected != configured:
            print(
                f"[warn] dense_dim={configured} but {provider} default is {expected}. "
                f"Update embeddings.dense_dim in configs/embeddings.yaml if needed.",
                file=sys.stderr,
            )


def _preflight_model_mismatch() -> None:
    """Abort when the collection was built with a different embedding model."""
    from src.core.exceptions import VectorStoreError
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    try:
        QdrantVectorStore.from_settings().validate_embedding_model()
    except VectorStoreError as exc:
        print(f"\n[error] {exc}\n", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed all chunks and sync Qdrant")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of chunks to embed per batch (default: 32 for local, 32 for API providers)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and count chunks without writing to Qdrant",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the Qdrant collection before upserting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Skip dense_dim warning checks "
            "(model-mismatch guard still runs unless --recreate-collection)"
        ),
    )
    args = parser.parse_args()

    from src.core.settings import settings
    from src.infrastructure.embeddings import get_embedding_provider
    from src.infrastructure.vectordb.bm25 import BM25Index
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    _check_api_key(settings)  # always — gives clear error before any API call
    if not args.force:
        _preflight_dimensions(settings)
    if not args.recreate_collection:
        _preflight_model_mismatch()  # always enforced unless rebuilding the collection

    # ── Load chunks from BM25 index ────────────────────────────────────────────
    bm25 = BM25Index.load_or_create()
    chunks = bm25.chunks

    if not chunks:
        print("BM25 index is empty — ingest documents first.", file=sys.stderr)
        sys.exit(1)

    provider_name = settings.embeddings.provider
    is_api = provider_name in API_EMBEDDING_PROVIDERS
    batch_size = args.batch_size or (API_BATCH_SIZE if is_api else _DEFAULT_BATCH_SIZE)

    print(f"Found {len(chunks)} chunks in BM25 index.")
    if args.dry_run:
        if is_api:
            n_batches = (len(chunks) + batch_size - 1) // batch_size
            print(
                f"Dry-run: would make ~{n_batches} API call(s) to {provider_name} "
                f"(batch_size={batch_size})."
            )
        print("Dry-run mode — no changes written.")
        return

    # ── Initialise infrastructure ───────────────────────────────────────────────
    embedder = get_embedding_provider()
    vector_store = QdrantVectorStore.from_settings()

    if args.recreate_collection:
        print(f"Dropping collection '{settings.qdrant.collection}'…")
        try:
            vector_store.drop_collection()
            print("Collection dropped.")
        except Exception as exc:
            print(f"Warning: could not drop collection: {exc}", file=sys.stderr)

    # ── Re-embed in batches ─────────────────────────────────────────────────────
    errors = 0
    ok = 0
    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Re-embedding", total=len(chunks))

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            progress.update(task, description=f"[cyan]Batch {i // batch_size + 1}")
            try:
                embed_and_upsert_batch(embedder, vector_store, batch)
                ok += len(batch)
                maybe_sleep_between_api_batches(
                    provider_name,
                    batch_start=i,
                    batch_size=batch_size,
                    total_chunks=len(chunks),
                )
            except Exception as exc:
                print(f"\nBatch {i}–{i + len(batch)} failed: {exc}", file=sys.stderr)
                errors += 1
            finally:
                progress.advance(task, len(batch))

    print(f"\n✓ Re-embedded {ok}/{len(chunks)} chunks → Qdrant '{settings.qdrant.collection}'")
    if errors:
        print(f"  {errors} batch(es) failed — check logs above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
