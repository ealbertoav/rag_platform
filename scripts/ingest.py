"""Ingest documents from a file or directory into the RAG platform.

Usage:
    uv run python scripts/ingest.py --source data/raw/
    uv run python scripts/ingest.py --source data/raw/manual.pdf
    uv run python scripts/ingest.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG platform")
    parser.add_argument("--source", help="File or directory to ingest")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save BM25 index after ingestion"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List ingested documents from the metadata store and exit",
    )
    args = parser.parse_args()

    from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline.from_settings()

    if args.list:
        docs = pipeline.list_documents()
        if not docs:
            print("No ingested documents found.")
            return
        for doc in docs:
            print(
                f"{doc.source_path}  chunks={doc.chunk_count}  "
                f"hash={doc.content_hash}  updated={doc.updated_at.isoformat()}"
            )
        return

    if not args.source:
        print("Error: --source is required unless --list is used", file=sys.stderr)
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"Error: {source} does not exist", file=sys.stderr)
        sys.exit(1)

    if source.is_file():
        result = pipeline.ingest_file(source)
        if result.skipped:
            print(f"⊘ {result.source}: skipped (unchanged)")
        else:
            print(f"✓ {result.source}: {result.chunk_count} chunks")
    else:
        results = pipeline.ingest_directory(source)
        ok = [r for r in results if r.error is None and not r.skipped]
        skipped = [r for r in results if r.skipped]
        failed = [r for r in results if r.error is not None]
        print(f"\n{len(ok)}/{len(results)} files ingested successfully")
        if skipped:
            print(f"{len(skipped)} skipped (unchanged)")
        if failed:
            for r in failed:
                print(f"  ✗ {r.source}: {r.error}", file=sys.stderr)

    if args.save:
        pipeline.save_indexes()


if __name__ == "__main__":
    main()
