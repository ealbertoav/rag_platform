"""Ingest documents from a file or directory into the RAG platform.

Usage:
    uv run python scripts/ingest.py --source data/raw/
    uv run python scripts/ingest.py --source data/raw/manual.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG platform")
    parser.add_argument("--source", required=True, help="File or directory to ingest")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save BM25 index after ingestion"
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: {source} does not exist", file=sys.stderr)
        sys.exit(1)

    from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline.from_settings()

    if source.is_file():
        result = pipeline.ingest_file(source)
        print(f"✓ {result.source}: {result.chunk_count} chunks")
    else:
        results = pipeline.ingest_directory(source)
        ok = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        print(f"\n{len(ok)}/{len(results)} files ingested successfully")
        if failed:
            for r in failed:
                print(f"  ✗ {r.source}: {r.error}", file=sys.stderr)

    if args.save:
        pipeline.save_indexes()


if __name__ == "__main__":
    main()
