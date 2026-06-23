from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.core.constants import SUPPORTED_EXTENSIONS
from src.core.exceptions import DocumentLoadError, IngestionError
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.domain.services.ingestion_service import IngestionService
from src.infrastructure.loaders import load_document
from src.infrastructure.vectordb.bm25 import BM25Index

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class IngestionResult:
    source: str
    chunk_count: int
    content_hash: str = ""
    skipped: bool = False
    error: str | None = None


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        f for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class IngestionPipeline:
    """Orchestrates the full ingestion flow for one or many files.

    Flow per file:
      load_document → IngestionService.prepare (chunk and embed)
                    → VectorStoreRepository.upsert
                    → BM25Index.add
    """

    def __init__(
        self,
        service: IngestionService,
        vector_store: VectorStoreRepository,
        bm25: BM25Index,
    ) -> None:
        self._service = service
        self._vector_store = vector_store
        self._bm25 = bm25

    # ── Public ─────────────────────────────────────────────────────────────────

    def ingest_file(self, path: Path) -> IngestionResult:
        """Ingest a single file.  Raises "IngestionError" on unrecoverable failure."""
        try:
            document = load_document(path)
        except DocumentLoadError as exc:
            raise IngestionError(f"Cannot load {path.name}", cause=exc) from exc

        content_hash = _content_hash(document.content)

        chunks = self._service.prepare(document)
        if not chunks:
            return IngestionResult(
                source=str(path), chunk_count=0, content_hash=content_hash
            )

        self._vector_store.upsert(chunks)
        self._bm25.add(chunks)

        logger.info("Ingested %s → %d chunks", path.name, len(chunks))
        return IngestionResult(
            source=str(path), chunk_count=len(chunks), content_hash=content_hash
        )

    def ingest_directory(self, path: Path) -> list[IngestionResult]:
        """Ingest all supported files under a *path*.

        Per-file errors are logged and recorded in the result — the pipeline
        continues with remaining files rather than aborting.
        """
        files = _discover(path)
        if not files:
            logger.warning("No supported files found in %s", path)
            return []

        results: list[IngestionResult] = []
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Ingesting", total=len(files))
            for f in files:
                progress.update(task, description=f"[cyan]{f.name}")
                try:
                    result = self.ingest_file(f)
                except Exception as exc:
                    logger.error("Skipping %s: %s", f.name, exc)
                    result = IngestionResult(source=str(f), chunk_count=0, error=str(exc))
                results.append(result)
                progress.advance(task)

        ok = sum(1 for r in results if r.error is None)
        logger.info("Ingestion complete: %d/%d files succeeded", ok, len(files))
        return results

    def save_indexes(self) -> None:
        """Persist the BM25 index to disk after batch ingestion."""
        self._bm25.save()

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> IngestionPipeline:
        """Build the pipeline from application settings (real dependencies)."""
        from src.core.settings import settings
        from src.infrastructure.embeddings import get_embedding_provider
        from src.infrastructure.vectordb.qdrant import QdrantVectorStore
        from src.rag.chunking import get_chunker

        cfg = settings.chunking
        chunker = get_chunker(
            cfg.strategy,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
        )
        embedder = get_embedding_provider()
        vector_store = QdrantVectorStore.from_settings()
        bm25 = BM25Index.load_or_create()
        service = IngestionService(chunker=chunker, embedder=embedder)
        return cls(service=service, vector_store=vector_store, bm25=bm25)
