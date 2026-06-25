from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.core.constants import SUPPORTED_EXTENSIONS
from src.core.exceptions import DocumentLoadError, IngestionError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.metadata_repository import DocumentRecord, MetadataRepository
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


def content_hash(source: str, text: str) -> str:
    """Stable hash for deduplication: sha256(normalized_text + source_path)."""
    normalized = text.strip()
    payload = f"{normalized}|{source}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class IngestionPipeline:
    """Orchestrates the full ingestion flow for one or many files.

    Flow per file:
      load_document → dedup check → IngestionService.prepare (chunk and embed)
                    → optional graph entity extraction
                    → VectorStoreRepository.upsert
                    → BM25Index.add
                    → MetadataRepository.upsert
    """

    def __init__(
        self,
        service: IngestionService,
        vector_store: VectorStoreRepository,
        bm25: BM25Index,
        metadata: MetadataRepository | None = None,
        graph_indexer: object | None = None,
    ) -> None:
        self._service = service
        self._vector_store = vector_store
        self._bm25 = bm25
        self._metadata = metadata
        self._graph_indexer = graph_indexer

    # ── Public ─────────────────────────────────────────────────────────────────

    def ingest_file(self, path: Path) -> IngestionResult:
        """Ingest a single file.  Raises "IngestionError" on unrecoverable failure."""
        t0 = time.monotonic()
        source = str(path.resolve())

        try:
            document = load_document(path)
        except DocumentLoadError as exc:
            raise IngestionError(f"Cannot load {path.name}", cause=exc) from exc

        doc_hash = content_hash(source, document.content)

        if self._metadata is not None:
            existing = self._metadata.get_by_source(source)
            if existing is not None and existing.content_hash == doc_hash:
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._metadata.upsert_document(
                    source,
                    doc_hash,
                    self._metadata.get_chunk_ids(existing.id),
                    duration_ms=elapsed_ms,
                    skipped=True,
                )
                logger.info("Skipped %s (unchanged)", path.name)
                return IngestionResult(
                    source=source,
                    chunk_count=existing.chunk_count,
                    content_hash=doc_hash,
                    skipped=True,
                )
            if existing is not None:
                self._remove_document_chunks(existing.id)

        chunks = self._service.prepare(document)
        if not chunks:
            elapsed_ms = (time.monotonic() - t0) * 1000
            if self._metadata is not None:
                self._metadata.upsert_document(source, doc_hash, [], duration_ms=elapsed_ms)
            return IngestionResult(source=source, chunk_count=0, content_hash=doc_hash)

        self._index_graph(chunks, document.id)
        self._vector_store.upsert(chunks)
        self._bm25.add(chunks)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if self._metadata is not None:
            self._metadata.upsert_document(
                source,
                doc_hash,
                [c.id for c in chunks],
                duration_ms=elapsed_ms,
            )

        logger.info("Ingested %s → %d chunks", path.name, len(chunks))
        return IngestionResult(
            source=source,
            chunk_count=len(chunks),
            content_hash=doc_hash,
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

    def list_documents(self) -> list[DocumentRecord]:
        """Return ingested document records when metadata store is enabled."""
        if self._metadata is None:
            return []
        return self._metadata.list_documents()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _remove_document_chunks(self, metadata_doc_id: str) -> None:
        if self._metadata is None:
            return
        old_ids = self._metadata.get_chunk_ids(metadata_doc_id)
        if old_ids:
            self._vector_store.delete(old_ids)
            self._bm25.remove_by_ids(old_ids)

    def _index_graph(self, chunks: list[Chunk], document_id: str) -> None:
        if self._graph_indexer is None:
            return
        try:
            self._graph_indexer.index_chunks(chunks, document_id)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Graph indexing failed for %s: %s", document_id, exc)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> IngestionPipeline:
        """Build the pipeline from application settings (real dependencies)."""
        from src.core.settings import settings
        from src.infrastructure.embeddings import get_embedding_provider
        from src.infrastructure.metadata.sqlite_store import SQLiteMetadataStore
        from src.infrastructure.vectordb.qdrant import QdrantVectorStore
        from src.rag.chunking import get_chunker

        cfg = settings.chunking
        chunker = get_chunker(
            cfg.strategy,
            use_contextual_headers=cfg.contextual_headers.enabled,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
        )
        embedder = get_embedding_provider()
        vector_store = QdrantVectorStore.from_settings()
        bm25 = BM25Index.load_or_create()
        service = IngestionService(chunker=chunker, embedder=embedder)

        metadata = SQLiteMetadataStore.from_settings() if settings.metadata.enabled else None
        graph_indexer = _build_graph_indexer() if settings.neo4j.enabled else None

        return cls(
            service=service,
            vector_store=vector_store,
            bm25=bm25,
            metadata=metadata,
            graph_indexer=graph_indexer,
        )


def _build_graph_indexer() -> object | None:
    """Build graph entity indexer when Neo4j ingest extraction is enabled."""
    from src.core.settings import settings

    if not settings.neo4j.extract_entities_on_ingest:
        return None
    try:
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
        from src.infrastructure.vectordb.neo4j_graph import Neo4jGraphRepository
        from src.rag.ingestion.graph_indexer import GraphIndexer
        from src.rag.retrieval.graph_retriever import EntityExtractor

        llm = LlamaCppProvider.from_settings()
        return GraphIndexer(
            extractor=EntityExtractor(llm=llm),
            graph=Neo4jGraphRepository.from_settings(),
        )
    except Exception as exc:
        logger.warning("Graph indexer unavailable: %s", exc)
        return None
