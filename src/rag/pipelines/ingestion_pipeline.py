from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from src.core.constants import SUPPORTED_EXTENSIONS
from src.core.exceptions import DocumentLoadError, IngestionError
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.metadata_repository import DocumentRecord, MetadataRepository
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.domain.services.ingestion_service import IngestionService
from src.infrastructure.loaders import load_document
from src.infrastructure.vectordb.bm25 import BM25Index
from src.rag.ingestion.ocr_fallback import apply_ocr_fallback
from src.rag.ingestion.table_chunker import (
    build_table_chunks,
    existing_table_chunk_ids,
    merged_table_chunk_ids,
    retained_table_chunk_ids_on_embed_failure,
    stale_table_ids_safe_to_purge,
    table_chunks_needing_upsert,
)

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25_disk import DiskBM25Index
    from src.rag.enrichment.document_augmentation import DocumentAugmentor
    from src.rag.enrichment.hierarchical_indexer import HierarchicalIndexer
    from src.rag.enrichment.hype_indexer import HyPEIndexer
    from src.rag.ingestion.graph_indexer import GraphIndexer
    from src.rag.ingestion.table_chunker import TableChunker

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
      load_document → optional OCR fallback (scanned PDFs) → dedup check
                    → IngestionService.prepare (chunk and embed)
                    → optional graph entity extraction
                    → VectorStoreRepository.upsert
                    → BM25Index.add
                    → MetadataRepository.upsert
    """

    def __init__(
        self,
        service: IngestionService,
        vector_store: VectorStoreRepository,
        bm25: BM25Index | DiskBM25Index,
        metadata: MetadataRepository | None = None,
        graph_indexer: GraphIndexer | None = None,
        augmentor: DocumentAugmentor | None = None,
        hype_indexer: HyPEIndexer | None = None,
        hierarchical_indexer: HierarchicalIndexer | None = None,
        table_chunker: TableChunker | None = None,
    ) -> None:
        self._service: IngestionService = service
        self._vector_store: VectorStoreRepository = vector_store
        self._bm25: BM25Index | DiskBM25Index = bm25
        self._metadata: MetadataRepository | None = metadata
        self._graph_indexer: GraphIndexer | None = graph_indexer
        self._augmentor: DocumentAugmentor | None = augmentor
        self._hype_indexer: HyPEIndexer | None = hype_indexer
        self._hierarchical_indexer: HierarchicalIndexer | None = hierarchical_indexer
        self._table_chunker: TableChunker | None = table_chunker

    # ── Public ─────────────────────────────────────────────────────────────────

    def ingest_file(self, path: Path) -> IngestionResult:
        """Ingest a single file.  Raises "IngestionError" on unrecoverable failure."""
        t0 = time.monotonic()
        source = str(path.resolve())

        try:
            document = load_document(path)
        except DocumentLoadError as exc:
            raise IngestionError(f"Cannot load {path.name}", cause=exc) from exc

        document = apply_ocr_fallback(document, path)
        doc_hash = content_hash(source, document.content)
        old_chunk_ids: list[str] = []

        if self._metadata is not None:
            existing = self._metadata.get_by_source(source)
            if existing is not None:
                document = document.model_copy(update={"id": existing.id})
            if existing is not None and existing.content_hash == doc_hash:
                if self._requires_full_reindex_on_skip():
                    old_chunk_ids = self._metadata.get_chunk_ids(existing.id)
                else:
                    backfill = self._backfill_table_chunks_on_skip(
                        document,
                        source,
                        doc_hash,
                        existing,
                        t0,
                    )
                    if backfill is not None:
                        return backfill
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    _ = self._metadata.upsert_document(
                        source,
                        doc_hash,
                        self._metadata.get_chunk_ids(existing.id),
                        chunk_count=existing.chunk_count,
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
                old_chunk_ids = self._metadata.get_chunk_ids(existing.id)

        with self._bm25.deferred_rebuild():
            chunks = self._service.prepare(document)
            indexed_chunks = list(chunks)

            if chunks:
                if self._hierarchical_indexer is not None:
                    chunks, summary_chunks = self._hierarchical_indexer.index(document, chunks)
                    indexed_chunks = list(chunks)
                    indexed_chunks.extend(summary_chunks)
                if self._augmentor is not None:
                    augmented = self._augmentor.augment(chunks)
                    indexed_chunks.extend(augmented)
                if self._hype_indexer is not None:
                    hype_chunks = self._hype_indexer.index(chunks)
                    indexed_chunks.extend(hype_chunks)

            built_table_chunks: list[Chunk] = []
            embedded_table_chunks: list[Chunk] = []
            if self._table_chunker is not None:
                built_table_chunks = build_table_chunks(document)
                embedded_table_chunks = self._table_chunker.index(document)
                indexed_chunks.extend(embedded_table_chunks)

            if not indexed_chunks:
                retained = self._retained_table_ids_on_embed_failure(
                    source,
                    document,
                    old_chunk_ids,
                    built_table_chunks,
                    embedded_table_chunks,
                )
                self._purge_superseded_chunks(old_chunk_ids, retained_chunk_ids=retained)
                elapsed_ms = (time.monotonic() - t0) * 1000
                if self._metadata is not None:
                    merged_ids = list(retained) if retained else []
                    _ = self._metadata.upsert_document(
                        source,
                        doc_hash,
                        merged_ids,
                        duration_ms=elapsed_ms,
                    )
                return IngestionResult(source=source, chunk_count=0, content_hash=doc_hash)

            self._index_graph(chunks, document.id)
            self._vector_store.upsert(indexed_chunks)
            retained = {chunk.id for chunk in indexed_chunks}
            retained.update(
                self._retained_table_ids_on_embed_failure(
                    source,
                    document,
                    old_chunk_ids,
                    built_table_chunks,
                    embedded_table_chunks,
                )
            )
            self._purge_superseded_chunks(old_chunk_ids, retained_chunk_ids=retained)
            self._bm25_add(indexed_chunks)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if self._metadata is not None:
            stored_chunk_ids = [c.id for c in indexed_chunks]
            stored_chunk_ids.extend(
                chunk_id
                for chunk_id in self._retained_table_ids_on_embed_failure(
                    source,
                    document,
                    old_chunk_ids,
                    built_table_chunks,
                    embedded_table_chunks,
                )
                if chunk_id not in stored_chunk_ids
            )
            _ = self._metadata.upsert_document(
                source,
                doc_hash,
                list(dict.fromkeys(stored_chunk_ids)),
                chunk_count=len(chunks),
                duration_ms=elapsed_ms,
            )

        logger.info(
            "Ingested %s → %d chunks (%d indexed extras)",
            path.name,
            len(chunks),
            len(indexed_chunks) - len(chunks),
        )
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
        with (
            self._bm25.deferred_rebuild(),
            Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            ) as progress,
        ):
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

    def _bm25_add(self, indexed_chunks: list[Chunk]) -> None:
        indexable = _bm25_indexable(indexed_chunks)
        if indexable:
            self._bm25.add(indexable)

    def _requires_full_reindex_on_skip(self) -> bool:
        """LLM-based supplemental indexers need to prepare() and cannot be backfilled cheaply."""
        return any(
            (
                self._augmentor is not None,
                self._hype_indexer is not None,
                self._hierarchical_indexer is not None,
            )
        )

    def _backfill_table_chunks_on_skip(
        self,
        document: Document,
        source: str,
        doc_hash: str,
        existing: DocumentRecord,
        t0: float,
    ) -> IngestionResult | None:
        """Sync table chunks when base content is unchanged."""
        if self._table_chunker is None or self._metadata is None:
            return None

        built_chunks = build_table_chunks(document)
        table_chunks = self._table_chunker.index(document) if built_chunks else []

        existing_ids = list(self._metadata.get_chunk_ids(existing.id))
        desired_table_ids = {chunk.id for chunk in built_chunks}
        known_table_ids = existing_table_chunk_ids(
            source,
            existing_ids,
            document=document,
            bm25=self._bm25,
        )
        stale_table_ids = [
            chunk_id
            for chunk_id in existing_ids
            if chunk_id in known_table_ids and chunk_id not in desired_table_ids
        ]
        chunks_to_upsert = table_chunks_needing_upsert(
            table_chunks,
            existing_ids,
            bm25=self._bm25,
        )
        stale_to_purge = stale_table_ids_safe_to_purge(
            document,
            built_chunks,
            table_chunks,
            stale_table_ids,
        )
        if not chunks_to_upsert and not stale_to_purge:
            return None

        with self._bm25.deferred_rebuild():
            if chunks_to_upsert:
                self._vector_store.upsert(chunks_to_upsert)
                self._bm25_add(chunks_to_upsert)
            if stale_to_purge:
                self._purge_superseded_chunks(stale_to_purge)

        non_table_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in known_table_ids]
        merged_table_ids_list = merged_table_chunk_ids(
            existing_ids,
            known_table_ids,
            document,
            built_chunks,
            table_chunks,
        )
        merged_ids = list(dict.fromkeys(non_table_ids + merged_table_ids_list))
        elapsed_ms = (time.monotonic() - t0) * 1000
        _ = self._metadata.upsert_document(
            source,
            doc_hash,
            merged_ids,
            chunk_count=existing.chunk_count,
            duration_ms=elapsed_ms,
        )
        if chunks_to_upsert:
            logger.info(
                "Backfilled %d table chunk(s) for %s (unchanged content)",
                len(chunks_to_upsert),
                Path(source).name,
            )
        if stale_to_purge:
            logger.info(
                "Purged %d stale table chunk(s) for %s (unchanged content)",
                len(stale_to_purge),
                Path(source).name,
            )
        return IngestionResult(
            source=source,
            chunk_count=existing.chunk_count,
            content_hash=doc_hash,
        )

    def _retained_table_ids_on_embed_failure(
        self,
        source: str,
        document: Document,
        old_chunk_ids: list[str],
        built_table_chunks: list[Chunk],
        embedded_table_chunks: list[Chunk],
    ) -> set[str]:
        if self._table_chunker is None or not old_chunk_ids:
            return set()
        return retained_table_chunk_ids_on_embed_failure(
            source,
            document,
            old_chunk_ids,
            built_table_chunks,
            embedded_table_chunks,
            bm25=self._bm25,
        )

    def _purge_superseded_chunks(
        self,
        chunk_ids: list[str],
        *,
        retained_chunk_ids: set[str] | None = None,
    ) -> None:
        """Remove superseded chunk IDs from dense and lexical indexes together.

        *retained_chunk_ids* excludes IDs present in the new indexed batch so
        stable table chunk IDs are not deleted immediately after upsert.
        """
        if not chunk_ids:
            return
        retained = retained_chunk_ids or set()
        superseded = [chunk_id for chunk_id in chunk_ids if chunk_id not in retained]
        if superseded:
            self._vector_store.delete(superseded)
            self._bm25.remove_by_ids(superseded)

    def _remove_document_chunks(self, metadata_doc_id: str) -> None:
        if self._metadata is None:
            return
        self._purge_superseded_chunks(self._metadata.get_chunk_ids(metadata_doc_id))

    def _index_graph(self, chunks: list[Chunk], document_id: str) -> None:
        if self._graph_indexer is None:
            return
        try:
            self._graph_indexer.index_chunks(chunks, document_id)
        except Exception as exc:
            logger.warning("Graph indexing failed for %s: %s", document_id, exc)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        bm25: BM25Index | DiskBM25Index | None = None,
        vector_store: VectorStoreRepository | None = None,
    ) -> IngestionPipeline:
        """Build the pipeline from application settings (real dependencies)."""
        from src.core.settings import settings
        from src.infrastructure.embeddings import get_embedding_provider
        from src.infrastructure.metadata.sqlite_store import SQLiteMetadataStore
        from src.infrastructure.vectordb.qdrant import QdrantVectorStore
        from src.rag.chunking import get_chunker

        cfg = settings.chunking
        chunker_kwargs: dict[str, object] = {
            "chunk_size": cfg.chunk_size,
            "overlap": cfg.overlap,
        }
        if cfg.strategy == "proposition":
            from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

            chunker_kwargs["llm"] = LlamaCppProvider.from_settings()
            chunker_kwargs["quality_threshold"] = cfg.proposition.quality_threshold
            chunker_kwargs["overlap"] = 0
        chunker = get_chunker(
            cfg.strategy,
            use_contextual_headers=cfg.contextual_headers.enabled,
            **chunker_kwargs,
        )
        embedder = get_embedding_provider()
        if vector_store is None:
            vector_store = QdrantVectorStore.from_settings()
        bm25_index = bm25 or BM25Index.load_or_create()
        service = IngestionService(chunker=chunker, embedder=embedder)

        metadata = SQLiteMetadataStore.from_settings() if settings.metadata.enabled else None
        graph_indexer = _build_graph_indexer() if settings.neo4j.enabled else None
        augmentor = _build_augmentor(embedder, cfg.augmentation)
        hype_indexer = _build_hype_indexer(embedder, settings.retrieval.hype)
        hierarchical_indexer = _build_hierarchical_indexer(embedder, cfg.hierarchical)
        table_chunker = _build_table_chunker(embedder, settings.parsing.table_chunks)

        return cls(
            service=service,
            vector_store=vector_store,
            bm25=bm25_index,
            metadata=metadata,
            graph_indexer=graph_indexer,
            augmentor=augmentor,
            hype_indexer=hype_indexer,
            hierarchical_indexer=hierarchical_indexer,
            table_chunker=table_chunker,
        )


def _build_graph_indexer() -> GraphIndexer | None:
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


def _build_augmentor(embedder: EmbeddingRepository, cfg: object) -> DocumentAugmentor | None:
    """Build document augmentor when synthetic question generation is enabled."""
    if not getattr(cfg, "enabled", False):
        return None
    try:
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
        from src.rag.enrichment.document_augmentation import DocumentAugmentor

        llm = LlamaCppProvider.from_settings()
        return DocumentAugmentor(
            llm=llm,
            embedder=embedder,
            n_questions=getattr(cfg, "n_questions", 3),
        )
    except Exception as exc:
        logger.warning("Document augmentor unavailable: %s", exc)
        return None


def _build_hype_indexer(embedder: EmbeddingRepository, cfg: object) -> HyPEIndexer | None:
    """Build HyPE indexer when hypothetical prompt embeddings are enabled."""
    if not getattr(cfg, "enabled", False):
        return None
    try:
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
        from src.rag.enrichment.hype_indexer import HyPEIndexer

        llm = LlamaCppProvider.from_settings()
        return HyPEIndexer(
            llm=llm,
            embedder=embedder,
            n_questions=getattr(cfg, "n_questions", 3),
        )
    except Exception as exc:
        logger.warning("HyPE indexer unavailable: %s", exc)
        return None


def _build_hierarchical_indexer(
    embedder: EmbeddingRepository,
    cfg: object,
) -> HierarchicalIndexer | None:
    """Build a hierarchical summary indexer when two-tier indexing is enabled."""
    if not getattr(cfg, "enabled", False):
        return None
    try:
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
        from src.rag.enrichment.hierarchical_indexer import HierarchicalIndexer

        llm = LlamaCppProvider.from_settings()
        return HierarchicalIndexer(llm=llm, embedder=embedder)
    except Exception as exc:
        logger.warning("Hierarchical indexer unavailable: %s", exc)
        return None


def _build_table_chunker(embedder: EmbeddingRepository, cfg: object) -> TableChunker | None:
    """Build a structured table chunker when table chunking is enabled."""
    if not getattr(cfg, "enabled", False):
        return None
    from src.rag.ingestion.table_chunker import TableChunker

    return TableChunker(embedder=embedder)


def _bm25_indexable(chunks: list[Chunk]) -> list[Chunk]:
    """Exclude vector-only index points from the lexical BM25 index."""
    from src.rag.enrichment.hierarchical_indexer import is_summary_chunk
    from src.rag.enrichment.hype_indexer import is_hype_question

    return [
        chunk for chunk in chunks if not is_hype_question(chunk) and not is_summary_chunk(chunk)
    ]
