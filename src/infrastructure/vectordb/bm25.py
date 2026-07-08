from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import types
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rank_bm25 import BM25Okapi

from src.core.constants import BM25_INDEX_PATH, BM25_LEGACY_PICKLE_PATH
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.rag.retrieval.filters import chunk_matches_filter

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25_disk import DiskBM25Index

logger = logging.getLogger(__name__)

_INDEX_FORMAT_VERSION = 1

BM25Backend = Literal["memory", "disk"]

_fcntl: types.ModuleType | None = None
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows and other non-Unix platforms
    _fcntl = None

fcntl = _fcntl


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Generator[None, None, None]:
    """Serialize index migration across processes on Unix-like systems."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _tokenize(text: str) -> list[str]:
    """Lowercase word-split tokenizer used for both indexing and querying."""
    return text.lower().split()


class BM25Index:
    """In-memory BM25 index with optional persistence to disk.

    The underlying `rank-bm25` model does not support incremental updates;
    `add()` rebuilds the index from the full chunk list each time.  This is
    acceptable for corpus sizes typical in enterprise RAG (≤ millions of
    chunks) where a full rebuild takes well under a second.

    Chunks are persisted as JSON (not pickle) to avoid unsafe deserialization.
    """

    def __init__(self, index_path: Path | None = None) -> None:
        self._path: Path = index_path or BM25_INDEX_PATH
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._defer_rebuild_depth = 0
        self._needs_rebuild = False
        self._dirty = False
        self._mutation_generation = 0

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """Replace the entire index with *chunks* and rebuild."""
        with self._lock:
            self._chunks = list(chunks)
            self._rebuild()
            self._needs_rebuild = False
            self._mark_dirty()
        logger.debug("BM25 index built: %d chunks", len(self._chunks))

    def add(self, chunks: list[Chunk]) -> None:
        """Append *chunks* to the index and rebuild unless inside: meth:`deferred_rebuild`.

        Existing chunks with the same "id" are replaced (deduplication).
        """
        with self._lock:
            incoming_ids = {c.id for c in chunks}
            self._chunks = [c for c in self._chunks if c.id not in incoming_ids]
            self._chunks.extend(chunks)
            self._schedule_rebuild()
            self._mark_dirty()
        logger.debug("BM25 index updated: +%d chunks, total %d", len(chunks), len(self._chunks))

    def remove_by_ids(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID and rebuild the index unless inside: meth:`deferred_rebuild`."""
        if not chunk_ids:
            return
        with self._lock:
            remove = set(chunk_ids)
            before = len(self._chunks)
            self._chunks = [c for c in self._chunks if c.id not in remove]
            self._schedule_rebuild()
            self._mark_dirty()
        logger.debug("BM25 index removed %d chunks", before - len(self._chunks))

    def rebuild(self) -> None:
        """Rebuild the in-memory BM25 model from the current chunk list."""
        with self._lock:
            self._rebuild()
            self._needs_rebuild = False

    @contextmanager
    def deferred_rebuild(self) -> Generator[BM25Index, None, None]:
        """Defer rebuilds until the context exits, then rebuild once if needed."""
        with self._lock:
            self._defer_rebuild_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._defer_rebuild_depth -= 1
                if self._defer_rebuild_depth == 0:
                    self._ensure_built()

    def remove_by_document_id(self, document_id: str) -> list[str]:
        """Remove all chunks belonging to *document_id*. Returns removed chunk IDs."""
        with self._lock:
            removed = [c.id for c in self._chunks if c.document_id == document_id]
        if removed:
            self.remove_by_ids(removed)
        return removed

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: RetrievalFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (chunk, score) pairs sorted by BM25 score desc."""
        with self._lock:
            self._ensure_built()
            bm25 = self._bm25
            chunks = self._chunks
            if bm25 is None or not chunks:
                return []
            tokens = _tokenize(query)
            scores: list[float] = bm25.get_scores(tokens).tolist()
            ranked = sorted(
                (
                    (chunk, score)
                    for chunk, score in zip(chunks, scores, strict=True)
                    if score > 0 and chunk_matches_filter(chunk, filters)
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:top_k]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist chunk metadata as JSON and rebuild BM25 on a load."""
        with self._lock:
            if not self._dirty:
                logger.debug("BM25 index unchanged — skipping save")
                return
            self._ensure_built()
            chunks = list(self._chunks)
            generation_at_snapshot = self._mutation_generation
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_FORMAT_VERSION,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
        tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            tmp_path.replace(self._path)
            with self._lock:
                if self._mutation_generation == generation_at_snapshot:
                    self._dirty = False
            logger.info("BM25 index saved to %s", self._path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise VectorStoreError(f"Cannot save BM25 index to {self._path}", cause=exc) from exc

    def load(self) -> None:
        """Load chunks from JSON at *self._path*."""
        try:
            with self._path.open(encoding="utf-8") as fh:
                payload: dict[str, Any] = json.load(fh)
        except FileNotFoundError as exc:
            raise VectorStoreError(f"BM25 index not found at {self._path}", cause=exc) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Cannot load BM25 index from {self._path}", cause=exc) from exc

        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise VectorStoreError(f"Invalid BM25 index format at {self._path}")

        with self._lock:
            self._chunks = [Chunk.model_validate(item) for item in raw_chunks]
            self._rebuild()
            self._needs_rebuild = False
            self._dirty = False
        logger.info("BM25 index loaded from %s (%d chunks)", self._path, len(self._chunks))

    @classmethod
    def load_or_create(
        cls,
        index_path: Path | None = None,
        *,
        backend: BM25Backend | None = None,
    ) -> BM25Index | DiskBM25Index:
        """Return a loaded index if the file exists, otherwise an empty one.

        Backend selection (T-165):
        - Explicit "backend=" always wins.
        - With no "index_path", falls back to "settings.retrieval.bm25.backend"
          (default "memory"). "disk" opens
          "DiskBM25Index" at "retrieval.bm25.disk_path".
        - An explicit "index_path" without "backend=" always uses the
          legacy JSON memory index so eval caches, "BM25Retriever.from_disk",
          and tests keep working when the global setting is "disk".
          Pass "backend="disk"" to treat that path as a segmented root.
        """
        from src.core.settings import settings

        if backend is not None:
            selected: BM25Backend = backend
        elif index_path is not None:
            selected = "memory"
        else:
            selected = settings.retrieval.bm25.backend

        if selected == "disk":
            from src.infrastructure.vectordb.bm25_disk import DiskBM25Index

            disk_path = (
                index_path if index_path is not None else Path(settings.retrieval.bm25.disk_path)
            )
            return DiskBM25Index.load_or_create(
                disk_path,
                segment_size=settings.retrieval.bm25.segment_size,
            )

        json_path = index_path or BM25_INDEX_PATH
        legacy_path = (
            BM25_LEGACY_PICKLE_PATH if index_path is None else json_path.with_suffix(".pkl")
        )
        instance = cls(index_path)
        instance.bootstrap_from_disk(legacy_path)
        return instance

    def bootstrap_from_disk(self, legacy_path: Path) -> None:
        """Load JSON at the instance path or migrate a legacy pickle index."""
        if self._path.exists():
            self.load()
        elif legacy_path.exists():
            self._migrate_legacy_pickle(legacy_path)

    # ── Properties ─────────────────────────────────────────────────────────────

    def _read_size(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _read_chunks(self) -> list[Chunk]:
        """Return a snapshot of all indexed chunks."""
        with self._lock:
            return list(self._chunks)

    def iter_chunks(self) -> Generator[Chunk, None, None]:
        """Yield indexed chunks one at a time without copying the full corpus."""
        with self._lock:
            yield from self._chunks

    @property
    def size(self) -> int:
        return self._read_size()

    @property
    def chunks(self) -> list[Chunk]:
        return self._read_chunks()

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with *chunk_id*, or "None" if not indexed."""
        with self._lock:
            for chunk in self._chunks:
                if chunk.id == chunk_id:
                    return chunk
        return None

    def update_chunk_metadata(self, chunk_id: str, updates: dict[str, object]) -> bool:
        """Merge *updates* into the indexed chunk's metadata. Returns False when missing."""
        if not updates:
            return False
        with self._lock:
            for index, chunk in enumerate(self._chunks):
                if chunk.id != chunk_id:
                    continue
                metadata = dict(chunk.metadata)
                metadata.update(updates)
                self._chunks[index] = chunk.model_copy(update={"metadata": metadata})
                return True
        return False

    # ── Internals ──────────────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._mutation_generation += 1

    def _migrate_legacy_pickle(self, legacy_path: Path) -> None:
        """Migrate a legacy pickle index to JSON under an exclusive file lock."""
        lock_path = self._path.with_suffix(f"{self._path.suffix}.lock")
        with _exclusive_file_lock(lock_path):
            if self._path.exists():
                self.load()
                return
            self._load_legacy_pickle(legacy_path)
            self.save()
            logger.info("Migrated BM25 index from pickle to JSON at %s", self._path)

    def _load_legacy_pickle(self, path: Path) -> None:
        """Load a legacy pickle index and rebuild the in-memory BM25 model."""
        try:
            with path.open("rb") as fh:
                payload: dict[str, Any] = pickle.load(fh)  # noqa: S301
        except (OSError, pickle.UnpicklingError) as exc:
            raise VectorStoreError(f"Cannot load legacy BM25 index from {path}", cause=exc) from exc

        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise VectorStoreError(f"Invalid legacy BM25 index format at {path}")

        with self._lock:
            self._chunks = list(chunks)
            self._rebuild()
            self._needs_rebuild = False
            self._mark_dirty()
        logger.info("BM25 legacy pickle loaded from %s (%d chunks)", path, len(self._chunks))

    def _ensure_built(self) -> None:
        """Rebuild the BM25 model when writes were deferred."""
        if self._needs_rebuild:
            self._rebuild()
            self._needs_rebuild = False

    def _schedule_rebuild(self) -> None:
        if self._defer_rebuild_depth > 0:
            self._needs_rebuild = True
            return
        self._rebuild()
        self._needs_rebuild = False

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
