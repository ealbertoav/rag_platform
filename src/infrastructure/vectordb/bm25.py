from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.core.constants import BM25_INDEX_PATH, BM25_LEGACY_PICKLE_PATH
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk

logger = logging.getLogger(__name__)

_INDEX_FORMAT_VERSION = 1


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
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """Replace the entire index with *chunks* and rebuild."""
        self._chunks = list(chunks)
        self._rebuild()
        logger.debug("BM25 index built: %d chunks", len(self._chunks))

    def add(self, chunks: list[Chunk]) -> None:
        """Append *chunks* to the index and rebuild.

        Existing chunks with the same "id" are replaced (deduplication).
        """
        incoming_ids = {c.id for c in chunks}
        self._chunks = [c for c in self._chunks if c.id not in incoming_ids]
        self._chunks.extend(chunks)
        self._rebuild()
        logger.debug("BM25 index updated: +%d chunks, total %d", len(chunks), len(self._chunks))

    def remove_by_ids(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID and rebuild the index."""
        if not chunk_ids:
            return
        remove = set(chunk_ids)
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.id not in remove]
        self._rebuild()
        logger.debug("BM25 index removed %d chunks", before - len(self._chunks))

    def remove_by_document_id(self, document_id: str) -> list[str]:
        """Remove all chunks belonging to *document_id*. Returns removed chunk IDs."""
        removed = [c.id for c in self._chunks if c.document_id == document_id]
        if removed:
            self.remove_by_ids(removed)
        return removed

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (chunk, score) pairs sorted by BM25 score desc."""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores: list[float] = self._bm25.get_scores(tokens).tolist()
        ranked = sorted(
            (
                (chunk, score)
                for chunk, score in zip(self._chunks, scores, strict=True)
                if score > 0
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist chunk metadata as JSON and rebuild BM25 on a load."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_FORMAT_VERSION,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        try:
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            logger.info("BM25 index saved to %s", self._path)
        except OSError as exc:
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

        self._chunks = [Chunk.model_validate(item) for item in raw_chunks]
        self._rebuild()
        logger.info("BM25 index loaded from %s (%d chunks)", self._path, len(self._chunks))

    @classmethod
    def load_or_create(cls, index_path: Path | None = None) -> BM25Index:
        """Return a loaded index if the file exists, otherwise an empty one."""
        instance = cls(index_path)
        json_path = instance._path
        legacy_path = (
            BM25_LEGACY_PICKLE_PATH if index_path is None else json_path.with_suffix(".pkl")
        )

        if json_path.exists():
            instance.load()
        elif legacy_path.exists():
            instance._load_legacy_pickle(legacy_path)
            instance.save()
            logger.info("Migrated BM25 index from pickle to JSON at %s", json_path)
        return instance

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """Return a snapshot of all indexed chunks."""
        return list(self._chunks)

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Return the chunk with *chunk_id*, or "None" if not indexed."""
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    # ── Internals ──────────────────────────────────────────────────────────────

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

        self._chunks = list(chunks)
        self._rebuild()
        logger.info("BM25 legacy pickle loaded from %s (%d chunks)", path, len(self._chunks))

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
