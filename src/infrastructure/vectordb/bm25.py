from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.core.constants import BM25_INDEX_PATH
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase word-split tokenizer used for both indexing and querying."""
    return text.lower().split()


class BM25Index:
    """In-memory BM25 index with optional persistence to disk.

    The underlying `rank-bm25` model does not support incremental updates;
    `add()` rebuilds the index from the full chunk list each time.  This is
    acceptable for corpus sizes typical in enterprise RAG (≤ millions of
    chunks) where a full rebuild takes well under a second.
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
        existing_ids = {c.id for c in self._chunks}
        new = [c for c in chunks if c.id not in existing_ids]
        self._chunks.extend(new)
        self._rebuild()
        logger.debug("BM25 index updated: +%d chunks, total %d", len(new), len(self._chunks))

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
        """Pickle the index to *self._path*."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"chunks": self._chunks, "bm25": self._bm25}
        try:
            with self._path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("BM25 index saved to %s", self._path)
        except OSError as exc:
            raise VectorStoreError(f"Cannot save BM25 index to {self._path}", cause=exc) from exc

    def load(self) -> None:
        """Load the index from *self._path*.  Raises "VectorStoreError" if missing."""
        try:
            with self._path.open("rb") as fh:
                payload: dict[str, Any] = pickle.load(fh)  # noqa: S301
            self._chunks = payload["chunks"]
            self._bm25 = payload["bm25"]
            logger.info("BM25 index loaded from %s (%d chunks)", self._path, len(self._chunks))
        except FileNotFoundError as exc:
            raise VectorStoreError(f"BM25 index not found at {self._path}", cause=exc) from exc
        except (OSError, KeyError, pickle.UnpicklingError) as exc:
            raise VectorStoreError(f"Cannot load BM25 index from {self._path}", cause=exc) from exc

    @classmethod
    def load_or_create(cls, index_path: Path | None = None) -> BM25Index:
        """Return a loaded index if the file exists, otherwise an empty one."""
        instance = cls(index_path)
        path = instance._path
        if path.exists():
            instance.load()
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

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
