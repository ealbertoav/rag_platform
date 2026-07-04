from __future__ import annotations

from pathlib import Path

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.infrastructure.vectordb.bm25 import BM25Index


class BM25Retriever:
    """RAG-layer wrapper around a "BM25Index".

    Keeps the RAG pipeline decoupled from the infrastructure-level index
    implementation — the pipeline calls "search()" and receives domain
    entities without needing to know how BM25 is built or persisted.
    """

    def __init__(self, index: BM25Index) -> None:
        self._index = index

    @property
    def bm25_index(self) -> BM25Index:
        return self._index

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_disk(cls, index_path: Path | None = None) -> BM25Retriever:
        """Load an existing index from disk (or create empty if none exists)."""
        return cls(BM25Index.load_or_create(index_path))

    # ── Public ─────────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """Replace the full index."""
        self._index.index(chunks)

    def add(self, chunks: list[Chunk]) -> None:
        """Append chunks and rebuild."""
        self._index.add(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: RetrievalFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (chunk, score) pairs ranked by BM25 score."""
        return self._index.search(query, top_k, filters=filters)

    def save(self) -> None:
        self._index.save()

    def get_by_id(self, chunk_id: str) -> object:
        """Return the ``Chunk`` with *chunk_id* from the index, or ``None``."""
        return self._index.get_by_id(chunk_id)

    @property
    def size(self) -> int:
        return self._index.size
