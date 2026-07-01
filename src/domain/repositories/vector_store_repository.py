from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.domain.repositories.embedding_repository import DenseVector, SparseVector

# A retrieved result pairs a Chunk with the score assigned by the search.
SearchResult = tuple[Chunk, float]


class VectorStoreRepository(ABC):
    """Contract for persisting and querying chunk vectors."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks (keyed by chunk.id).

        Each chunk must have both *embedding* and *sparse_vector* populated
        before calling this method.
        """

    @abstractmethod
    def search_dense(
        self,
        query_vector: DenseVector,
        top_k: int,
        *,
        type_equals: str | None = None,
        exclude_types: frozenset[str] | None = None,
        document_ids: frozenset[str] | None = None,
        filters: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        """Approximate nearest-neighbor search on the dense index.

        Optional payload filters restrict results by chunk metadata "type":
        *type_equals* keeps only matching types; *exclude_types* drops them.
        *document_ids* scopes results to the given document IDs.
        *filters* adds document scope, metadata exact-match, and related constraints.
        """

    @abstractmethod
    def search_sparse(self, query_sparse: SparseVector, top_k: int) -> list[SearchResult]:
        """Exact or approximate search on the sparse (BM25/lexical) index."""

    @abstractmethod
    def search_hybrid(
        self,
        query_vector: DenseVector,
        query_sparse: SparseVector,
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        """Fused dense and sparse search.

        *alpha* controls the blend: 1.0 = dense only, 0.0 = sparse only.
        Implementations typically use Reciprocal Rank Fusion (RRF).
        """

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID (used during re-ingestion)."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently stored."""
