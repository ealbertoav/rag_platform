from __future__ import annotations

from abc import ABC, abstractmethod

# Domain-layer type aliases — kept here so both repositories and infrastructure
# share the same definitions without any external dependency.
DenseVector = list[float]          # e.g. 1024-dim BGE-M3 output
SparseVector = dict[int, float]    # token_id → weight (BGE-M3 lexical head)


class EmbeddingRepository(ABC):
    """Contract for producing dense and sparse vector representations of text."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Return one dense vector per input text, in the same order."""

    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return one sparse vector per input text, in the same order."""

    def embed_both(self, texts: list[str]) -> tuple[list[DenseVector], list[SparseVector]]:
        """Return (dense, sparse) vectors for each text in a single call.

        Default: two separate calls. Override for a single model forward pass.
        """
        return self.embed(texts), self.embed_sparse(texts)
