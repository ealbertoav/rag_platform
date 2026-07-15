from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.exceptions import EmbeddingError

# Domain-layer type aliases — kept here so both repositories and infrastructure
# share the same definitions without any external dependency.
DenseVector = list[float]  # e.g. 1024-dim BGE-M3 output
SparseVector = dict[int, float]  # token_id → weight (BGE-M3 lexical head)


class EmbeddingRepository(ABC):
    """Contract for producing dense and sparse vector representations of text."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Return one dense vector per input text, in the same order."""

    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Return one sparse vector per input text, in the same order."""

    def embed_query(self, texts: list[str]) -> list[DenseVector]:
        """Return one dense vector per query text, optimized for retrieval.

        Providers that distinguish between document and query embeddings
        (Cohere search_query, Voyage query, Gemini RETRIEVAL_QUERY) must
        override this method.  The default delegates to embed() for providers
        that use the same representation for both roles (BGE-M3, Nomic, Qwen,
        OpenAI).
        """
        return self.embed(texts)

    def embed_passage(self, texts: list[str]) -> list[DenseVector]:
        """Return one dense vector per passage text for passage-to-passage similarity.

        Used by MMR diversity ranking and other steps that compare chunk texts
        in document embedding space.  Providers with separate query/document
        modes (Cohere, Voyage, Gemini) must keep this on the document path —
        the default delegates to embed().
        """
        return self.embed(texts)

    def embed_both(self, texts: list[str]) -> tuple[list[DenseVector], list[SparseVector]]:
        """Return (dense, sparse) vectors for each text in a single call.

        Default: two separate calls. Override for a single model forward pass.
        """
        return self.embed(texts), self.embed_sparse(texts)

    def embed_image(self, paths: list[Path]) -> list[DenseVector]:
        """Return one dense vector per image asset, in the same order.

        Unlike embed_query/embed_passage, there is no text-based fallback for
        image embeddings — only multimodal providers (CLIP, Voyage-multimodal
        — T-251) can implement this. The default raises so callers get a
        clear failure instead of a silent, meaningless result.
        """
        raise EmbeddingError(f"{type(self).__name__} does not support image embeddings")
