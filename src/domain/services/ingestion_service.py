from __future__ import annotations

import logging

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.chunking import Chunker

logger = logging.getLogger(__name__)


class IngestionService:
    """Domain service: chunk a Document and attach dense and sparse embeddings.

    The pipeline layer handles storage (Qdrant, BM25), so this service
    remains free of infrastructure dependencies.
    """

    def __init__(self, chunker: Chunker, embedder: EmbeddingRepository) -> None:
        self._chunker = chunker
        self._embedder = embedder

    def prepare(self, document: Document) -> list[Chunk]:
        """Chunk *document* and embed each chunk.

        Returns a list of Chunks with "embedding" and "sparse_vector"
        populated, ready for vector-store upsert.  Returns an empty list if the
        document has no content or all chunks fail to embed.
        """
        chunks = self._chunker.chunk(document)
        if not chunks:
            logger.debug("No chunks produced for %s", document.source)
            return []

        texts = [c.text for c in chunks]
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both(texts)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", document.source, exc)
            return []

        return [
            chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
            for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs, strict=True)
        ]
