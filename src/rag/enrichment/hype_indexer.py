from __future__ import annotations

import logging
from string import Template

from src.core.constants import (
    CHUNK_RAW_TEXT_KEY,
    CHUNK_TYPE_HYPE,
    CHUNK_TYPE_KEY,
    SOURCE_CHUNK_ID_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.enrichment.document_augmentation import generate_questions

logger = logging.getLogger(__name__)


def is_hype_question(chunk: Chunk) -> bool:
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_HYPE


def make_hype_chunk(source: Chunk, question: str) -> Chunk:
    """Build a HyPE index point linking a hypothetical question to a *source*."""
    metadata = {
        k: v for k, v in source.metadata.items() if k not in {CHUNK_RAW_TEXT_KEY, CHUNK_TYPE_KEY}
    }
    metadata[CHUNK_TYPE_KEY] = CHUNK_TYPE_HYPE
    metadata[SOURCE_CHUNK_ID_KEY] = source.id
    return Chunk(document_id=source.document_id, text=question, metadata=metadata)


class HyPEIndexer:
    """Precomputes hypothetical questions per chunk for question-question retrieval."""

    def __init__(
        self,
        llm: LLMRepository,
        embedder: EmbeddingRepository,
        n_questions: int = 3,
        template: Template | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._n_questions = n_questions
        self._template = template

    def index(self, source_chunks: list[Chunk]) -> list[Chunk]:
        """Return embedded HyPE question chunks for *source_chunks*."""
        hype_chunks: list[Chunk] = []
        for chunk in source_chunks:
            try:
                questions = generate_questions(chunk, self._llm, self._n_questions, self._template)
            except Exception as exc:
                logger.warning("HyPE question generation failed for chunk %s: %s", chunk.id, exc)
                continue

            for question in questions[: self._n_questions]:
                if question.strip():
                    hype_chunks.append(make_hype_chunk(chunk, question.strip()))

        if not hype_chunks:
            return []

        texts = [c.text for c in hype_chunks]
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both(texts)
        except Exception as exc:
            logger.warning("Embedding HyPE questions failed: %s", exc)
            return []

        return [
            chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
            for chunk, dense, sparse in zip(hype_chunks, dense_vecs, sparse_vecs, strict=True)
        ]
