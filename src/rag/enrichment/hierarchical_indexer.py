from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from src.core.constants import (
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_DETAIL,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_SUMMARY,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.metadata import chunk_metadata

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "ingestion" / "generate_document_summary.txt"


def load_summary_template(path: Path | None = None) -> Template:
    """Load the document-summary generation prompt from the disk."""
    template_path = path or _PROMPT_PATH
    return Template(template_path.read_text(encoding="utf-8").strip())


def is_summary_chunk(chunk: Chunk) -> bool:
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_SUMMARY


def is_detail_chunk(chunk: Chunk) -> bool:
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_DETAIL


def tag_detail_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Mark *chunks* as hierarchical detail index points."""
    tagged: list[Chunk] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata[CHUNK_TYPE_KEY] = CHUNK_TYPE_DETAIL
        tagged.append(chunk.model_copy(update={"metadata": metadata}))
    return tagged


def make_summary_chunk(document: Document, summary: str) -> Chunk:
    """Build a document-level summary index point."""
    metadata = chunk_metadata(document.metadata)
    metadata[CHUNK_TYPE_KEY] = CHUNK_TYPE_SUMMARY
    metadata[CHUNK_SOURCE_KEY] = document.source
    return Chunk(document_id=document.id, text=summary, metadata=metadata)


def generate_document_summary(
    document: Document,
    llm: LLMRepository,
    template: Template | None = None,
) -> str:
    """Return a concise summary of a *document* using the LLM."""
    tmpl = template or load_summary_template()
    prompt = tmpl.substitute(source=document.source, document=document.content.strip())
    return llm.generate(prompt=prompt, context="").strip()


class HierarchicalIndexer:
    """Generates and embeds document-level summary nodes for two-tier retrieval."""

    def __init__(
        self,
        llm: LLMRepository,
        embedder: EmbeddingRepository,
        template: Template | None = None,
    ) -> None:
        self._llm: LLMRepository = llm
        self._embedder: EmbeddingRepository = embedder
        self._template: Template | None = template

    def index(
        self,
        document: Document,
        detail_chunks: list[Chunk],
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Return (tagged detail chunks, embedded summary chunks).

        Detail chunks are always tagged with "type=detail".  When summary
        generation or embedding fails, the second list is empty.
        """
        tagged = tag_detail_chunks(detail_chunks)
        try:
            summary_text = generate_document_summary(document, self._llm, self._template)
        except Exception as exc:
            logger.warning("Document summary generation failed for %s: %s", document.source, exc)
            return tagged, []

        if not summary_text:
            logger.warning("Empty document summary for %s", document.source)
            return tagged, []

        summary = make_summary_chunk(document, summary_text)
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both([summary.text])
        except Exception as exc:
            logger.warning("Embedding document summary failed for %s: %s", document.source, exc)
            return tagged, []

        embedded = summary.model_copy(
            update={"embedding": dense_vecs[0], "sparse_vector": sparse_vecs[0]}
        )
        return tagged, [embedded]
