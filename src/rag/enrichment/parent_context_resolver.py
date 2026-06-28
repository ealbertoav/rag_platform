"""Parent context resolution for parent-child chunking (T-124)."""

from __future__ import annotations

import logging
from typing import Protocol

from opentelemetry import trace

from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.rag.chunking.contextual_headers import chunk_context_text

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")


class ChunkLookup(Protocol):
    """Minimal interface for resolving parent chunks by ID."""

    def get_by_id(self, chunk_id: str) -> Chunk | None: ...


def enrich_with_parent_context(
    chunks: list[Chunk],
    lookup: ChunkLookup,
) -> tuple[list[Chunk], int]:
    """Replace child chunk LLM context with parent text when available.

    Retrieved child chunk IDs are preserved, so citations still point at the
    precise match.  When a parent cannot be found, the child text is used as-is.

    Returns "(enriched_chunks, resolved_count)" where *resolved_count* is the
    number of chunks whose context was expanded to a parent.
    """
    if not chunks:
        return [], 0

    enriched: list[Chunk] = []
    resolved_count = 0
    parent_bodies: dict[str, str] = {}

    for chunk in chunks:
        parent_id = chunk.metadata.get(CHUNK_PARENT_ID_KEY)
        if not isinstance(parent_id, str) or not parent_id:
            enriched.append(chunk)
            continue

        parent_body = parent_bodies.get(parent_id)
        if parent_body is None:
            parent = lookup.get_by_id(parent_id)
            if parent is None:
                logger.debug(
                    "Parent chunk %s not found for child %s; using child text",
                    parent_id,
                    chunk.id,
                )
                enriched.append(chunk)
                continue
            parent_body = chunk_context_text(parent)
            if not parent_body:
                logger.debug(
                    "Parent chunk %s resolved to empty text for child %s; using child text",
                    parent_id,
                    chunk.id,
                )
                enriched.append(chunk)
                continue
            parent_bodies[parent_id] = parent_body

        enriched.append(
            chunk.model_copy(
                update={
                    "metadata": {
                        **chunk.metadata,
                        PARENT_CONTEXT_TEXT_KEY: parent_body,
                    }
                }
            )
        )
        resolved_count += 1

    return enriched, resolved_count


def drop_redundant_parent_hits(chunks: list[Chunk]) -> list[Chunk]:
    """Drop parent-level hits when enriched children already cover the same parent."""
    enriched_parent_ids: set[str] = {
        parent_id
        for chunk in chunks
        if isinstance((parent_id := chunk.metadata.get(CHUNK_PARENT_ID_KEY)), str)
        and parent_id
        and isinstance((ctx := chunk.metadata.get(PARENT_CONTEXT_TEXT_KEY)), str)
        and ctx
    }
    if not enriched_parent_ids:
        return chunks
    return [chunk for chunk in chunks if chunk.id not in enriched_parent_ids]
