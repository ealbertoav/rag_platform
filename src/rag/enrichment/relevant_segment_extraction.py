"""Relevant Segment Extraction (RSE) — merge adjacent retrieved chunks (T-123)."""

from __future__ import annotations

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_RAW_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.rag.chunking.contextual_headers import chunk_context_text
from src.rag.compression.token_reducer import count_tokens, truncate_to_tokens

RSE_MERGED_KEY = "rse_merged"
RSE_SOURCE_CHUNK_IDS_KEY = "rse_source_chunk_ids"

# Upper bound when searching for shared suffix/prefix between overlapping siblings.
_MAX_OVERLAP_SEARCH = 512

RankedChunk = tuple[int, Chunk]
IndexedChunk = tuple[int, Chunk, int]


def merge_adjacent(chunks: list[Chunk], max_segment_tokens: int) -> list[Chunk]:
    """Merge consecutive chunks from the same document into longer segments.

    Chunks are grouped by "document_id" and merged when their
    "metadata["chunk_index"]" values form a consecutive run among the
    retrieved set.  Merged segments never exceed *max_segment_tokens*.
    """
    if not chunks:
        return []

    indexed: list[IndexedChunk] = []
    standalone: list[RankedChunk] = []

    for order, chunk in enumerate(chunks):
        raw_index = chunk.metadata.get(CHUNK_INDEX_KEY)
        if raw_index is None:
            standalone.append((order, chunk))
            continue
        try:
            chunk_index = int(raw_index)
        except (TypeError, ValueError):
            standalone.append((order, chunk))
            continue
        indexed.append((order, chunk, chunk_index))

    merged: list[RankedChunk] = []
    by_document: dict[str, list[IndexedChunk]] = {}
    for order, chunk, chunk_index in indexed:
        by_document.setdefault(chunk.document_id, []).append((order, chunk, chunk_index))

    for doc_chunks in by_document.values():
        doc_chunks.sort(key=lambda entry: entry[2])
        run: list[IndexedChunk] = []
        prev_index: int | None = None

        for order, chunk, chunk_index in doc_chunks:
            gap_break = prev_index is not None and chunk_index != prev_index + 1
            token_break = bool(run) and not _can_extend_run(run, chunk, max_segment_tokens)
            if run and (gap_break or token_break):
                segment = _merge_run(run, max_segment_tokens)
                merged.append((min(entry[0] for entry in run), segment))
                run = []
            run.append((order, chunk, chunk_index))
            prev_index = chunk_index

        if run:
            segment = _merge_run(run, max_segment_tokens)
            merged.append((min(entry[0] for entry in run), segment))

    output = standalone + merged
    output.sort(key=lambda ranked: ranked[0])
    return [chunk for _, chunk in output]


def chunk_source_ids(chunk: Chunk) -> list[str]:
    """Return citation chunk IDs, expanding RSE merges to all source chunks."""
    ids = chunk.metadata.get(RSE_SOURCE_CHUNK_IDS_KEY)
    if isinstance(ids, list) and ids:
        return [str(chunk_id) for chunk_id in ids]
    return [chunk.id]


def _can_extend_run(
    run: list[IndexedChunk],
    next_chunk: Chunk,
    max_segment_tokens: int,
) -> bool:
    if not run:
        return True
    bodies = [_context_body(entry[1]) for entry in run] + [_context_body(next_chunk)]
    combined_text = _join_overlapping(bodies)
    return count_tokens(combined_text) <= max_segment_tokens


def _merge_run(run: list[IndexedChunk], max_segment_tokens: int) -> Chunk:
    if len(run) == 1:
        return run[0][1]

    source_chunks = [entry[1] for entry in run]
    context_bodies = [_context_body(chunk) for chunk in source_chunks]
    merged_context = _join_overlapping(context_bodies)
    merged_embedded = _join_overlapping([chunk.text for chunk in source_chunks])

    if count_tokens(merged_context) > max_segment_tokens:
        merged_context = truncate_to_tokens(merged_context, max_segment_tokens)
        merged_embedded = merged_context

    first = source_chunks[0]
    metadata = {
        **first.metadata,
        RSE_MERGED_KEY: True,
        RSE_SOURCE_CHUNK_IDS_KEY: [chunk.id for chunk in source_chunks],
        CHUNK_RAW_TEXT_KEY: merged_context,
    }
    return first.model_copy(update={"text": merged_embedded, "metadata": metadata})


def _context_body(chunk: Chunk) -> str:
    """Return the passage text used for LLM context (respects CCH raw_text)."""
    return chunk_context_text(chunk)


def _join_overlapping(texts: list[str]) -> str:
    """Join texts, deduplicating shared suffix/prefix from chunk overlap."""
    parts = [text.strip() for text in texts if text.strip()]
    if not parts:
        return ""

    result = parts[0]
    for part in parts[1:]:
        overlap = _find_overlap(result, part)
        result = result + part[overlap:] if overlap else f"{result}\n\n{part}"
    return result


def _find_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right), _MAX_OVERLAP_SEARCH)
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0
