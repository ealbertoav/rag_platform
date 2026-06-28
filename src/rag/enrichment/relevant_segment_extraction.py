"""Relevant Segment Extraction (RSE) — merge adjacent retrieved chunks (T-123)."""

from __future__ import annotations

from collections import defaultdict

from src.core.constants import (
    CHUNK_INDEX_KEY,
    CHUNK_PARENT_ID_KEY,
    CHUNK_RAW_TEXT_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_PROPOSITION,
    MERGED_CHUNK_IDS_KEY,
    RSE_MERGED_KEY,
)
from src.domain.entities.chunk import Chunk
from src.rag.chunking.contextual_headers import chunk_context_text
from src.rag.compression.token_reducer import count_tokens, truncate_to_tokens

_MAX_OVERLAP_SEARCH = 512


def merge_adjacent(chunks: list[Chunk], max_segment_tokens: int) -> tuple[list[Chunk], int]:
    """Merge adjacent retrieved chunks from the same document into longer segments.

    Chunks are eligible when they share a document, belong to the same merge
    group (parent-level or same ``parent_id`` for children), and have
    consecutive ``metadata["chunk_index"]`` values.  Merged segments never
    exceed *max_segment_tokens* (approximate token count).

    Returns ``(merged_chunks, merge_count)`` where *merge_count* is the number
    of chunk boundaries eliminated (input count minus output count).
    """
    if not chunks:
        return [], 0

    original_rank = {chunk.id: index for index, chunk in enumerate(chunks)}
    mergeable: list[Chunk] = []
    passthrough: list[Chunk] = []

    for chunk in chunks:
        if _is_rse_mergeable(chunk):
            mergeable.append(chunk)
        else:
            passthrough.append(chunk)

    by_group: dict[tuple[str, str | None], list[Chunk]] = defaultdict(list)
    for chunk in mergeable:
        by_group[_merge_group(chunk)].append(chunk)

    merged: list[Chunk] = []

    for doc_chunks in by_group.values():
        doc_chunks.sort(key=lambda c: int(c.metadata[CHUNK_INDEX_KEY]))
        run: list[Chunk] = []

        for chunk in doc_chunks:
            if not run:
                run = [chunk]
                continue

            prev_idx = int(run[-1].metadata[CHUNK_INDEX_KEY])
            curr_idx = int(chunk.metadata[CHUNK_INDEX_KEY])
            can_extend = curr_idx == prev_idx + 1 and _can_extend_run(
                run, chunk, max_segment_tokens
            )

            if can_extend:
                run.append(chunk)
            else:
                merged.append(_merge_run(run, max_segment_tokens) if len(run) > 1 else run[0])
                run = [chunk]

        if run:
            merged.append(_merge_run(run, max_segment_tokens) if len(run) > 1 else run[0])

    merged.extend(passthrough)

    def _sort_key(chunk: Chunk) -> int:
        merged_ids = chunk.metadata.get(MERGED_CHUNK_IDS_KEY)
        if isinstance(merged_ids, list) and merged_ids:
            return min(original_rank.get(cid, len(chunks)) for cid in merged_ids)
        return original_rank.get(chunk.id, len(chunks))

    merged.sort(key=_sort_key)
    merge_count = len(chunks) - len(merged)
    return merged, merge_count


def chunk_source_ids(chunk: Chunk) -> list[str]:
    """Return citation chunk IDs, expanding RSE merges to all source chunks."""
    ids = chunk.metadata.get(MERGED_CHUNK_IDS_KEY)
    if isinstance(ids, list) and ids:
        return [str(chunk_id) for chunk_id in ids]
    return [chunk.id]


def _is_rse_mergeable(chunk: Chunk) -> bool:
    """True when a chunk represents an adjacent passage slice eligible for RSE."""
    if chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_PROPOSITION:
        return False
    return CHUNK_INDEX_KEY in chunk.metadata


def _merge_group(chunk: Chunk) -> tuple[str, str | None]:
    """Group key for merge eligibility within a document.

    Parent-level chunks (no ``parent_id``) merge only with other parents.
    Child chunks merge only with siblings that share the same ``parent_id``.
    """
    parent_id = chunk.metadata.get(CHUNK_PARENT_ID_KEY)
    return chunk.document_id, parent_id if parent_id is not None else None


def _can_extend_run(run: list[Chunk], next_chunk: Chunk, max_segment_tokens: int) -> bool:
    bodies = [_context_body(chunk) for chunk in run] + [_context_body(next_chunk)]
    return count_tokens(_join_overlapping(bodies)) <= max_segment_tokens


def _merge_run(run: list[Chunk], max_segment_tokens: int) -> Chunk:
    """Combine a consecutive run of chunks into a single segment."""
    anchor = run[0]
    context_bodies = [_context_body(chunk) for chunk in run]
    merged_context = _join_overlapping(context_bodies)
    merged_embedded = _join_overlapping([chunk.text for chunk in run])

    if count_tokens(merged_context) > max_segment_tokens:
        merged_context = truncate_to_tokens(merged_context, max_segment_tokens)
        merged_embedded = merged_context

    metadata = {
        **anchor.metadata,
        MERGED_CHUNK_IDS_KEY: [chunk.id for chunk in run],
        RSE_MERGED_KEY: True,
        CHUNK_RAW_TEXT_KEY: merged_context,
    }
    return anchor.model_copy(update={"text": merged_embedded, "metadata": metadata})


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
