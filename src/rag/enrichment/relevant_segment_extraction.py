from __future__ import annotations

from collections import defaultdict

from src.core.constants import (
    CHUNK_INDEX_KEY,
    CHUNK_PARENT_ID_KEY,
    MERGED_CHUNK_IDS_KEY,
    RSE_MERGED_KEY,
)
from src.domain.entities.chunk import Chunk
from src.rag.compression.token_reducer import count_tokens


def _join_texts(chunks: list[Chunk]) -> str:
    return "\n\n".join(c.text for c in chunks)


def _merge_run(run: list[Chunk]) -> Chunk:
    """Combine a consecutive run of chunks into a single segment."""
    anchor = run[0]
    merged_text = _join_texts(run)
    return anchor.model_copy(
        update={
            "text": merged_text,
            "metadata": {
                **anchor.metadata,
                MERGED_CHUNK_IDS_KEY: [c.id for c in run],
                RSE_MERGED_KEY: True,
            },
        }
    )


def _merge_group(chunk: Chunk) -> tuple[str, str | None]:
    """Group key for merge eligibility within a document.

    Parent-level chunks (no ``parent_id``) merge only with other parents.
    Child chunks merge only with siblings that share the same ``parent_id``.
    """
    parent_id = chunk.metadata.get(CHUNK_PARENT_ID_KEY)
    return chunk.document_id, parent_id if parent_id is not None else None


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
        if CHUNK_INDEX_KEY in chunk.metadata:
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
            can_extend = (
                curr_idx == prev_idx + 1
                and count_tokens(_join_texts([*run, chunk])) <= max_segment_tokens
            )

            if can_extend:
                run.append(chunk)
            else:
                merged.append(_merge_run(run) if len(run) > 1 else run[0])
                run = [chunk]

        if run:
            merged.append(_merge_run(run) if len(run) > 1 else run[0])

    merged.extend(passthrough)

    def _sort_key(chunk: Chunk) -> int:
        merged_ids = chunk.metadata.get(MERGED_CHUNK_IDS_KEY)
        if isinstance(merged_ids, list) and merged_ids:
            return min(original_rank.get(cid, len(chunks)) for cid in merged_ids)
        return original_rank.get(chunk.id, len(chunks))

    merged.sort(key=_sort_key)
    merge_count = len(chunks) - len(merged)
    return merged, merge_count
