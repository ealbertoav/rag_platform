"""Relevant Segment Extraction (RSE) — merge adjacent retrieved chunks (T-123)."""

from __future__ import annotations

from src.core.constants import CHUNK_INDEX_KEY
from src.domain.entities.chunk import Chunk
from src.rag.compression.token_reducer import count_tokens, truncate_to_tokens

RSE_MERGED_KEY = "rse_merged"
RSE_SOURCE_CHUNK_IDS_KEY = "rse_source_chunk_ids"

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


def _can_extend_run(
    run: list[IndexedChunk],
    next_chunk: Chunk,
    max_segment_tokens: int,
) -> bool:
    if not run:
        return True
    combined_text = _join_texts([entry[1].text for entry in run] + [next_chunk.text])
    return count_tokens(combined_text) <= max_segment_tokens


def _merge_run(run: list[IndexedChunk], max_segment_tokens: int) -> Chunk:
    if len(run) == 1:
        return run[0][1]

    source_chunks = [entry[1] for entry in run]
    combined_text = _join_texts([chunk.text for chunk in source_chunks])
    if count_tokens(combined_text) > max_segment_tokens:
        combined_text = truncate_to_tokens(combined_text, max_segment_tokens)

    first = source_chunks[0]
    metadata = {
        **first.metadata,
        RSE_MERGED_KEY: True,
        RSE_SOURCE_CHUNK_IDS_KEY: [chunk.id for chunk in source_chunks],
    }
    return first.model_copy(update={"text": combined_text, "metadata": metadata})


def _join_texts(texts: list[str]) -> str:
    return "\n\n".join(text.strip() for text in texts if text.strip())
