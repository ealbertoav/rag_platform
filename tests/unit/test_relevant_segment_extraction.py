"""T-123 — relevant segment extraction (RSE) tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.constants import (
    CHUNK_INDEX_KEY,
    CHUNK_PARENT_ID_KEY,
    CHUNK_RAW_TEXT_KEY,
    MERGED_CHUNK_IDS_KEY,
    RSE_MERGED_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.retrieval_service import RetrievalService
from src.rag.chunking.contextual_headers import chunk_context_text
from src.rag.compression.token_reducer import count_tokens
from src.rag.enrichment.relevant_segment_extraction import chunk_source_ids, merge_adjacent


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    text: str = "segment text",
    chunk_index: int | None = 0,
    parent_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    chunk_metadata: dict[str, object] = dict(metadata or {})
    if chunk_index is not None:
        chunk_metadata[CHUNK_INDEX_KEY] = chunk_index
    if parent_id is not None:
        chunk_metadata[CHUNK_PARENT_ID_KEY] = parent_id
    return Chunk(id=chunk_id, document_id=document_id, text=text, metadata=chunk_metadata)


class TestMergeAdjacentBasics:
    def test_empty_list(self):
        merged, merge_count = merge_adjacent([], max_segment_tokens=100)
        assert merged == []
        assert merge_count == 0

    def test_single_chunk_unchanged(self):
        chunk = _chunk("c0", text="only chunk")
        merged, merge_count = merge_adjacent([chunk], max_segment_tokens=100)
        assert merged == [chunk]
        assert merge_count == 0

    def test_non_adjacent_indices_not_merged(self):
        chunks = [
            _chunk("c0", text="part zero", chunk_index=0),
            _chunk("c2", text="part two", chunk_index=2),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 2
        assert merge_count == 0

    def test_different_documents_not_merged(self):
        chunks = [
            _chunk("c0", document_id="doc-a", text="alpha", chunk_index=0),
            _chunk("c1", document_id="doc-b", text="beta", chunk_index=1),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 2
        assert merge_count == 0

    def test_chunks_without_index_pass_through(self):
        chunk = Chunk(id="c0", document_id="doc-1", text="no index", metadata={})
        merged, merge_count = merge_adjacent([chunk], max_segment_tokens=500)
        assert merged == [chunk]
        assert merge_count == 0


class TestMergeAdjacentMerging:
    def test_adjacent_chunks_merge(self):
        chunks = [
            _chunk("c1", text="first part.", chunk_index=1),
            _chunk("c2", text="second part.", chunk_index=2),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 1
        assert merge_count == 1
        assert "first part." in merged[0].text
        assert "second part." in merged[0].text
        assert merged[0].metadata[MERGED_CHUNK_IDS_KEY] == ["c1", "c2"]
        assert merged[0].metadata[RSE_MERGED_KEY] is True
        assert merged[0].id == "c1"

    def test_three_consecutive_chunks_merge(self):
        chunks = [
            _chunk("c0", text="a", chunk_index=0),
            _chunk("c1", text="b", chunk_index=1),
            _chunk("c2", text="c", chunk_index=2),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 1
        assert merge_count == 2
        assert merged[0].text == "a\n\nb\n\nc"

    def test_respects_max_segment_tokens(self):
        long_a = "a" * 400
        long_b = "b" * 400
        chunks = [
            _chunk("c0", text=long_a, chunk_index=0),
            _chunk("c1", text=long_b, chunk_index=1),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=150)
        assert len(merged) == 2
        assert merge_count == 0

    def test_partial_run_when_third_chunk_exceeds_limit(self):
        medium = "y" * 200
        huge = "z" * 800
        chunks = [
            _chunk("c0", text=medium, chunk_index=0),
            _chunk("c1", text=medium, chunk_index=1),
            _chunk("c2", text=huge, chunk_index=2),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=120)
        assert len(merged) == 2
        assert merge_count == 1
        assert merged[0].metadata.get(MERGED_CHUNK_IDS_KEY) == ["c0", "c1"]
        assert merged[1].id == "c2"

    def test_merged_segment_never_exceeds_max_tokens(self):
        chunks = [
            _chunk("c0", text="word " * 20, chunk_index=0),
            _chunk("c1", text="word " * 20, chunk_index=1),
            _chunk("c2", text="word " * 20, chunk_index=2),
        ]
        max_tokens = 30
        merged, _ = merge_adjacent(chunks, max_segment_tokens=max_tokens)
        for chunk in merged:
            assert count_tokens(chunk.text) <= max_tokens

    def test_preserves_rerank_order_by_first_seen(self):
        chunks = [
            _chunk("c5", text="later in doc", chunk_index=5),
            _chunk("c0", text="doc two", document_id="doc-2", chunk_index=0),
            _chunk("c4", text="earlier in doc", chunk_index=4),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert merge_count == 1
        assert len(merged) == 2
        assert merged[0].metadata[MERGED_CHUNK_IDS_KEY] == ["c4", "c5"]
        assert merged[1].id == "c0"

    def test_deduplicates_overlap_between_siblings(self):
        chunks = [
            _chunk("c0", text="The quick brown fox jumps over the lazy", chunk_index=0),
            _chunk("c1", text="jumps over the lazy dog near the river", chunk_index=1),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert merge_count == 1
        assert merged[0].text.count("jumps over the lazy") == 1
        assert "dog near the river" in merged[0].text

    def test_merged_raw_text_includes_all_bodies_with_cch(self):
        chunks = [
            _chunk(
                "c0",
                text="[Document: Report | Section: — | Page: —]\npart one",
                chunk_index=0,
                metadata={CHUNK_RAW_TEXT_KEY: "part one"},
            ),
            _chunk(
                "c1",
                text="[Document: Report | Section: — | Page: —]\npart two",
                chunk_index=1,
                metadata={CHUNK_RAW_TEXT_KEY: "part two"},
            ),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert merge_count == 1
        assert chunk_context_text(merged[0]) == "part one\n\npart two"
        assert merged[0].metadata[CHUNK_RAW_TEXT_KEY] == "part one\n\npart two"


class TestMergeAdjacentParentChild:
    def test_parent_and_child_with_consecutive_index_not_merged(self):
        chunks = [
            _chunk("parent-0", text="full parent passage.", chunk_index=0),
            _chunk("child-1", text="child slice.", chunk_index=1, parent_id="parent-0"),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 2
        assert merge_count == 0

    def test_sibling_children_with_consecutive_index_merge(self):
        chunks = [
            _chunk("child-0", text="first child.", chunk_index=0, parent_id="parent-0"),
            _chunk("child-1", text="second child.", chunk_index=1, parent_id="parent-0"),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 1
        assert merge_count == 1
        assert merged[0].metadata[MERGED_CHUNK_IDS_KEY] == ["child-0", "child-1"]

    def test_children_of_different_parents_not_merged(self):
        chunks = [
            _chunk("child-a", text="last of parent A.", chunk_index=2, parent_id="parent-a"),
            _chunk("child-b", text="first of parent B.", chunk_index=3, parent_id="parent-b"),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 2
        assert merge_count == 0

    def test_parent_level_chunks_still_merge(self):
        chunks = [
            _chunk("parent-0", text="section one.", chunk_index=0),
            _chunk("parent-1", text="section two.", chunk_index=1),
        ]
        merged, merge_count = merge_adjacent(chunks, max_segment_tokens=500)
        assert len(merged) == 1
        assert merge_count == 1
        assert merged[0].metadata[MERGED_CHUNK_IDS_KEY] == ["parent-0", "parent-1"]


class TestChunkSourceIds:
    def test_single_chunk_returns_own_id(self):
        chunk = _chunk("c0")
        assert chunk_source_ids(chunk) == ["c0"]

    def test_merged_chunk_returns_all_source_ids(self):
        chunk = _chunk(
            "c0",
            chunk_index=0,
            metadata={
                RSE_MERGED_KEY: True,
                MERGED_CHUNK_IDS_KEY: ["c0", "c1", "c2"],
            },
        )
        assert chunk_source_ids(chunk) == ["c0", "c1", "c2"]


class TestRetrievalServiceRSE:
    @pytest.mark.asyncio
    async def test_rse_disabled_leaves_chunks_unmerged(self):
        chunks = [
            _chunk("c0", text="part one", chunk_index=0),
            _chunk("c1", text="part two", chunk_index=1),
        ]
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            rse_enabled=False,
        )
        svc._dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        svc._hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])

        result = await svc.retrieve(Query(text="test"))
        assert len(result.chunks) == 2

    @pytest.mark.asyncio
    async def test_rse_enabled_merges_adjacent_chunks(self):
        chunks = [
            _chunk("c0", text="part one", chunk_index=0),
            _chunk("c1", text="part two", chunk_index=1),
        ]
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            rse_enabled=True,
            rse_max_segment_tokens=500,
        )
        svc._dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        svc._hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])

        result = await svc.retrieve(Query(text="test"))
        assert len(result.chunks) == 1
        assert result.chunks[0].metadata.get(RSE_MERGED_KEY) is True

    @pytest.mark.asyncio
    async def test_rse_runs_before_compression(self):
        chunks = [_chunk("c0", text="part one", chunk_index=0)]
        compressor = MagicMock()
        compressor.compress.side_effect = lambda _q, cs: cs

        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            compressor=compressor,
            rse_enabled=True,
        )
        svc._dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        svc._hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])

        await svc.retrieve(Query(text="test"))
        compressor.compress.assert_called_once()
