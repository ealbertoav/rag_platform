"""T-123 — Relevant Segment Extraction tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.constants import CHUNK_INDEX_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.retrieval_service import RetrievalService
from src.rag.compression.token_reducer import count_tokens
from src.rag.enrichment.relevant_segment_extraction import (
    RSE_MERGED_KEY,
    RSE_SOURCE_CHUNK_IDS_KEY,
    merge_adjacent,
)


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "doc1",
    text: str = "sample text",
    chunk_index: int | None = None,
) -> Chunk:
    metadata: dict[str, object] = {}
    if chunk_index is not None:
        metadata[CHUNK_INDEX_KEY] = chunk_index
    return Chunk(id=chunk_id, document_id=document_id, text=text, metadata=metadata)


class TestMergeAdjacent:
    def test_empty_list(self):
        assert merge_adjacent([], max_segment_tokens=100) == []

    def test_single_chunk_unchanged(self):
        chunk = _chunk("c0", text="only chunk", chunk_index=0)
        result = merge_adjacent([chunk], max_segment_tokens=100)
        assert result == [chunk]

    def test_merges_consecutive_chunks_same_document(self):
        chunks = [
            _chunk("c0", text="part one", chunk_index=0),
            _chunk("c1", text="part two", chunk_index=1),
        ]
        result = merge_adjacent(chunks, max_segment_tokens=100)
        assert len(result) == 1
        assert "part one" in result[0].text
        assert "part two" in result[0].text
        assert result[0].metadata[RSE_MERGED_KEY] is True
        assert result[0].metadata[RSE_SOURCE_CHUNK_IDS_KEY] == ["c0", "c1"]

    def test_does_not_merge_non_consecutive_indices(self):
        chunks = [
            _chunk("c0", text="first", chunk_index=0),
            _chunk("c2", text="third", chunk_index=2),
        ]
        result = merge_adjacent(chunks, max_segment_tokens=100)
        assert len(result) == 2
        assert result[0].id == "c0"
        assert result[1].id == "c2"

    def test_does_not_merge_different_documents(self):
        chunks = [
            _chunk("c0", document_id="doc-a", text="a0", chunk_index=0),
            _chunk("c1", document_id="doc-b", text="b0", chunk_index=0),
        ]
        result = merge_adjacent(chunks, max_segment_tokens=100)
        assert len(result) == 2

    def test_chunks_without_index_remain_standalone(self):
        indexed = _chunk("c0", text="indexed", chunk_index=0)
        plain = _chunk("c1", text="plain")
        result = merge_adjacent([plain, indexed], max_segment_tokens=100)
        assert len(result) == 2
        assert result[0].id == "c1"
        assert result[1].id == "c0"

    def test_respects_max_segment_tokens(self):
        long_text = "word " * 200
        chunks = [
            _chunk("c0", text=long_text, chunk_index=0),
            _chunk("c1", text=long_text, chunk_index=1),
        ]
        max_tokens = count_tokens(long_text) + 10
        result = merge_adjacent(chunks, max_segment_tokens=max_tokens)
        assert len(result) == 2

    def test_merged_segment_never_exceeds_max_tokens(self):
        chunks = [
            _chunk("c0", text="a" * 80, chunk_index=0),
            _chunk("c1", text="b" * 80, chunk_index=1),
            _chunk("c2", text="c" * 80, chunk_index=2),
        ]
        result = merge_adjacent(chunks, max_segment_tokens=30)
        for segment in result:
            assert count_tokens(segment.text) <= 30

    def test_preserves_rerank_order(self):
        chunks = [
            _chunk("c2", document_id="doc-b", text="b2", chunk_index=2),
            _chunk("c0", document_id="doc-a", text="a0", chunk_index=0),
            _chunk("c1", document_id="doc-a", text="a1", chunk_index=1),
        ]
        result = merge_adjacent(chunks, max_segment_tokens=100)
        assert [chunk.id for chunk in result] == ["c2", "c0"]

    def test_three_chunk_run_merges_into_one(self):
        chunks = [
            _chunk("c0", text="one", chunk_index=0),
            _chunk("c1", text="two", chunk_index=1),
            _chunk("c2", text="three", chunk_index=2),
        ]
        result = merge_adjacent(chunks, max_segment_tokens=100)
        assert len(result) == 1
        assert result[0].metadata[RSE_SOURCE_CHUNK_IDS_KEY] == ["c0", "c1", "c2"]


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
            rse_max_segment_tokens=100,
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
