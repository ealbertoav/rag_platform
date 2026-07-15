"""T-243 — Modality chunk type registry & index routing."""

from __future__ import annotations

import pytest

from src.core.constants import (
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_DETAIL,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_HYPE,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_PAGE,
    CHUNK_TYPE_PROPOSITION,
    CHUNK_TYPE_SUMMARY,
    CHUNK_TYPE_SYNTHETIC,
    CHUNK_TYPE_TABLE,
)
from src.domain.entities.chunk import Chunk
from src.rag.ingestion.chunk_type_registry import (
    DEFAULT_ROUTING,
    ChunkIndexRouting,
    filter_bm25_indexable,
    filter_dense_indexable,
    is_bm25_indexable,
    is_dense_indexable,
    routing_for_chunk,
    routing_for_type,
)


def _chunk(chunk_type: str | None) -> Chunk:
    metadata = {} if chunk_type is None else {CHUNK_TYPE_KEY: chunk_type}
    return Chunk(document_id="doc-1", text="hello", metadata=metadata)


class TestRoutingForType:
    def test_none_defaults_to_both_stores(self):
        assert routing_for_type(None) == DEFAULT_ROUTING

    def test_unknown_type_defaults_to_both_stores(self):
        assert routing_for_type("some_future_type") == DEFAULT_ROUTING

    @pytest.mark.parametrize("chunk_type", [CHUNK_TYPE_HYPE, CHUNK_TYPE_SUMMARY])
    def test_vector_only_types_excluded_from_bm25(self, chunk_type: str):
        routing = routing_for_type(chunk_type)
        assert routing.index_dense is True
        assert routing.index_bm25 is False

    @pytest.mark.parametrize(
        "chunk_type",
        [
            CHUNK_TYPE_SYNTHETIC,
            CHUNK_TYPE_TABLE,
            CHUNK_TYPE_CAPTION,
            CHUNK_TYPE_DETAIL,
            CHUNK_TYPE_PROPOSITION,
            CHUNK_TYPE_PAGE,
            CHUNK_TYPE_FIGURE,
        ],
    )
    def test_dual_indexed_types(self, chunk_type: str):
        assert routing_for_type(chunk_type) == DEFAULT_ROUTING


class TestRoutingForChunk:
    def test_reads_type_from_metadata(self):
        chunk = _chunk(CHUNK_TYPE_HYPE)
        routing = routing_for_chunk(chunk)
        assert routing == ChunkIndexRouting(index_dense=True, index_bm25=False)

    def test_missing_type_metadata_defaults_to_both(self):
        chunk = _chunk(None)
        assert routing_for_chunk(chunk) == DEFAULT_ROUTING


class TestPredicates:
    def test_is_dense_indexable_true_for_all_known_types(self):
        for chunk_type in (
            None,
            CHUNK_TYPE_HYPE,
            CHUNK_TYPE_SUMMARY,
            CHUNK_TYPE_TABLE,
            CHUNK_TYPE_CAPTION,
        ):
            assert is_dense_indexable(_chunk(chunk_type)) is True

    def test_is_bm25_indexable_false_only_for_vector_only_types(self):
        assert is_bm25_indexable(_chunk(CHUNK_TYPE_HYPE)) is False
        assert is_bm25_indexable(_chunk(CHUNK_TYPE_SUMMARY)) is False
        assert is_bm25_indexable(_chunk(CHUNK_TYPE_TABLE)) is True
        assert is_bm25_indexable(_chunk(CHUNK_TYPE_CAPTION)) is True
        assert is_bm25_indexable(_chunk(None)) is True


class TestFilters:
    def test_filter_bm25_indexable_excludes_hype_and_summary(self):
        chunks = [
            _chunk(None),
            _chunk(CHUNK_TYPE_HYPE),
            _chunk(CHUNK_TYPE_SUMMARY),
            _chunk(CHUNK_TYPE_TABLE),
            _chunk(CHUNK_TYPE_SYNTHETIC),
        ]
        result = filter_bm25_indexable(chunks)
        assert [c.metadata.get(CHUNK_TYPE_KEY) for c in result] == [
            None,
            CHUNK_TYPE_TABLE,
            CHUNK_TYPE_SYNTHETIC,
        ]

    def test_filter_dense_indexable_keeps_everything(self):
        chunks = [
            _chunk(None),
            _chunk(CHUNK_TYPE_HYPE),
            _chunk(CHUNK_TYPE_SUMMARY),
            _chunk(CHUNK_TYPE_TABLE),
        ]
        assert filter_dense_indexable(chunks) == chunks

    def test_filters_preserve_order_and_are_pure(self):
        chunks = [_chunk(CHUNK_TYPE_TABLE), _chunk(CHUNK_TYPE_HYPE), _chunk(CHUNK_TYPE_CAPTION)]
        original = list(chunks)
        filter_bm25_indexable(chunks)
        filter_dense_indexable(chunks)
        assert chunks == original
