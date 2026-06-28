"""Shared helpers for HybridRetriever unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.retrieval.hybrid_retriever import HybridRetriever


async def assert_optional_retriever_participates_in_rrf(
    retriever_kwarg: str,
    *,
    dense_chunk: Chunk | None = None,
    optional_chunk: Chunk | None = None,
    top_k: int = 3,
    query_text: str = "test",
) -> None:
    """Build a HybridRetriever with one optional source and assert RRF includes both hits."""
    dense = dense_chunk or Chunk(id="c0", document_id="doc", text="chunk 0")
    optional = optional_chunk or Chunk(id="c1", document_id="doc", text="chunk 1")

    dense_mock = MagicMock()
    dense_mock.retrieve.return_value = [(dense, 0.9)]
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = []
    bm25_mock.get_by_id.side_effect = lambda cid: optional if cid == optional.id else dense
    optional_mock = MagicMock()
    optional_mock.retrieve.return_value = [(optional, 0.95)]

    hr = HybridRetriever(
        dense=dense_mock,
        bm25=bm25_mock,
        **{retriever_kwarg: optional_mock},
    )
    results = await hr.retrieve(Query(text=query_text), top_k=top_k)
    ids = {chunk.id for chunk, _ in results}
    assert dense.id in ids
    assert optional.id in ids
    optional_mock.retrieve.assert_called_once()
