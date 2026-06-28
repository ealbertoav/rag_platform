"""T-125 — Hierarchical indexer and retriever tests."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.core.constants import CHUNK_TYPE_DETAIL, CHUNK_TYPE_KEY, CHUNK_TYPE_SUMMARY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.entities.query import Query
from src.rag.enrichment.hierarchical_indexer import (
    HierarchicalIndexer,
    is_detail_chunk,
    is_summary_chunk,
    make_summary_chunk,
    tag_detail_chunks,
)
from src.rag.retrieval.hierarchical_retriever import HierarchicalRetriever
from tests.unit.hybrid_retriever_helpers import assert_optional_retriever_participates_in_rrf


def _document(text: str = "Revenue grew 12%. Costs fell 3%.") -> Document:
    return Document(id="doc-1", source="/data/raw/report.pdf", content=text)


def _detail_chunk(text: str = "Revenue grew 12%.") -> Chunk:
    return Chunk(id="detail-1", document_id="doc-1", text=text, metadata={"section": "Revenue"})


class TestChunkMetadata:
    def test_tag_detail_chunks(self):
        tagged = tag_detail_chunks([_detail_chunk()])
        assert tagged[0].metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_DETAIL
        assert is_detail_chunk(tagged[0])

    def test_make_summary_chunk(self):
        summary = make_summary_chunk(_document(), "Annual report covering revenue and costs.")
        assert summary.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_SUMMARY
        assert summary.document_id == "doc-1"
        assert is_summary_chunk(summary)


class TestHierarchicalIndexer:
    def test_indexes_tagged_details_and_embedded_summary(self):
        llm = MagicMock()
        llm.generate.return_value = "Summary of revenue and cost trends."
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1, 0.2]], [{1: 0.9}])
        indexer = HierarchicalIndexer(llm=llm, embedder=embedder)

        tagged, summaries = indexer.index(_document(), [_detail_chunk()])

        assert len(tagged) == 1
        assert is_detail_chunk(tagged[0])
        assert len(summaries) == 1
        assert is_summary_chunk(summaries[0])
        assert summaries[0].embedding == [0.1, 0.2]

    def test_summary_failure_returns_tagged_details_only(self, caplog):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        embedder = MagicMock()
        indexer = HierarchicalIndexer(llm=llm, embedder=embedder)

        with caplog.at_level(logging.WARNING):
            tagged, summaries = indexer.index(_document(), [_detail_chunk()])

        assert is_detail_chunk(tagged[0])
        assert summaries == []
        assert "Document summary generation failed" in caplog.text

    def test_embedding_failure_returns_tagged_details_only(self, caplog):
        llm = MagicMock()
        llm.generate.return_value = "Summary text."
        embedder = MagicMock()
        embedder.embed_both.side_effect = RuntimeError("embed failed")
        indexer = HierarchicalIndexer(llm=llm, embedder=embedder)

        with caplog.at_level(logging.WARNING):
            tagged, summaries = indexer.index(_document(), [_detail_chunk()])

        assert is_detail_chunk(tagged[0])
        assert summaries == []
        assert "Embedding document summary failed" in caplog.text


class TestHierarchicalRetriever:
    def test_two_stage_search_returns_detail_chunks_only(self):
        summary = make_summary_chunk(_document(), "Revenue and cost overview.")
        detail = Chunk(
            id="detail-2",
            document_id="doc-1",
            text="Costs fell 3%.",
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_DETAIL},
        )
        embedder = MagicMock()
        embedder.embed_query.return_value = [[0.5, 0.6]]
        vector_store = MagicMock()
        vector_store.search_dense.side_effect = [
            [(summary, 0.91)],
            [(detail, 0.82)],
        ]

        retriever = HierarchicalRetriever(
            embedder=embedder, vector_store=vector_store, summary_top_k=3
        )
        results = retriever.retrieve(Query(text="revenue trends"), top_k=5)

        assert len(results) == 1
        assert results[0][0].id == detail.id
        assert not is_summary_chunk(results[0][0])

        first_call, second_call = vector_store.search_dense.call_args_list
        assert first_call.kwargs["type_equals"] == CHUNK_TYPE_SUMMARY
        assert second_call.kwargs["type_equals"] == CHUNK_TYPE_DETAIL
        assert second_call.kwargs["document_ids"] == frozenset({"doc-1"})

    def test_no_summary_matches_returns_empty(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [[0.1]]
        vector_store = MagicMock()
        vector_store.search_dense.return_value = []

        retriever = HierarchicalRetriever(embedder=embedder, vector_store=vector_store)
        assert retriever.retrieve(Query(text="q"), top_k=5) == []
        vector_store.search_dense.assert_called_once()

    def test_uses_precomputed_query_embedding(self):
        embedder = MagicMock()
        vector_store = MagicMock()
        vector_store.search_dense.return_value = []

        retriever = HierarchicalRetriever(embedder=embedder, vector_store=vector_store)
        retriever.retrieve(Query(text="q", embedding=[0.1, 0.2]), top_k=5)
        embedder.embed_query.assert_not_called()


class TestHybridHierarchicalIntegration:
    @pytest.mark.asyncio
    async def test_hierarchical_results_participate_in_rrf(self):
        await assert_optional_retriever_participates_in_rrf("hierarchical_retriever")
