"""T-122 — HyPE indexer and retriever tests."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_KEY, SOURCE_CHUNK_ID_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.enrichment.hype_indexer import HyPEIndexer, is_hype_question, make_hype_chunk
from src.rag.retrieval.hype_retriever import HyPERetriever


def _source_chunk(text: str = "Revenue grew 12% year over year.") -> Chunk:
    return Chunk(
        id="source-1",
        document_id="doc-1",
        text=text,
        metadata={"section": "Revenue"},
    )


class TestMakeHypeChunk:
    def test_metadata_links_to_source(self):
        source = _source_chunk()
        hype = make_hype_chunk(source, "What was revenue growth?")
        assert hype.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_HYPE
        assert hype.metadata[SOURCE_CHUNK_ID_KEY] == source.id
        assert hype.document_id == source.document_id
        assert hype.text == "What was revenue growth?"
        assert is_hype_question(hype)


class TestHyPEIndexer:
    def test_indexes_embedded_questions(self):
        llm = MagicMock()
        llm.generate.return_value = '["What was revenue growth?"]'
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1, 0.2]], [{1: 0.9}])
        indexer = HyPEIndexer(llm=llm, embedder=embedder, n_questions=2)
        result = indexer.index([_source_chunk()])
        assert len(result) == 1
        assert is_hype_question(result[0])
        assert result[0].embedding == [0.1, 0.2]

    def test_failure_on_one_chunk_continues(self, caplog):
        llm = MagicMock()
        llm.generate.side_effect = [RuntimeError("LLM down"), '["Question two?"]']
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1]], [{1: 0.9}])
        indexer = HyPEIndexer(llm=llm, embedder=embedder, n_questions=1)
        with caplog.at_level(logging.WARNING):
            result = indexer.index([_source_chunk("A"), _source_chunk("B")])
        assert len(result) == 1
        assert "HyPE question generation failed" in caplog.text

    def test_returns_empty_when_embedding_fails(self, caplog):
        llm = MagicMock()
        llm.generate.return_value = '["Question one?"]'
        embedder = MagicMock()
        embedder.embed_both.side_effect = RuntimeError("embed failed")
        indexer = HyPEIndexer(llm=llm, embedder=embedder, n_questions=1)
        with caplog.at_level(logging.WARNING):
            assert indexer.index([_source_chunk()]) == []
        assert "Embedding HyPE questions failed" in caplog.text


class TestHyPERetriever:
    def test_searches_hype_type_and_resolves_source(self):
        source = _source_chunk()
        hype = make_hype_chunk(source, "What was revenue growth?")
        embedder = MagicMock()
        embedder.embed_query.return_value = [[0.5, 0.6]]
        vector_store = MagicMock()
        vector_store.search_dense.return_value = [(hype, 0.88)]
        lookup = MagicMock()
        lookup.get_by_id.return_value = source

        retriever = HyPERetriever(embedder=embedder, vector_store=vector_store, chunk_lookup=lookup)
        results = retriever.retrieve(Query(text="revenue growth?"), top_k=3)

        vector_store.search_dense.assert_called_once()
        _, kwargs = vector_store.search_dense.call_args
        assert kwargs["type_equals"] == CHUNK_TYPE_HYPE
        assert len(results) == 1
        assert results[0][0].id == source.id
        assert results[0][1] == pytest.approx(0.88)

    def test_uses_precomputed_query_embedding(self):
        source = _source_chunk()
        embedder = MagicMock()
        vector_store = MagicMock()
        vector_store.search_dense.return_value = []
        lookup = MagicMock()
        lookup.get_by_id.return_value = source

        retriever = HyPERetriever(embedder=embedder, vector_store=vector_store, chunk_lookup=lookup)
        retriever.retrieve(Query(text="q", embedding=[0.1, 0.2]), top_k=5)
        embedder.embed_query.assert_not_called()

    def test_empty_hits_returns_empty(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [[0.1]]
        vector_store = MagicMock()
        vector_store.search_dense.return_value = []
        lookup = MagicMock()

        retriever = HyPERetriever(embedder=embedder, vector_store=vector_store, chunk_lookup=lookup)
        assert retriever.retrieve(Query(text="q"), top_k=5) == []


class TestHybridHyPEIntegration:
    @pytest.mark.asyncio
    async def test_hype_results_participate_in_rrf(self):
        from src.rag.retrieval.hybrid_retriever import HybridRetriever

        source = Chunk(id="c0", document_id="doc", text="chunk 0")
        hype_only = Chunk(id="c1", document_id="doc", text="chunk 1")

        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(source, 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []
        bm25_mock.get_by_id.side_effect = lambda cid: source if cid == source.id else hype_only
        hype_mock = MagicMock()
        hype_mock.retrieve.return_value = [(hype_only, 0.95)]

        hr = HybridRetriever(
            dense=dense_mock,
            bm25=bm25_mock,
            hype_retriever=hype_mock,
        )
        results = await hr.retrieve(Query(text="test"), top_k=3)
        ids = {chunk.id for chunk, _ in results}
        assert "c0" in ids
        assert "c1" in ids
        hype_mock.retrieve.assert_called_once()
