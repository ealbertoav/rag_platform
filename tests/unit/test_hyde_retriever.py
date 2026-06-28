"""T-130 — HyDE (Hypothetical Document Embedding) retriever tests."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from src.core.constants import CHUNK_TYPE_HYPE
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.retrieval.hyde_retriever import HyDERetriever
from tests.unit.hybrid_retriever_helpers import assert_optional_retriever_participates_in_rrf


def _chunk(chunk_id: str = "c1", text: str = "Revenue grew 12%.") -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=text)


class TestHyDERetriever:
    def test_generates_hypothetical_doc_and_searches_dense(self):
        llm = MagicMock()
        llm.generate.return_value = "Annual revenue increased by 12% year over year."
        embedder = MagicMock()
        embedder.embed_query.return_value = [[0.3, 0.4]]
        vector_store = MagicMock()
        chunk = _chunk()
        vector_store.search_dense.return_value = [(chunk, 0.91)]

        retriever = HyDERetriever(llm=llm, embedder=embedder, vector_store=vector_store)
        results = retriever.retrieve(Query(text="How did revenue change?"), top_k=5)

        llm.generate.assert_called_once()
        embedder.embed_query.assert_called_once_with(
            ["Annual revenue increased by 12% year over year."],
        )
        vector_store.search_dense.assert_called_once()
        _, kwargs = vector_store.search_dense.call_args
        assert CHUNK_TYPE_HYPE in kwargs["exclude_types"]
        assert len(results) == 1
        assert results[0][0].id == "c1"
        assert results[0][1] == pytest.approx(0.91)

    def test_llm_failure_returns_empty(self, caplog):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        embedder = MagicMock()
        vector_store = MagicMock()

        retriever = HyDERetriever(llm=llm, embedder=embedder, vector_store=vector_store)
        with caplog.at_level(logging.WARNING):
            results = retriever.retrieve(Query(text="revenue?"), top_k=5)

        assert results == []
        embedder.embed_query.assert_not_called()
        assert "HyDE retrieval failed" in caplog.text

    def test_empty_hypothetical_doc_returns_empty(self):
        llm = MagicMock()
        llm.generate.return_value = "   "
        embedder = MagicMock()
        vector_store = MagicMock()

        retriever = HyDERetriever(llm=llm, embedder=embedder, vector_store=vector_store)
        assert retriever.retrieve(Query(text="q"), top_k=3) == []
        embedder.embed_query.assert_not_called()

    def test_generate_hypothetical_doc_uses_prompt(self):
        llm = MagicMock()
        llm.generate.return_value = "Hypothetical passage."
        retriever = HyDERetriever(llm=llm, embedder=MagicMock(), vector_store=MagicMock())

        text = retriever.generate_hypothetical_doc("What is EBITDA?")
        assert text == "Hypothetical passage."
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert "What is EBITDA?" in prompt


class TestHybridHyDEIntegration:
    @pytest.mark.asyncio
    async def test_hyde_results_participate_in_rrf(self):
        await assert_optional_retriever_participates_in_rrf("hyde_retriever")

    @pytest.mark.asyncio
    async def test_hyde_failure_still_returns_standard_results(self):
        from src.rag.retrieval.hybrid_retriever import HybridRetriever

        dense_chunk = Chunk(id="c0", document_id="doc", text="dense hit")
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(dense_chunk, 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []
        hyde_mock = MagicMock()
        hyde_mock.retrieve.return_value = []

        hr = HybridRetriever(
            dense=dense_mock,
            bm25=bm25_mock,
            hyde_retriever=hyde_mock,
        )
        results = await hr.retrieve(Query(text="test"), top_k=3)
        assert len(results) == 1
        assert results[0][0].id == "c0"
