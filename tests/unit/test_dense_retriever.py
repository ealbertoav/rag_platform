"""T-021 — DenseRetriever tests."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.retrieval.dense_retriever import DenseRetriever

# ── helpers ────────────────────────────────────────────────────────────────────


_VEC = [0.1, 0.2, 0.3, 0.4]
_CHUNK = Chunk(document_id="doc-1", text="relevant chunk")


def _retriever(
    embedding: list[float] | None = None,
    results: list[tuple[Chunk, float]] | None = None,
) -> tuple[DenseRetriever, MagicMock, MagicMock]:
    embedder = MagicMock()
    embedder.embed.return_value = [embedding if embedding is not None else _VEC]
    vector_store = MagicMock()
    vector_store.search_dense.return_value = results if results is not None else [(_CHUNK, 0.9)]
    return DenseRetriever(embedder=embedder, vector_store=vector_store), embedder, vector_store


def _query(text: str = "What is IAM?", embedding: list[float] | None = None) -> Query:
    return Query(text=text, embedding=embedding)


# ── retrieve ───────────────────────────────────────────────────────────────────


class TestRetrieve:
    def test_returns_list_of_tuples(self):
        retriever, *_ = _retriever()
        result = retriever.retrieve(_query(), top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_chunk_and_score_types(self):
        retriever, *_ = _retriever()
        chunk, score = retriever.retrieve(_query(), top_k=1)[0]
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    def test_calls_embed_for_unembedded_query(self):
        retriever, embedder, _ = _retriever()
        retriever.retrieve(_query(), top_k=5)
        embedder.embed.assert_called_once_with(["What is IAM?"])

    def test_skips_embed_when_embedding_present(self):
        retriever, embedder, _ = _retriever()
        retriever.retrieve(_query(embedding=_VEC), top_k=5)
        embedder.embed.assert_not_called()  # type: ignore[attr-defined]

    def test_uses_precomputed_embedding_for_search(self):
        retriever, _, vector_store = _retriever()
        custom_vec = [0.9, 0.8, 0.7, 0.6]
        retriever.retrieve(_query(embedding=custom_vec), top_k=3)
        vector_store.search_dense.assert_called_once_with(custom_vec, top_k=3)

    def test_top_k_forwarded(self):
        retriever, _, vector_store = _retriever()
        retriever.retrieve(_query(), top_k=7)
        _, kwargs = vector_store.search_dense.call_args
        assert kwargs["top_k"] == 7

    def test_returns_vector_store_results(self):
        expected = [(_CHUNK, 0.95)]
        retriever, *_ = _retriever(results=expected)
        assert retriever.retrieve(_query(), top_k=1) == expected

    def test_empty_results_returned_as_is(self):
        retriever, *_ = _retriever(results=[])
        assert retriever.retrieve(_query(), top_k=5) == []


# ── embed_query ────────────────────────────────────────────────────────────────


class TestEmbedQuery:
    def test_returns_query_with_embedding(self):
        retriever, *_ = _retriever()
        result = retriever.embed_query(_query())
        assert result.embedding == _VEC

    def test_noop_when_embedding_already_set(self):
        retriever, embedder, _ = _retriever()
        q = _query(embedding=_VEC)
        result = retriever.embed_query(q)
        assert result is q
        embedder.embed.assert_not_called()  # type: ignore[attr-defined]

    def test_original_text_preserved(self):
        retriever, *_ = _retriever()
        result = retriever.embed_query(_query("my question"))
        assert result.text == "my question"

    def test_embeds_query_text(self):
        retriever, embedder, _ = _retriever()
        retriever.embed_query(_query("specific question"))
        embedder.embed.assert_called_once_with(["specific question"])
