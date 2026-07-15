"""T-260 — ImageDenseRetriever tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.retrieval.image_dense_retriever import ImageDenseRetriever

# ── helpers ────────────────────────────────────────────────────────────────────


_VEC = [0.1, 0.2, 0.3, 0.4]
_CHUNK = Chunk(document_id="doc-1", text="relevant figure", modality="figure")


def _retriever(
    embedding: list[float] | None = None,
    results: list[tuple[Chunk, float]] | None = None,
    image_dense_dim: int | None = 512,
) -> tuple[ImageDenseRetriever, MagicMock, MagicMock]:
    embedder = MagicMock()
    embedder.embed_query.return_value = [embedding if embedding is not None else _VEC]
    vector_store = MagicMock()
    vector_store.image_dense_dim = image_dense_dim
    vector_store.search_image_dense.return_value = (
        results if results is not None else [(_CHUNK, 0.9)]
    )
    return (
        ImageDenseRetriever(embedder=embedder, vector_store=vector_store),
        embedder,
        vector_store,
    )


def _query(text: str = "a chart of revenue", embedding: list[float] | None = None) -> Query:
    return Query(text=text, embedding=embedding)


# ── enabled ────────────────────────────────────────────────────────────────────


class TestEnabled:
    def test_true_when_store_has_image_dense_dim(self):
        retriever, *_ = _retriever(image_dense_dim=512)
        assert retriever.enabled is True

    def test_false_when_store_has_no_image_dense_dim(self):
        retriever, *_ = _retriever(image_dense_dim=None)
        assert retriever.enabled is False


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
        embedder.embed_query.assert_called_once_with(["a chart of revenue"])

    def test_skips_embed_when_embedding_present(self):
        retriever, embedder, _ = _retriever()
        retriever.retrieve(_query(embedding=_VEC), top_k=5)
        embedder.embed_query.assert_not_called()  # type: ignore[attr-defined]

    def test_uses_precomputed_embedding_for_search(self):
        retriever, _, vector_store = _retriever()
        custom_vec = [0.9, 0.8, 0.7, 0.6]
        retriever.retrieve(_query(embedding=custom_vec), top_k=3)
        vector_store.search_image_dense.assert_called_once_with(
            custom_vec,
            top_k=3,
            filters=None,
        )

    def test_top_k_forwarded(self):
        retriever, _, vector_store = _retriever()
        retriever.retrieve(_query(), top_k=7)
        _, kwargs = vector_store.search_image_dense.call_args
        assert kwargs["top_k"] == 7

    def test_returns_vector_store_results(self):
        expected = [(_CHUNK, 0.95)]
        retriever, *_ = _retriever(results=expected)
        assert retriever.retrieve(_query(), top_k=1) == expected

    def test_empty_results_returned_as_is(self):
        retriever, *_ = _retriever(results=[])
        assert retriever.retrieve(_query(), top_k=5) == []

    def test_returns_empty_when_disabled(self):
        retriever, embedder, vector_store = _retriever(image_dense_dim=None)
        result = retriever.retrieve(_query(), top_k=5)
        assert result == []
        embedder.embed_query.assert_not_called()  # type: ignore[attr-defined]
        vector_store.search_image_dense.assert_not_called()  # type: ignore[attr-defined]


class TestImageDenseRetrieverProperties:
    def test_vector_store_property(self):
        retriever, _, vector_store = _retriever()
        assert retriever.vector_store is vector_store


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
        embedder.embed_query.assert_not_called()  # type: ignore[attr-defined]

    def test_original_text_preserved(self):
        retriever, *_ = _retriever()
        result = retriever.embed_query(_query("my question"))
        assert result.text == "my question"

    def test_embeds_query_text(self):
        retriever, embedder, _ = _retriever()
        retriever.embed_query(_query("specific question"))
        embedder.embed_query.assert_called_once_with(["specific question"])
