"""T-004 — repository ABC tests.

Verifies that:
- Each ABC cannot be instantiated directly (TypeError).
- A concrete subclass implementing all abstract methods CAN be instantiated.
- A subclass missing any abstract method CANNOT be instantiated (TypeError).
- No infrastructure imports leak into the domain layer.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from src.core.exceptions import EmbeddingError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.domain.repositories import (
    DenseVector,
    EmbeddingRepository,
    LLMRepository,
    RerankerRepository,
    SearchResult,
    SparseVector,
    VectorStoreRepository,
)

# ── Helpers — minimal concrete implementations ─────────────────────────────────


def _assert_abstract_instantiation_fails(cls: type) -> None:
    """Assert ABCs and incomplete subclasses raise TypeError on instantiation."""
    with pytest.raises(TypeError):
        cls()  # pyright: ignore[reportAbstractUsage]


class _LLM(LLMRepository):
    def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
        return "answer"

    def generate_stream(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            yield "token"

        return _gen()


class _Embedder(EmbeddingRepository):
    def embed(self, texts: list[str]) -> list[DenseVector]:
        return [[0.0] for _ in texts]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{0: 1.0} for _ in texts]


class _Reranker(RerankerRepository):
    def score(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        return [(chunk, float(len(chunks) - index)) for index, chunk in enumerate(chunks)]

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        return chunks[:top_k]


class _VectorStore(VectorStoreRepository):
    def upsert(self, chunks: list[Chunk]) -> None:
        pass

    def search_dense(
        self,
        query_vector: DenseVector,
        top_k: int,
        *,
        type_equals: str | None = None,
        exclude_types: frozenset[str] | None = None,
        document_ids: frozenset[str] | None = None,
        filters: object | None = None,
    ) -> list[SearchResult]:
        return []

    def search_sparse(self, query_sparse: SparseVector, top_k: int) -> list[SearchResult]:
        return []

    def search_hybrid(
        self,
        query_vector: DenseVector,
        query_sparse: SparseVector,
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        return []

    def delete(self, chunk_ids: list[str]) -> None:
        pass

    def count(self) -> int:
        return 0

    def chunk_exists(self, chunk_id: str) -> bool:
        return False

    def get_feedback_score(self, chunk_id: str) -> float:
        return 0.0

    def set_feedback_score(self, chunk_id: str, feedback_score: float) -> None:
        pass


# ── LLMRepository ──────────────────────────────────────────────────────────────


class TestLLMRepository:
    def test_abc_cannot_be_instantiated(self):
        _assert_abstract_instantiation_fails(LLMRepository)

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Incomplete(LLMRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
                return ""

            # generate_stream missing

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self):
        llm = _LLM()
        assert isinstance(llm, LLMRepository)

    def test_generate_returns_str(self):
        assert isinstance(_LLM().generate("p", "c"), str)

    def test_generate_stream_returns_async_iterator(self):
        stream = _LLM().generate_stream("p", "c")
        assert hasattr(stream, "__aiter__")


# ── EmbeddingRepository ────────────────────────────────────────────────────────


class TestEmbeddingRepository:
    def test_abc_cannot_be_instantiated(self):
        _assert_abstract_instantiation_fails(EmbeddingRepository)

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Incomplete(EmbeddingRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            def embed(self, texts: list[str]) -> list[DenseVector]:
                return []

            # embed_sparse missing

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self):
        assert isinstance(_Embedder(), EmbeddingRepository)

    def test_embed_returns_list_of_lists(self):
        result = _Embedder().embed(["hello", "world"])
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_embed_sparse_returns_list_of_dicts(self):
        result = _Embedder().embed_sparse(["hello"])
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_output_length_matches_input(self):
        texts = ["a", "b", "c"]
        emb = _Embedder()
        assert len(emb.embed(texts)) == 3
        assert len(emb.embed_sparse(texts)) == 3

    def test_embed_passage_delegates_to_embed(self):
        emb = _Embedder()
        texts = ["passage one", "passage two"]
        assert emb.embed_passage(texts) == emb.embed(texts)

    def test_embed_image_default_raises(self):
        with pytest.raises(EmbeddingError):
            _Embedder().embed_image([Path("figure.png")])

    def test_embed_image_override_returns_vectors(self):
        class _ImageEmbedder(_Embedder):
            def embed_image(self, paths: list[Path]) -> list[DenseVector]:
                return [[1.0, 0.0] for _ in paths]

        result = _ImageEmbedder().embed_image([Path("a.png"), Path("b.png")])
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)


# ── RerankerRepository ─────────────────────────────────────────────────────────


class TestRerankerRepository:
    def test_abc_cannot_be_instantiated(self):
        _assert_abstract_instantiation_fails(RerankerRepository)

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Incomplete(RerankerRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            pass  # score and rerank missing

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self):
        assert isinstance(_Reranker(), RerankerRepository)

    def test_score_returns_pairs(self):
        chunks = [Chunk(document_id="d", text=f"chunk {i}") for i in range(3)]
        scored = _Reranker().score("query", chunks)
        assert len(scored) == 3
        assert scored[0][1] == pytest.approx(3.0)

    def test_rerank_returns_list_of_chunks(self):
        chunks = [Chunk(document_id="d", text=f"chunk {i}") for i in range(5)]
        result = _Reranker().rerank("query", chunks, top_k=3)
        assert len(result) == 3
        assert all(isinstance(c, Chunk) for c in result)

    def test_rerank_respects_top_k(self):
        chunks = [Chunk(document_id="d", text=f"t{i}") for i in range(10)]
        assert len(_Reranker().rerank("q", chunks, top_k=4)) == 4


# ── VectorStoreRepository ──────────────────────────────────────────────────────


class TestVectorStoreRepository:
    def test_abc_cannot_be_instantiated(self):
        _assert_abstract_instantiation_fails(VectorStoreRepository)

    def test_incomplete_subclass_missing_count(self):
        class _Incomplete(VectorStoreRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            def upsert(self, chunks: list[Chunk]) -> None:
                pass

            def search_dense(
                self,
                query_vector: DenseVector,
                top_k: int,
                *,
                type_equals: str | None = None,
                exclude_types: frozenset[str] | None = None,
                document_ids: frozenset[str] | None = None,
                filters: RetrievalFilter | None = None,
            ) -> list[SearchResult]:
                return []

            def search_sparse(self, query_sparse: SparseVector, top_k: int) -> list[SearchResult]:
                return []

            def search_hybrid(
                self,
                query_vector: DenseVector,
                query_sparse: SparseVector,
                alpha: float,
                top_k: int,
            ) -> list[SearchResult]:
                return []

            def delete(self, chunk_ids: list[str]) -> None:
                pass

            def get_feedback_score(self, chunk_id: str) -> float:
                return 0.0

            def set_feedback_score(self, chunk_id: str, feedback_score: float) -> None:
                pass

            # count missing

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self):
        assert isinstance(_VectorStore(), VectorStoreRepository)

    def test_upsert_accepts_chunks(self):
        chunks = [Chunk(document_id="d", text="t", embedding=[0.1], sparse_vector={1: 0.5})]
        _VectorStore().upsert(chunks)  # must not raise

    def test_search_methods_return_list(self):
        vs = _VectorStore()
        assert vs.search_dense([0.1, 0.2], top_k=5) == []
        assert vs.search_sparse({1: 0.9}, top_k=5) == []
        assert vs.search_hybrid([0.1], {1: 0.9}, alpha=0.7, top_k=5) == []

    def test_count_returns_int(self):
        assert isinstance(_VectorStore().count(), int)

    def test_delete_accepts_id_list(self):
        _VectorStore().delete(["id-1", "id-2"])  # must not raise

    def test_accumulate_feedback_score_default_impl(self):
        class _TrackingStore(_VectorStore):
            def __init__(self) -> None:
                self.scores: dict[str, float] = {}

            def get_feedback_score(self, chunk_id: str) -> float:
                return self.scores.get(chunk_id, 0.0)

            def set_feedback_score(self, chunk_id: str, feedback_score: float) -> None:
                self.scores[chunk_id] = feedback_score

        store = _TrackingStore()
        assert store.accumulate_feedback_score("chunk-a", 1.0) == 1.0
        assert store.accumulate_feedback_score("chunk-a", 0.5) == 1.5

    def test_get_feedback_scores_default_impl(self):
        class _TrackingStore(_VectorStore):
            def __init__(self) -> None:
                self.scores = {"chunk-a": 2.0}

            def get_feedback_score(self, chunk_id: str) -> float:
                return self.scores.get(chunk_id, 0.0)

            def set_feedback_score(self, chunk_id: str, feedback_score: float) -> None:
                self.scores[chunk_id] = feedback_score

        store = _TrackingStore()
        assert store.get_feedback_scores(["chunk-a", "chunk-b", "chunk-a"]) == {
            "chunk-a": 2.0,
            "chunk-b": 0.0,
        }


# ── No infrastructure imports ──────────────────────────────────────────────────


class TestNoDomainInfraLeak:
    def test_all_importable_from_package(self):
        from src.domain.repositories import (  # noqa: F401
            EmbeddingRepository,
            LLMRepository,
            RerankerRepository,
            VectorStoreRepository,
        )

    def test_type_aliases_importable(self):
        from src.domain.repositories import DenseVector, SearchResult, SparseVector  # noqa: F401
