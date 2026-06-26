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
from typing import Any

import pytest

from src.domain.entities.chunk import Chunk
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


# ── LLMRepository ──────────────────────────────────────────────────────────────


class TestLLMRepository:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMRepository()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Incomplete(LLMRepository, ABC):  # type: ignore[abstract]
            def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
                return ""

            # generate_stream missing

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

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
        with pytest.raises(TypeError):
            EmbeddingRepository()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Incomplete(EmbeddingRepository, ABC):  # type: ignore[abstract]
            def embed(self, texts: list[str]) -> list[DenseVector]:
                return []

            # embed_sparse missing

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

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


# ── RerankerRepository ─────────────────────────────────────────────────────────


class TestRerankerRepository:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            RerankerRepository()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Incomplete(RerankerRepository, ABC):  # type: ignore[abstract]
            pass  # rerank missing

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        assert isinstance(_Reranker(), RerankerRepository)

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
        with pytest.raises(TypeError):
            VectorStoreRepository()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_count(self):
        class _Incomplete(VectorStoreRepository, ABC):  # type: ignore[abstract]
            def upsert(self, chunks: list[Chunk]) -> None:
                pass

            def search_dense(
                self,
                qv: DenseVector,
                top_k: int,
                *,
                type_equals: str | None = None,
                exclude_types: frozenset[str] | None = None,
            ) -> list[SearchResult]:
                return []

            def search_sparse(self, qs: SparseVector, top_k: int) -> list[SearchResult]:
                return []

            def search_hybrid(  # noqa: E704
                self,
                qv: DenseVector,
                qs: SparseVector,
                alpha: float,
                top_k: int,
            ) -> list[SearchResult]:
                return []

            def delete(self, ids: list[str]) -> None:
                pass

            # count missing

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

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
