"""T-014 — BM25 index and retriever tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.infrastructure.vectordb.bm25 import BM25Index
from src.rag.retrieval.bm25_retriever import BM25Retriever

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(text: str, doc_id: str = "doc-1", idx: int = 0) -> Chunk:
    return Chunk(id=f"chunk-{idx:04d}", document_id=doc_id, text=text)


_CORPUS = [
    _chunk("The quick brown fox jumps over the lazy dog", idx=0),
    _chunk("Kubernetes pod scheduling and node affinity rules", idx=1),
    _chunk("IAM roles and policies for AWS EKS clusters", idx=2),
    _chunk("Vector databases store embeddings for similarity search", idx=3),
    _chunk("Python async programming with asyncio", idx=4),
]


# ── tokenization behavior (tested through the public index API) ───────────────


class TestTokenizationBehaviour:
    def test_search_is_case_insensitive(self):
        # BM25 needs multiple documents for non-zero IDF; uppercase query must
        # find the same chunk as the lowercase equivalent.
        idx = BM25Index()
        idx.index(_CORPUS)
        lower = idx.search("kubernetes", top_k=1)
        upper = idx.search("KUBERNETES", top_k=1)
        assert lower and upper
        assert lower[0][0].id == upper[0][0].id

    def test_query_matches_mixed_case_corpus(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("IAM Roles AWS", top_k=1)
        assert results[0][0].id == "chunk-0002"

    def test_empty_query_returns_empty(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        assert idx.search("", top_k=5) == []


# ── BM25Index ──────────────────────────────────────────────────────────────────


class TestBM25Index:
    def test_empty_index_search_returns_empty(self):
        idx = BM25Index()
        assert idx.search("kubernetes", top_k=5) == []

    def test_size_zero_before_indexing(self):
        assert BM25Index().size == 0

    def test_size_after_index(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        assert idx.size == len(_CORPUS)

    def test_search_returns_list_of_tuples(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("kubernetes", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_returns_chunks(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("kubernetes", top_k=3)
        assert all(isinstance(c, Chunk) for c, _ in results)

    def test_search_scores_are_float(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("kubernetes", top_k=3)
        assert all(isinstance(s, float) for _, s in results)

    def test_search_sorted_descending(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("kubernetes", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_respected(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        assert len(idx.search("the", top_k=2)) <= 2

    def test_relevant_chunk_ranks_first(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("kubernetes pod scheduling", top_k=3)
        assert results[0][0].id == "chunk-0001"

    def test_zero_score_results_filtered(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        results = idx.search("kubernetes", top_k=10)
        assert all(s > 0 for _, s in results)

    def test_index_replaces_existing(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        new_chunks = [_chunk("completely different content", idx=99)]
        idx.index(new_chunks)
        assert idx.size == 1
        results = idx.search("kubernetes", top_k=5)
        assert results == []

    def test_add_appends_and_rebuilds(self):
        idx = BM25Index()
        idx.index(_CORPUS[:3])
        idx.add(_CORPUS[3:])
        assert idx.size == len(_CORPUS)

    def test_add_deduplicates_by_id(self):
        idx = BM25Index()
        idx.index(_CORPUS[:2])
        idx.add(_CORPUS[:2])  # same chunks again
        assert idx.size == 2

    def test_add_new_chunk_is_searchable(self):
        idx = BM25Index()
        idx.index(_CORPUS[:3])
        extra = _chunk("LangGraph agentic workflow orchestration", idx=99)
        idx.add([extra])
        results = idx.search("langgraph agentic", top_k=1)
        assert results[0][0].id == extra.id


# ── persistence ────────────────────────────────────────────────────────────────


class TestBM25IndexPersistence:
    def test_save_creates_file(self, tmp_path: Path):
        path = tmp_path / "bm25.pkl"
        idx = BM25Index(index_path=path)
        idx.index(_CORPUS)
        idx.save()
        assert path.exists()

    def test_load_restores_size(self, tmp_path: Path):
        path = tmp_path / "bm25.pkl"
        idx = BM25Index(index_path=path)
        idx.index(_CORPUS)
        idx.save()

        idx2 = BM25Index(index_path=path)
        idx2.load()
        assert idx2.size == len(_CORPUS)

    def test_load_restores_search(self, tmp_path: Path):
        path = tmp_path / "bm25.pkl"
        idx = BM25Index(index_path=path)
        idx.index(_CORPUS)
        idx.save()

        idx2 = BM25Index(index_path=path)
        idx2.load()
        results = idx2.search("kubernetes pod scheduling", top_k=1)
        assert results[0][0].id == "chunk-0001"

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(VectorStoreError):
            BM25Index(index_path=tmp_path / "missing.pkl").load()

    def test_load_or_create_returns_empty_when_missing(self, tmp_path: Path):
        idx = BM25Index.load_or_create(tmp_path / "missing.pkl")
        assert idx.size == 0

    def test_load_or_create_loads_existing(self, tmp_path: Path):
        path = tmp_path / "bm25.pkl"
        saved = BM25Index(index_path=path)
        saved.index(_CORPUS)
        saved.save()

        idx = BM25Index.load_or_create(path)
        assert idx.size == len(_CORPUS)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "bm25.pkl"
        idx = BM25Index(index_path=path)
        idx.index([_CORPUS[0]])
        idx.save()
        assert path.exists()


# ── BM25Retriever ──────────────────────────────────────────────────────────────


class TestBM25Retriever:
    def test_search_delegates_to_index(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        retriever = BM25Retriever(idx)
        results = retriever.search("IAM roles AWS", top_k=2)
        assert len(results) <= 2
        assert results[0][0].id == "chunk-0002"

    def test_index_replaces_chunks(self):
        retriever = BM25Retriever(BM25Index())
        retriever.index(_CORPUS)
        assert retriever.size == len(_CORPUS)

    def test_add_appends(self):
        retriever = BM25Retriever(BM25Index())
        retriever.index(_CORPUS[:3])
        retriever.add(_CORPUS[3:])
        assert retriever.size == len(_CORPUS)

    def test_from_disk_returns_retriever(self, tmp_path: Path):
        retriever = BM25Retriever.from_disk(tmp_path / "new.pkl")
        assert isinstance(retriever, BM25Retriever)
        assert retriever.size == 0

    def test_from_disk_loads_existing(self, tmp_path: Path):
        path = tmp_path / "idx.pkl"
        idx = BM25Index(index_path=path)
        idx.index(_CORPUS)
        idx.save()

        retriever = BM25Retriever.from_disk(path)
        assert retriever.size == len(_CORPUS)

    def test_save_persists(self, tmp_path: Path):
        path = tmp_path / "ret.pkl"
        retriever = BM25Retriever(BM25Index(index_path=path))
        retriever.index(_CORPUS)
        retriever.save()
        assert path.exists()
