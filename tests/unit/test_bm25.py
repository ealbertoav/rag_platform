"""T-014 — BM25 index and retriever tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
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

    def test_document_id_filter_restricts_results(self):
        idx = BM25Index()
        scoped = [
            _chunk("kubernetes pod scheduling", doc_id="doc-a", idx=0),
            _chunk("vector databases store embeddings", doc_id="doc-b", idx=1),
            _chunk("python async programming", doc_id="doc-c", idx=2),
        ]
        idx.index(scoped)
        filt = RetrievalFilter(document_ids=["doc-a"])
        results = idx.search("kubernetes", top_k=5, filters=filt)
        assert len(results) == 1
        assert results[0][0].document_id == "doc-a"

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

    def test_save_skips_when_unchanged(self, tmp_path: Path, caplog):
        path = tmp_path / "bm25.pkl"
        idx = BM25Index(index_path=path)
        idx.index(_CORPUS)
        idx.save()
        mtime = path.stat().st_mtime
        import time

        time.sleep(0.01)
        idx.save()
        assert path.stat().st_mtime == mtime

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

    def test_bm25_index_property(self):
        idx = BM25Index()
        retriever = BM25Retriever(idx)
        assert retriever.bm25_index is idx


class TestBM25IndexErrors:
    def test_save_os_error_raises(self, tmp_path: Path):
        idx = BM25Index(index_path=tmp_path / "bm25.pkl")
        idx.index(_CORPUS)
        with (
            patch("pathlib.Path.open", side_effect=OSError("disk full")),
            pytest.raises(VectorStoreError, match="Cannot save"),
        ):
            idx.save()

    def test_load_corrupt_json_raises(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        path.write_text("not-json", encoding="utf-8")
        with pytest.raises(VectorStoreError, match="Cannot load"):
            BM25Index(index_path=path).load()

    def test_migrates_legacy_pickle_to_json(self, tmp_path: Path):
        import pickle

        legacy = tmp_path / "bm25.pkl"
        json_path = tmp_path / "bm25.json"
        payload = {"chunks": _CORPUS}
        legacy.write_bytes(pickle.dumps(payload))

        idx = BM25Index.load_or_create(json_path)
        assert idx.size == len(_CORPUS)
        assert json_path.exists()
        assert json_path.read_text(encoding="utf-8").startswith("{")

    def test_migrate_legacy_loads_json_when_created_under_lock(self, tmp_path: Path):
        """Cover the race where another process writes JSON before we migrate."""
        import pickle

        legacy = tmp_path / "bm25.pkl"
        json_path = tmp_path / "bm25.json"
        legacy.write_bytes(pickle.dumps({"chunks": _CORPUS}))

        winner = BM25Index(index_path=json_path)
        winner.index(_CORPUS[:2])
        winner.save()

        idx = BM25Index(index_path=json_path)
        idx._migrate_legacy_pickle(legacy)

        assert idx.size == 2
        assert {c.id for c in idx.chunks} == {c.id for c in _CORPUS[:2]}

    def test_concurrent_pickle_migration_is_safe(self, tmp_path: Path):
        import json
        import pickle
        import threading

        legacy = tmp_path / "bm25.pkl"
        json_path = tmp_path / "bm25.json"
        legacy.write_bytes(pickle.dumps({"chunks": _CORPUS}))

        results: list[int] = []
        errors: list[Exception] = []

        def migrate() -> None:
            try:
                idx = BM25Index.load_or_create(json_path)
                results.append(idx.size)
            except Exception as exc:  # pragma: no cover - test assertion below
                errors.append(exc)

        threads = [threading.Thread(target=migrate) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert all(size == len(_CORPUS) for size in results)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(payload["chunks"]) == len(_CORPUS)

    def test_chunks_property_returns_snapshot(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        chunks = idx.chunks
        assert len(chunks) == len(_CORPUS)
        chunks.clear()
        assert idx.size == len(_CORPUS)

    def test_iter_chunks_yields_without_copying_corpus(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        streamed = list(idx.iter_chunks())
        assert [c.id for c in streamed] == [c.id for c in _CORPUS]

    def test_get_by_id_miss_returns_none(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        assert idx.get_by_id("missing-id") is None

    def test_get_by_id_hit_returns_chunk(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        found = idx.get_by_id("chunk-0001")
        assert found is not None
        assert found.id == "chunk-0001"

    def test_remove_by_ids_empty_is_noop(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        idx.remove_by_ids([])
        assert idx.size == len(_CORPUS)

    def test_remove_by_ids_removes_chunks(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        idx.remove_by_ids(["chunk-0000", "chunk-0001"])
        assert idx.size == len(_CORPUS) - 2
        assert idx.get_by_id("chunk-0000") is None

    def test_remove_by_document_id_returns_removed_ids(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        removed = idx.remove_by_document_id("doc-1")
        assert len(removed) == len(_CORPUS)
        assert idx.size == 0
        assert idx.search("kubernetes", top_k=1) == []

    def test_remove_by_document_id_unknown_returns_empty(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        assert idx.remove_by_document_id("missing-doc") == []


class TestBM25DeferredRebuild:
    def test_deferred_rebuild_rebuilds_once_on_exit(self):
        idx = BM25Index()
        with patch.object(idx, "_rebuild", wraps=idx._rebuild) as mock_rebuild:
            with idx.deferred_rebuild():
                idx.add([_chunk("alpha text", idx=10)])
                idx.add([_chunk("beta text", idx=11)])
                assert mock_rebuild.call_count == 0
            assert mock_rebuild.call_count == 1

    def test_deferred_chunks_searchable_after_exit(self):
        idx = BM25Index()
        with idx.deferred_rebuild():
            idx.add(_CORPUS)
        assert idx.size == len(_CORPUS)
        results = idx.search("kubernetes", top_k=1)
        assert results
        assert results[0][0].id == "chunk-0001"

    def test_save_flushes_pending_rebuild(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        idx = BM25Index(index_path=path)
        with idx.deferred_rebuild():
            idx.add(_CORPUS)
            idx.save()
        loaded = BM25Index(index_path=path)
        loaded.load()
        assert loaded.size == len(_CORPUS)
        assert loaded.search("kubernetes", top_k=1)

    def test_rebuild_clears_index_when_all_chunks_removed(self):
        idx = BM25Index()
        idx.index(_CORPUS)
        idx.remove_by_ids([c.id for c in _CORPUS])
        assert idx.size == 0
        assert idx.search("kubernetes", top_k=1) == []

    def test_deferred_search_rebuilds_on_read(self):
        idx = BM25Index()
        idx.index(_CORPUS[:2])
        with idx.deferred_rebuild():
            idx.add([_chunk("unique new term xyzzy", idx=99)])
            results = idx.search("xyzzy", top_k=1)
            assert results
            assert results[0][0].id == "chunk-0099"
        assert idx.search("xyzzy", top_k=1)

    def test_concurrent_search_during_deferred_rebuild(self):
        import threading

        idx = BM25Index()
        idx.index(_CORPUS[:3])
        errors: list[Exception] = []
        started = threading.Barrier(5)

        def search_loop() -> None:
            try:
                started.wait(timeout=5)
                for _ in range(100):
                    idx.search("kubernetes", top_k=5)
            except Exception as exc:  # pragma: no cover - test assertion below
                errors.append(exc)

        search_threads = [threading.Thread(target=search_loop) for _ in range(4)]
        with idx.deferred_rebuild():
            for thread in search_threads:
                thread.start()
            started.wait(timeout=5)
            for extra in _CORPUS[3:]:
                idx.add([extra])
            for thread in search_threads:
                thread.join(timeout=10)
        assert not errors
        assert idx.search("vector databases", top_k=1)

    def test_search_sees_chunks_added_during_shared_batch_ingest(self):
        """Simulates API ingest_directory + concurrent chat retrieval on one index."""
        import threading

        idx = BM25Index()
        idx.index(_CORPUS[:3])
        new_chunk = _chunk("kubernetes deployment rollout strategy", idx=99)
        search_ready = threading.Event()
        added = threading.Event()
        found_during_batch = threading.Event()

        def batch_ingest() -> None:
            with idx.deferred_rebuild():
                search_ready.wait(timeout=5)
                idx.add([new_chunk])
                added.set()
                found_during_batch.wait(timeout=5)

        def concurrent_search() -> None:
            search_ready.set()
            added.wait(timeout=5)
            hits = idx.search("kubernetes deployment rollout", top_k=1)
            if hits and hits[0][0].id == new_chunk.id:
                found_during_batch.set()

        ingest_thread = threading.Thread(target=batch_ingest)
        search_thread = threading.Thread(target=concurrent_search)
        ingest_thread.start()
        search_thread.start()
        ingest_thread.join(timeout=5)
        search_thread.join(timeout=5)
        assert found_during_batch.is_set()
        results = idx.search("kubernetes deployment rollout", top_k=1)
        assert results[0][0].id == new_chunk.id

    def test_save_preserves_dirty_when_mutated_during_write(self, tmp_path: Path):
        """Concurrent mutations during save must not drop unsaved changes."""
        import threading
        import time

        path = tmp_path / "bm25.json"
        idx = BM25Index(index_path=path)
        idx.index(_CORPUS)
        extra = _chunk("concurrent mutation term zzz", idx=100)
        write_started = threading.Event()
        mutation_done = threading.Event()

        original_open = Path.open

        def slow_open(path_obj: Path, *args, **kwargs):
            handle = original_open(path_obj, *args, **kwargs)
            if path_obj == path.with_suffix(f"{path.suffix}.tmp"):
                write_started.set()
                mutation_done.wait(timeout=5)
                time.sleep(0.01)
            return handle

        def mutate_during_save() -> None:
            write_started.wait(timeout=5)
            idx.add([extra])
            mutation_done.set()

        with patch.object(Path, "open", slow_open):
            mutator = threading.Thread(target=mutate_during_save)
            mutator.start()
            idx.save()
            mutator.join(timeout=5)

        assert idx._dirty is True
        idx.save()
        loaded = BM25Index(index_path=path)
        loaded.load()
        assert loaded.get_by_id(extra.id) is not None
