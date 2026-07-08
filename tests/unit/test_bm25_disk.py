"""T-165 — Disk-backed BM25 index tests (100% module coverage target)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.infrastructure.vectordb import bm25_disk as bm25_disk_mod
from src.infrastructure.vectordb.bm25 import BM25Index
from src.infrastructure.vectordb.bm25_disk import DiskBM25Index

# Private helpers exercised for branch coverage (T-165).
# noinspection PyProtectedMember
_atomic_write_json = bm25_disk_mod._atomic_write_json
# noinspection PyProtectedMember
_bm25_idf = bm25_disk_mod._bm25_idf
# noinspection PyProtectedMember
_read_json = bm25_disk_mod._read_json
# noinspection PyProtectedMember
_score_term = bm25_disk_mod._score_term
# noinspection PyProtectedMember
_SegmentWriter = bm25_disk_mod._SegmentWriter
# noinspection PyProtectedMember
_tokenize = bm25_disk_mod._tokenize


def _chunk(text: str, idx: int = 0, doc_id: str = "doc-1") -> Chunk:
    return Chunk(id=f"chunk-{idx:05d}", document_id=doc_id, text=text)


_CORPUS = [
    _chunk("The quick brown fox jumps over the lazy dog", idx=0),
    _chunk("Kubernetes pod scheduling and node affinity rules", idx=1),
    _chunk("IAM roles and policies for AWS EKS clusters", idx=2),
    _chunk("Vector databases store embeddings for similarity search", idx=3),
    _chunk("Python async programming with asyncio", idx=4),
]


@pytest.fixture
def disk_index(tmp_path: Path) -> DiskBM25Index:
    return DiskBM25Index(tmp_path / "bm25_disk", segment_size=2)


# ── helpers ────────────────────────────────────────────────────────────────────


class TestDiskHelpers:
    def test_tokenize_lowercases(self):
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_bm25_idf_empty(self):
        assert _bm25_idf(0, {"a": 1}) == {}
        assert _bm25_idf(10, {}) == {}

    def test_bm25_idf_negative_floor(self):
        # Term in most docs → negative raw IDF → epsilon floor.
        df = {"common": 8, "rare": 1}
        idf = _bm25_idf(10, df)
        assert idf["common"] > 0
        assert idf["rare"] > idf["common"]

    def test_score_term_zero_cases(self):
        assert _score_term(0, 10, 10, 1.0) == 0.0
        assert _score_term(1, 10, 10, 0.0) == 0.0
        assert _score_term(1, 10, 0.0, 1.0) == 0.0

    def test_atomic_write_os_error(self, tmp_path: Path):
        path = tmp_path / "out.json"
        with (
            patch("pathlib.Path.open", side_effect=OSError("disk full")),
            pytest.raises(VectorStoreError, match="Cannot write"),
        ):
            _atomic_write_json(path, {"a": 1})

    def test_read_json_missing(self, tmp_path: Path):
        with pytest.raises(VectorStoreError, match="missing file"):
            _read_json(tmp_path / "missing.json")

    def test_read_json_corrupt(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not-json", encoding="utf-8")
        with pytest.raises(VectorStoreError, match="Cannot read"):
            _read_json(path)


# ── DiskBM25Index core ─────────────────────────────────────────────────────────


class TestDiskBM25Index:
    def test_segment_size_must_be_positive(self, tmp_path: Path):
        with pytest.raises(ValueError, match="segment_size"):
            DiskBM25Index(tmp_path / "x", segment_size=0)

    def test_empty_search(self, disk_index: DiskBM25Index):
        assert disk_index.search("kubernetes", top_k=5) == []

    def test_empty_query(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        assert disk_index.search("", top_k=5) == []

    def test_size_and_search(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        assert disk_index.size == len(_CORPUS)
        results = disk_index.search("kubernetes pod scheduling", top_k=3)
        assert results
        assert results[0][0].id == "chunk-00001"
        assert results[0][1] > 0

    def test_scores_match_memory_backend(self, tmp_path: Path):
        mem = BM25Index()
        mem.index(_CORPUS)
        disk = DiskBM25Index(tmp_path / "disk", segment_size=2)
        disk.index(_CORPUS)
        query = "kubernetes pod scheduling"
        mem_hits = mem.search(query, top_k=5)
        disk_hits = disk.search(query, top_k=5)
        assert [c.id for c, _ in mem_hits] == [c.id for c, _ in disk_hits]
        for (_, ms), (_, ds) in zip(mem_hits, disk_hits, strict=True):
            assert ms == pytest.approx(ds, rel=1e-6, abs=1e-9)

    def test_case_insensitive(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        lower = disk_index.search("kubernetes", top_k=1)
        upper = disk_index.search("KUBERNETES", top_k=1)
        assert lower and upper
        assert lower[0][0].id == upper[0][0].id

    def test_document_filter(self, disk_index: DiskBM25Index):
        # Need ≥3 docs so unique query terms have non-zero Okapi IDF.
        scoped = [
            _chunk("kubernetes pod scheduling", idx=0, doc_id="doc-a"),
            _chunk("vector databases store embeddings", idx=1, doc_id="doc-b"),
            _chunk("python async programming", idx=2, doc_id="doc-c"),
        ]
        disk_index.index(scoped)
        filt = RetrievalFilter(document_ids=["doc-a"])
        results = disk_index.search("kubernetes", top_k=5, filters=filt)
        assert len(results) == 1
        assert results[0][0].document_id == "doc-a"

    def test_add_and_dedup(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS[:2])
        disk_index.add(_CORPUS[2:])
        assert disk_index.size == len(_CORPUS)
        disk_index.add(_CORPUS[:2])
        assert disk_index.size == len(_CORPUS)

    def test_add_new_chunk_searchable(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS[:3])
        extra = _chunk("LangGraph agentic workflow orchestration", idx=99)
        disk_index.add([extra])
        results = disk_index.search("langgraph agentic", top_k=1)
        assert results[0][0].id == extra.id

    def test_remove_by_ids(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        disk_index.remove_by_ids([])
        assert disk_index.size == len(_CORPUS)
        disk_index.remove_by_ids(["chunk-00000", "chunk-00001"])
        assert disk_index.size == len(_CORPUS) - 2
        assert disk_index.get_by_id("chunk-00000") is None
        assert disk_index.search("quick brown fox", top_k=1) == []

    def test_remove_by_document_id(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        removed = disk_index.remove_by_document_id("doc-1")
        assert len(removed) == len(_CORPUS)
        assert disk_index.size == 0
        assert disk_index.remove_by_document_id("missing") == []

    def test_get_by_id_and_chunks_snapshot(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        assert disk_index.get_by_id("missing") is None
        found = disk_index.get_by_id("chunk-00001")
        assert found is not None and found.id == "chunk-00001"
        snapshot = disk_index.chunks
        assert len(snapshot) == len(_CORPUS)
        snapshot.clear()
        assert disk_index.size == len(_CORPUS)

    def test_iter_chunks_streams_from_disk_segments(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        streamed = list(disk_index.iter_chunks())
        assert [c.id for c in streamed] == [c.id for c in _CORPUS]

    def test_iter_chunks_yields_pending_without_flush(self, disk_index: DiskBM25Index):
        with disk_index.deferred_rebuild():
            disk_index.add([_chunk("pending only chunk", idx=99)])
            streamed = list(disk_index.iter_chunks())
        assert len(streamed) == 1
        assert streamed[0].id == "chunk-00099"

    def test_update_chunk_metadata(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        assert disk_index.update_chunk_metadata("chunk-00001", {}) is False
        assert disk_index.update_chunk_metadata("missing", {"k": 1}) is False
        assert disk_index.update_chunk_metadata("chunk-00001", {"feedback_score": 2.0}) is True
        updated = disk_index.get_by_id("chunk-00001")
        assert updated is not None
        assert updated.metadata["feedback_score"] == 2.0

    def test_index_replaces(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        disk_index.index([_chunk("completely different content", idx=99)])
        assert disk_index.size == 1
        assert disk_index.search("kubernetes", top_k=5) == []


# ── persistence ────────────────────────────────────────────────────────────────


class TestDiskBM25Persistence:
    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "disk"
        idx = DiskBM25Index(path, segment_size=2)
        idx.index(_CORPUS)
        idx.save()
        assert (path / "manifest.json").exists()

        loaded = DiskBM25Index(path, segment_size=2)
        loaded.load()
        assert loaded.size == len(_CORPUS)
        results = loaded.search("kubernetes pod scheduling", top_k=1)
        assert results[0][0].id == "chunk-00001"

    def test_save_skips_when_clean(self, tmp_path: Path):
        path = tmp_path / "disk"
        idx = DiskBM25Index(path, segment_size=2)
        idx.index(_CORPUS)
        idx.save()
        mtime = (path / "manifest.json").stat().st_mtime
        idx.save()
        assert (path / "manifest.json").stat().st_mtime == mtime

    def test_load_or_create(self, tmp_path: Path):
        empty = DiskBM25Index.load_or_create(tmp_path / "missing")
        assert empty.size == 0
        path = tmp_path / "exists"
        saved = DiskBM25Index(path, segment_size=2)
        saved.index(_CORPUS)
        saved.save()
        loaded = DiskBM25Index.load_or_create(path, segment_size=2)
        assert loaded.size == len(_CORPUS)

    def test_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(VectorStoreError, match="not found"):
            DiskBM25Index(tmp_path / "missing").load()

    def test_load_bad_version_raises(self, tmp_path: Path):
        path = tmp_path / "disk"
        path.mkdir()
        (path / "manifest.json").write_text('{"version": 99}', encoding="utf-8")
        with pytest.raises(VectorStoreError, match="Unsupported"):
            DiskBM25Index(path).load()

    def test_bootstrap_from_disk(self, tmp_path: Path):
        path = tmp_path / "disk"
        idx = DiskBM25Index(path, segment_size=2)
        idx.index(_CORPUS)
        idx.save()
        other = DiskBM25Index(path, segment_size=2)
        other.bootstrap_from_disk()
        assert other.size == len(_CORPUS)
        bare = DiskBM25Index(tmp_path / "empty", segment_size=2)
        bare.bootstrap_from_disk()
        assert bare.size == 0

    def test_save_preserves_dirty_on_concurrent_mutation(self, tmp_path: Path):
        import threading
        import time

        path = tmp_path / "disk"
        idx = DiskBM25Index(path, segment_size=2)
        idx.index(_CORPUS)
        extra = _chunk("concurrent mutation term zzz", idx=100)
        write_started = threading.Event()
        mutation_done = threading.Event()

        original_dump = json.dump

        def slow_dump(obj: object, fp: object, *args: object, **kwargs: object) -> None:
            if isinstance(obj, dict) and "version" in obj:
                write_started.set()
                mutation_done.wait(timeout=5)
                time.sleep(0.01)
            original_dump(obj, fp, *args, **kwargs)  # type: ignore[arg-type]

        def mutate() -> None:
            write_started.wait(timeout=5)
            idx.add([extra])
            mutation_done.set()

        with patch("src.infrastructure.vectordb.bm25_disk.json.dump", side_effect=slow_dump):
            mutator = threading.Thread(target=mutate)
            mutator.start()
            idx.save()
            mutator.join(timeout=5)

        assert idx._dirty is True
        idx.save()
        loaded = DiskBM25Index.load_or_create(path, segment_size=2)
        assert loaded.get_by_id(extra.id) is not None

    def test_flush_empty_after_removes(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        disk_index.remove_by_ids([c.id for c in _CORPUS])
        assert disk_index.size == 0
        disk_index.save()
        assert disk_index.search("kubernetes", top_k=1) == []


# ── deferred rebuild ───────────────────────────────────────────────────────────


class TestDiskDeferredRebuild:
    def test_deferred_rebuild_once(self, disk_index: DiskBM25Index):
        with patch.object(disk_index, "_flush_to_disk", wraps=disk_index._flush_to_disk) as mock:
            with disk_index.deferred_rebuild():
                disk_index.add([_chunk("alpha text", idx=10)])
                disk_index.add([_chunk("beta text", idx=11)])
                assert mock.call_count == 0
            assert mock.call_count == 1

    def test_deferred_searchable(self, disk_index: DiskBM25Index):
        with disk_index.deferred_rebuild():
            disk_index.add(_CORPUS)
            results = disk_index.search("kubernetes", top_k=1)
            assert results
            assert results[0][0].id == "chunk-00001"
        assert disk_index.size == len(_CORPUS)

    def test_rebuild_method(self, disk_index: DiskBM25Index):
        with disk_index.deferred_rebuild():
            disk_index.add(_CORPUS)
        disk_index.rebuild()
        assert disk_index.size == len(_CORPUS)


# ── errors / edge paths ────────────────────────────────────────────────────────


class TestDiskBM25Errors:
    def test_segment_writer_flush_os_error(self, tmp_path: Path):
        writer = _SegmentWriter(0, tmp_path / "seg")
        writer.add(_CORPUS[0])
        with (
            patch("pathlib.Path.open", side_effect=OSError("fail")),
            pytest.raises(VectorStoreError, match="Cannot flush"),
        ):
            writer.flush()

    def test_load_chunk_corrupt(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=2)
        idx.index(_CORPUS[:1])
        seg = idx._segment_dir(0)
        (seg / "chunks.jsonl").write_text("{bad", encoding="utf-8")
        with pytest.raises(VectorStoreError, match="Cannot load chunk"):
            idx._load_chunk(0, 0)

    def test_load_chunk_missing_line(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=2)
        idx.index(_CORPUS[:1])
        assert idx._load_chunk(0, 99) is None

    def test_mmap_lengths_missing(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=2)
        idx.index(_CORPUS[:1])
        (idx._segment_dir(0) / "lengths.npy").unlink()
        with pytest.raises(VectorStoreError, match="Cannot mmap"):
            idx._mmap_lengths(0)

    def test_invalid_postings(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=2)
        idx.index(_CORPUS[:1])
        (idx._segment_dir(0) / "postings.json").write_text("[]", encoding="utf-8")
        with pytest.raises(VectorStoreError, match="Invalid postings"):
            idx._load_postings(0)

    def test_search_skips_out_of_range_posting(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=10)
        idx.index(_CORPUS)
        postings = idx._load_postings(0)
        term = next(iter(postings))
        postings[term].append([9999, 1])
        (idx._segment_dir(0) / "postings.json").write_text(json.dumps(postings), encoding="utf-8")
        results = idx.search("kubernetes", top_k=5)
        assert isinstance(results, list)

    def test_default_path_uses_constant(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from src.infrastructure.vectordb import bm25_disk as mod

        monkeypatch.setattr(mod, "BM25_DISK_PATH", tmp_path / "default_disk")
        idx = DiskBM25Index(segment_size=2)
        idx.index([_CORPUS[0]])
        assert idx.path == tmp_path / "default_disk"

    def test_search_reloads_segment_ids_when_cache_missing(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=10)
        idx.index(_CORPUS)
        idx._segment_chunk_ids.clear()
        results = idx.search("kubernetes", top_k=1)
        assert results

    def test_update_metadata_on_pending_only_chunk(self, disk_index: DiskBM25Index):
        with disk_index.deferred_rebuild():
            disk_index.add([_chunk("pending only text", idx=7)])
            assert disk_index.update_chunk_metadata("chunk-00007", {"k": "v"}) is True
            found = disk_index.get_by_id("chunk-00007")
            assert found is not None
            assert found.metadata["k"] == "v"

    def test_save_materialize_error_cleans_staging(self, tmp_path: Path):
        path = tmp_path / "disk"
        idx = DiskBM25Index(path, segment_size=2)
        idx.index(_CORPUS)
        with (
            patch.object(
                DiskBM25Index,
                "_materialize_segments",
                side_effect=OSError("boom"),
            ),
            pytest.raises(OSError, match="boom"),
        ):
            idx.save()
        assert not list(tmp_path.glob("*.staging-*"))

    def test_memory_resident_bytes_positive(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        assert disk_index.memory_resident_bytes() > 0

    def test_soft_deleted_get_returns_none_while_deferred(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        with disk_index.deferred_rebuild():
            disk_index.remove_by_ids(["chunk-00000"])
            assert disk_index.get_by_id("chunk-00000") is None

    def test_ensure_searchable_flushes_pending_outside_defer(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        disk_index._pending_chunks["chunk-00999"] = _chunk("brand new injectable term", idx=999)
        disk_index._needs_rebuild = True
        assert disk_index.search("injectable", top_k=1)

    def test_search_skips_negative_posting_offset(self, tmp_path: Path):
        idx = DiskBM25Index(tmp_path / "disk", segment_size=10)
        idx.index(_CORPUS)
        postings = idx._load_postings(0)
        assert "kubernetes" in postings
        postings["kubernetes"].append([-1, 1])
        (idx._segment_dir(0) / "postings.json").write_text(json.dumps(postings), encoding="utf-8")
        assert isinstance(idx.search("kubernetes", top_k=5), list)

    def test_deferred_search_skips_deleted_and_replaced(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        with disk_index.deferred_rebuild():
            disk_index.remove_by_ids(["chunk-00001"])
            disk_index.add([_chunk("kubernetes brand new replacement text", idx=1)])
            # live disk posting for old chunk-00001 is skipped; pending is scored
            hits = disk_index.search("kubernetes", top_k=5)
            assert hits
            assert all(c.id != "chunk-00001" or "replacement" in c.text for c, _ in hits)

    def test_deferred_remove_adjusts_pending_corpus_stats(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        with disk_index.deferred_rebuild():
            disk_index.remove_by_ids(["chunk-00000"])
            disk_index.add([_chunk("unique pending term zzqqww", idx=50)])
            hits = disk_index.search("zzqqww", top_k=1)
            assert hits and hits[0][0].id == "chunk-00050"

    def test_deferred_soft_view_matches_memory_scores(self, tmp_path: Path):
        """Live ingest soft-view must match in-memory Okapi during deferred_rebuild."""
        mem = BM25Index()
        disk = DiskBM25Index(tmp_path / "disk", segment_size=2)
        mem.index(_CORPUS)
        disk.index(_CORPUS)
        replacement = _chunk("kubernetes brand new replacement text", idx=1)
        added = _chunk("brand new injectable zzqqww term", idx=99)
        with mem.deferred_rebuild(), disk.deferred_rebuild():
            mem.remove_by_ids(["chunk-00000"])
            disk.remove_by_ids(["chunk-00000"])
            mem.add([replacement, added])
            disk.add([replacement, added])
            for query in (
                "kubernetes",
                "zzqqww",
                "replacement",
                "vector",
                "quick brown fox",
                "python async",
            ):
                mem_hits = mem.search(query, top_k=5)
                disk_hits = disk.search(query, top_k=5)
                assert [c.id for c, _ in mem_hits] == [c.id for c, _ in disk_hits], query
                for (_, ms), (_, ds) in zip(mem_hits, disk_hits, strict=True):
                    assert ms == pytest.approx(ds, rel=1e-6, abs=1e-9), query

    def test_deferred_soft_view_stats_cache_hit(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        with disk_index.deferred_rebuild():
            disk_index.add([_chunk("cacheable pending term", idx=80)])
            first = disk_index._soft_view_corpus_stats()
            second = disk_index._soft_view_corpus_stats()
            assert first is second
            disk_index.search("cacheable", top_k=1)
            third = disk_index._soft_view_corpus_stats()
            assert third is first

    def test_soft_view_skips_missing_deleted_chunk(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS[:2])
        with disk_index.deferred_rebuild():
            disk_index.remove_by_ids(["chunk-00000"])
            with patch.object(disk_index, "_load_chunk", return_value=None):
                hits = disk_index.search("kubernetes", top_k=5)
            assert isinstance(hits, list)

    def test_soft_view_ignores_deleted_id_absent_from_map(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS[:2])
        with disk_index.deferred_rebuild():
            disk_index._deleted_ids.add("never-indexed")
            disk_index._soft_view_stats_cache = None
            df, size, total_dl = disk_index._soft_view_corpus_stats()
            assert size == disk_index._corpus_size
            assert total_dl == disk_index._total_dl
            assert df == disk_index._df

    def test_search_unlocked_empty_after_deferred_full_delete(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        with disk_index.deferred_rebuild():
            disk_index.remove_by_ids([c.id for c in _CORPUS])
            assert disk_index.search("kubernetes", top_k=5) == []
            assert disk_index._search_unlocked(["kubernetes"], 5, filters=None) == []
            df, size, total_dl = disk_index._soft_view_corpus_stats()
            assert size == 0
            assert total_dl == 0
            assert df == {}

    def test_search_skips_missing_chunk_and_filtered(self, disk_index: DiskBM25Index):
        disk_index.index(_CORPUS)
        ranked_ids: list[str] = []

        original = disk_index._get_by_id_unlocked

        def flaky_get(chunk_id: str):
            ranked_ids.append(chunk_id)
            if len(ranked_ids) == 1:
                return None
            return original(chunk_id)

        with patch.object(disk_index, "_get_by_id_unlocked", side_effect=flaky_get):
            hits = disk_index.search("kubernetes", top_k=5)
            assert isinstance(hits, list)

        filt = RetrievalFilter(document_ids=["no-such-doc"])
        assert disk_index.search("kubernetes", top_k=5, filters=filt) == []

    def test_ensure_built_flushes(self, disk_index: DiskBM25Index):
        with disk_index.deferred_rebuild():
            disk_index.add(_CORPUS)
            # leave dirty pending; explicit rebuild
        disk_index._needs_rebuild = True
        disk_index._pending_chunks["chunk-00000"] = _CORPUS[0]
        disk_index.rebuild()
        assert disk_index.size == len(_CORPUS)


# ── factory (settings.backend) ─────────────────────────────────────────────────


class TestBM25Factory:
    def test_default_backend_is_memory(self, tmp_path: Path):
        idx = BM25Index.load_or_create(tmp_path / "mem.json", backend="memory")
        assert isinstance(idx, BM25Index)

    def test_disk_backend_via_factory(self, tmp_path: Path):
        idx = BM25Index.load_or_create(tmp_path / "disk", backend="disk")
        assert isinstance(idx, DiskBM25Index)
        idx.index(_CORPUS)
        assert idx.search("kubernetes", top_k=1)

    def test_disk_backend_from_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RETRIEVAL__BM25__BACKEND", "disk")
        monkeypatch.setenv("RETRIEVAL__BM25__DISK_PATH", str(tmp_path / "from_settings"))
        monkeypatch.setenv("RETRIEVAL__BM25__SEGMENT_SIZE", "3")
        from src.core.settings import Settings

        s = Settings()
        assert s.retrieval.bm25.backend == "disk"
        assert s.retrieval.bm25.segment_size == 3

        with patch("src.core.settings.settings", s):
            idx = BM25Index.load_or_create()
            assert isinstance(idx, DiskBM25Index)
            assert idx.path == tmp_path / "from_settings"
            assert idx._segment_size == 3

    def test_explicit_json_path_stays_memory_when_settings_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        json_path = tmp_path / "bm25_index.json"
        saved = BM25Index(index_path=json_path)
        saved.index(_CORPUS)
        saved.save()
        assert json_path.is_file()

        monkeypatch.setenv("RETRIEVAL__BM25__BACKEND", "disk")
        monkeypatch.setenv("RETRIEVAL__BM25__DISK_PATH", str(tmp_path / "disk_root"))
        from src.core.settings import Settings

        s = Settings()
        with patch("src.core.settings.settings", s):
            idx = BM25Index.load_or_create(json_path)
            assert isinstance(idx, BM25Index)
            assert idx.size == len(_CORPUS)
            assert json_path.is_file()
            assert not json_path.is_dir()

            from src.rag.retrieval.bm25_retriever import BM25Retriever

            retriever = BM25Retriever.from_disk(json_path)
            assert isinstance(retriever.bm25_index, BM25Index)
            assert retriever.size == len(_CORPUS)

    def test_explicit_path_with_backend_disk_still_uses_disk(self, tmp_path: Path):
        path = tmp_path / "explicit_disk"
        idx = BM25Index.load_or_create(path, backend="disk")
        assert isinstance(idx, DiskBM25Index)
        assert idx.path == path


# ── scale / memory bounded ────────────────────────────────────────────────────


class TestDiskBM25Scale:
    def test_indexes_and_searches_100k_chunks(self, tmp_path: Path):
        n = 100_000
        path = tmp_path / "large"
        idx = DiskBM25Index(path, segment_size=5_000)

        batch: list[Chunk] = []
        with idx.deferred_rebuild():
            for i in range(n):
                text = f"token{i % 97} document row {i} content payload"
                if i == 42_000:
                    text = "unique needle xyzzy scale fixture marker " + text
                batch.append(_chunk(text, idx=i))
                if len(batch) >= 5_000:
                    idx.add(batch)
                    batch = []
            if batch:
                idx.add(batch)

        assert idx.size == n
        hits = idx.search("unique needle xyzzy", top_k=3)
        assert hits
        assert hits[0][0].id == "chunk-42000"

        idx.remove_by_ids(["chunk-42000"])
        assert idx.get_by_id("chunk-42000") is None
        replacement = _chunk("unique needle xyzzy restored after remove", idx=42_000)
        idx.add([replacement])
        hits2 = idx.search("unique needle xyzzy restored", top_k=1)
        assert hits2[0][0].id == "chunk-42000"

        idx.save()
        reloaded = DiskBM25Index.load_or_create(path, segment_size=5_000)
        assert reloaded.size == n
        assert reloaded.search("unique needle xyzzy", top_k=1)

    def test_memory_stays_bounded_as_corpus_grows(self, tmp_path: Path):
        def build(n: int) -> int:
            idx = DiskBM25Index(tmp_path / f"m{n}", segment_size=1_000)
            chunks = [_chunk(f"word{i} content shared", idx=i) for i in range(n)]
            idx.index(chunks)
            idx.search("word0", top_k=1)
            return idx.memory_resident_bytes()

        small = build(2_000)
        large = build(8_000)
        assert large < small * 6
        assert large < 50 * 1024 * 1024
