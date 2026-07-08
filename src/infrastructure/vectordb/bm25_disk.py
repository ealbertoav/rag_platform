"""T-165 — Disk-backed / segmented BM25 index for large corpora.

Stores chunk payloads and inverted postings on disk in fixed-size segments.
Search memory stays bounded to roughly one segment's postings plus the global
IDF table and the id map — independent of loading the full Okapi model.

Scoring matches "rank_bm25.BM25Okapi" (k1=1.5, b=0.75, epsilon=0.25).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import threading
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np

from src.core.constants import BM25_DISK_PATH
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.rag.retrieval.filters import chunk_matches_filter

logger = logging.getLogger(__name__)

_INDEX_FORMAT_VERSION = 1
_K1 = 1.5
_B = 0.75
_EPSILON = 0.25
_MANIFEST_NAME = "manifest.json"
_IDF_NAME = "idf.json"
_DF_NAME = "df.json"
_ID_MAP_NAME = "id_map.json"
_SEGMENTS_DIR = "segments"


def _tokenize(text: str) -> list[str]:
    """Lowercase word-split tokenizer used for both indexing and querying."""
    return text.lower().split()


def _term_freqs(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for token in tokens:
        counts[token] += 1
    return dict(counts)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise VectorStoreError(f"Cannot write BM25 disk file {path}", cause=exc) from exc


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise VectorStoreError(f"BM25 disk index missing file {path}", cause=exc) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorStoreError(f"Cannot read BM25 disk file {path}", cause=exc) from exc


def _bm25_idf(corpus_size: int, df: dict[str, int]) -> dict[str, float]:
    """Okapi IDF with an epsilon floor for negative values (matches BM25Okapi)."""
    if corpus_size <= 0 or not df:
        return {}
    idf: dict[str, float] = {}
    idf_sum = 0.0
    negative: list[str] = []
    for term, freq in df.items():
        value = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
        idf[term] = value
        idf_sum += value
        if value < 0:
            negative.append(term)
    average_idf = idf_sum / len(idf)
    floor = _EPSILON * average_idf
    for term in negative:
        idf[term] = floor
    return idf


def _score_term(tf: float, dl: float, avgdl: float, idf: float) -> float:
    if idf == 0.0 or tf <= 0.0 or avgdl <= 0.0:
        return 0.0
    denom = tf + _K1 * (1.0 - _B + _B * dl / avgdl)
    return idf * (tf * (_K1 + 1.0) / denom)


class _SegmentWriter:
    """Accumulate chunks for one on-disk segment, then flush atomically."""

    def __init__(self, segment_id: int, directory: Path) -> None:
        self.segment_id = segment_id
        self.directory = directory
        self.chunks: list[Chunk] = []
        self.lengths: list[int] = []
        self.postings: dict[str, list[list[int]]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunk: Chunk) -> None:
        tokens = _tokenize(chunk.text)
        local_idx = len(self.chunks)
        self.chunks.append(chunk)
        self.lengths.append(len(tokens))
        for term, tf in _term_freqs(tokens).items():
            self.postings[term].append([local_idx, tf])

    def flush(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        chunks_path = self.directory / "chunks.jsonl"
        lengths_path = self.directory / "lengths.npy"
        postings_path = self.directory / "postings.json"
        ids_path = self.directory / "ids.json"
        tmp_chunks = chunks_path.with_suffix(".jsonl.tmp")
        try:
            with tmp_chunks.open("w", encoding="utf-8") as fh:
                for chunk in self.chunks:
                    fh.write(chunk.model_dump_json())
                    fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            tmp_chunks.replace(chunks_path)
            np.save(lengths_path, np.asarray(self.lengths, dtype=np.int32))
            _atomic_write_json(postings_path, dict(self.postings))
            _atomic_write_json(ids_path, [c.id for c in self.chunks])
            _atomic_write_json(
                self.directory / "meta.json",
                {"segment_id": self.segment_id, "size": len(self.chunks)},
            )
        except OSError as exc:
            tmp_chunks.unlink(missing_ok=True)
            raise VectorStoreError(
                f"Cannot flush BM25 disk segment {self.directory}",
                cause=exc,
            ) from exc


class DiskBM25Index:
    """Segmented, memmapped BM25 index with the same public surface as "BM25Index".

    In-memory state is limited to document-frequency / IDF tables, the
    "chunk_id → (segment_id, local_idx)" map, and an optional write buffer
    while inside: meth:`deferred_rebuild`. Chunk payloads and postings live
    on disk under "index_path/".
    """

    def __init__(
        self,
        index_path: Path | None = None,
        *,
        segment_size: int = 10_000,
    ) -> None:
        if segment_size < 1:
            raise ValueError("segment_size must be >= 1")
        self._path: Path = Path(index_path) if index_path is not None else BM25_DISK_PATH
        self._segment_size = segment_size
        self._lock = threading.RLock()
        self._dirty = False
        self._mutation_generation = 0
        self._defer_rebuild_depth = 0
        self._needs_rebuild = False

        self._df: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._corpus_size = 0
        self._total_dl = 0
        self._id_map: dict[str, tuple[int, int]] = {}
        self._segment_ids: list[int] = []
        self._segment_chunk_ids: dict[int, list[str]] = {}
        # Pending mutations held until flush (bounded by ingest batch size).
        self._pending_chunks: dict[str, Chunk] = {}
        self._deleted_ids: set[str] = set()
        # Lazily computed Okapi (df, size, total_dl) for deferred soft-view search.
        self._soft_view_stats_cache: tuple[dict[str, int], int, int] | None = None

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """Replace the entire index with *chunks*."""
        with self._lock:
            self._clear_state()
            self._pending_chunks = {c.id: c for c in chunks}
            self._deleted_ids.clear()
            self._schedule_rebuild()
            self._mark_dirty()
        logger.debug("Disk BM25 index replaced: %d chunks", len(chunks))

    def add(self, chunks: list[Chunk]) -> None:
        """Append *chunks*; same-id rows replace existing ones."""
        with self._lock:
            for chunk in chunks:
                self._deleted_ids.discard(chunk.id)
                if chunk.id in self._id_map:
                    self._deleted_ids.add(chunk.id)
                self._pending_chunks[chunk.id] = chunk
            self._schedule_rebuild()
            self._mark_dirty()
        logger.debug(
            "Disk BM25 index updated: +%d pending, live size %d",
            len(chunks),
            self._read_size_unlocked(),
        )

    def remove_by_ids(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        if not chunk_ids:
            return
        with self._lock:
            for chunk_id in chunk_ids:
                self._pending_chunks.pop(chunk_id, None)
                if chunk_id in self._id_map:
                    self._deleted_ids.add(chunk_id)
            self._schedule_rebuild()
            self._mark_dirty()

    def rebuild(self) -> None:
        """Materialize pending mutations into on-disk segments."""
        with self._lock:
            self._flush_to_disk()
            self._needs_rebuild = False

    @contextmanager
    def deferred_rebuild(self) -> Generator[DiskBM25Index, None, None]:
        """Defer segment rebuilds until the context exits."""
        with self._lock:
            self._defer_rebuild_depth += 1
        try:
            yield self
        finally:
            with self._lock:
                self._defer_rebuild_depth -= 1
                if self._defer_rebuild_depth == 0:
                    self._ensure_built()

    def remove_by_document_id(self, document_id: str) -> list[str]:
        """Remove all chunks belonging to *document_id*. Returns removed chunk IDs."""
        with self._lock:
            removed = [
                cid
                for cid, chunk in self._iter_live_chunks_unlocked()
                if chunk.document_id == document_id
            ]
        if removed:
            self.remove_by_ids(removed)
        return removed

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: RetrievalFilter | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return up to *top_k* (chunk, score) pairs sorted by BM25 score desc."""
        with self._lock:
            self._ensure_searchable()
            if self._read_size_unlocked() == 0:
                return []
            tokens = _tokenize(query)
            if not tokens:
                return []
            return self._search_unlocked(tokens, top_k, filters=filters)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Flush pending mutations and write manifest/IDF metadata.

        Chunk snapshot is taken under the lock; segment I/O runs outside so
        concurrent mutations can land.  "_dirty" clears only when the
        generation is unchanged after the writing (same contract as:
        class:`~src.infrastructure.vectordb.bm25.BM25Index`).
        """
        with self._lock:
            if not self._dirty:
                logger.debug("Disk BM25 index unchanged — skipping save")
                return
            generation_at_snapshot = self._mutation_generation
            live_chunks = [chunk for _, chunk in self._iter_live_chunks_unlocked()]
            segment_size = self._segment_size
            path = self._path

        staging = path.with_name(f"{path.name}.staging-{os.getpid()}")
        try:
            meta = self._materialize_segments(staging, live_chunks, segment_size)
        except Exception:
            with suppress(OSError):
                shutil.rmtree(staging)
            raise

        with self._lock:
            if self._mutation_generation != generation_at_snapshot:
                with suppress(OSError):
                    shutil.rmtree(staging)
                # Keep dirty; next save will rewrite with the newer mutations.
                return
            if path.exists():
                with suppress(OSError):
                    shutil.rmtree(path)
            staging.rename(path)
            self._apply_materialized_state(meta)
            self._pending_chunks.clear()
            self._deleted_ids.clear()
            self._needs_rebuild = False
            self._dirty = False
        logger.info("Disk BM25 index saved to %s", self._path)

    def load(self) -> None:
        """Load an existing on-disk index from "self._path"."""
        manifest_path = self._path / _MANIFEST_NAME
        if not manifest_path.exists():
            raise VectorStoreError(f"BM25 disk index not found at {self._path}")
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("version") != _INDEX_FORMAT_VERSION:
            raise VectorStoreError(f"Unsupported BM25 disk index format at {self._path}")

        with self._lock:
            self._df = {str(k): int(v) for k, v in _read_json(self._path / _DF_NAME).items()}
            self._idf = {str(k): float(v) for k, v in _read_json(self._path / _IDF_NAME).items()}
            raw_id_map = _read_json(self._path / _ID_MAP_NAME)
            self._id_map = {
                str(cid): (int(pair[0]), int(pair[1])) for cid, pair in raw_id_map.items()
            }
            self._segment_ids = [int(x) for x in manifest.get("segment_ids", [])]
            self._corpus_size = int(manifest.get("corpus_size", 0))
            self._total_dl = int(manifest.get("total_dl", 0))
            self._segment_size = int(manifest.get("segment_size", self._segment_size))
            self._segment_chunk_ids = {
                seg_id: [str(x) for x in _read_json(self._segment_dir(seg_id) / "ids.json")]
                for seg_id in self._segment_ids
            }
            self._pending_chunks.clear()
            self._deleted_ids.clear()
            self._soft_view_stats_cache = None
            self._dirty = False
            self._needs_rebuild = False
        logger.info(
            "Disk BM25 index loaded from %s (%d chunks, %d segments)",
            self._path,
            self._corpus_size,
            len(self._segment_ids),
        )

    @classmethod
    def load_or_create(
        cls,
        index_path: Path | None = None,
        *,
        segment_size: int = 10_000,
    ) -> DiskBM25Index:
        """Return a loaded index if present, otherwise an empty one."""
        instance = cls(index_path, segment_size=segment_size)
        root = instance.path
        if (root / _MANIFEST_NAME).exists():
            instance.load()
        else:
            root.mkdir(parents=True, exist_ok=True)
        return instance

    def bootstrap_from_disk(self, _legacy_path: Path | None = None) -> None:
        """Compatibility shim — disk backend has no pickle migration path."""
        if (self.path / _MANIFEST_NAME).exists():
            self.load()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Root directory for the on-disk index."""
        return self._path

    @property
    def size(self) -> int:
        with self._lock:
            return self._read_size_unlocked()

    def iter_chunks(self) -> Generator[Chunk, None, None]:
        """Yield live chunks one at a time without materializing the full corpus.

        Each chunk is resolved under the index lock, so concurrent flushes cannot
        read segment files after the on-disk tree is replaced. If a flush lands
        mid-iteration, already-yielded IDs are skipped and the scan restarts.
        """
        yielded: set[str] = set()
        while True:
            restart = False
            with self._lock:
                generation = self._mutation_generation
                chunk_ids = list(self._iter_live_chunk_ids_unlocked())

            for chunk_id in chunk_ids:
                if chunk_id in yielded:
                    continue
                with self._lock:
                    if self._mutation_generation != generation:
                        restart = True
                        break
                    try:
                        chunk = self._get_by_id_unlocked(chunk_id)
                    except VectorStoreError:
                        restart = True
                        break
                if chunk is not None:
                    yielded.add(chunk_id)
                    yield chunk

            if not restart:
                return

    @property
    def chunks(self) -> list[Chunk]:
        return list(self.iter_chunks())

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            return self._get_by_id_unlocked(chunk_id)

    def update_chunk_metadata(self, chunk_id: str, updates: dict[str, object]) -> bool:
        if not updates:
            return False
        with self._lock:
            chunk = self._get_by_id_unlocked(chunk_id)
            if chunk is None:
                return False
            metadata = dict(chunk.metadata)
            metadata.update(updates)
            updated = chunk.model_copy(update={"metadata": metadata})
            if chunk_id in self._id_map:
                self._deleted_ids.add(chunk_id)
            self._pending_chunks[chunk_id] = updated
            self._mark_dirty()
            return True

    def memory_resident_bytes(self) -> int:
        """Approximate bytes retained for search state (excludes mmap pages)."""
        with self._lock:
            total = 0
            for obj in (
                self._df,
                self._idf,
                self._id_map,
                self._pending_chunks,
                self._deleted_ids,
                self._segment_ids,
                self._segment_chunk_ids,
            ):
                total += sys.getsizeof(obj)
            for mapping in (self._df, self._idf, self._id_map, self._pending_chunks):
                total += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in mapping.items())
            for ids in self._segment_chunk_ids.values():
                total += sys.getsizeof(ids) + sum(sys.getsizeof(x) for x in ids)
            return total

    # ── Internals ──────────────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._mutation_generation += 1
        self._soft_view_stats_cache = None

    def _read_size_unlocked(self) -> int:
        return sum(1 for _ in self._iter_live_chunk_ids_unlocked())

    def _iter_live_chunk_ids_unlocked(self) -> Generator[str, None, None]:
        seen: set[str] = set()
        for chunk_id in self._pending_chunks:
            seen.add(chunk_id)
            yield chunk_id
        for chunk_id in self._id_map:
            if chunk_id in seen or chunk_id in self._deleted_ids:
                continue
            yield chunk_id

    def _iter_live_chunks_unlocked(self) -> Generator[tuple[str, Chunk], None, None]:
        for chunk_id in self._iter_live_chunk_ids_unlocked():
            chunk = self._get_by_id_unlocked(chunk_id)
            if chunk is not None:
                yield chunk_id, chunk

    def _get_by_id_unlocked(self, chunk_id: str) -> Chunk | None:
        if chunk_id in self._pending_chunks:
            return self._pending_chunks[chunk_id]
        if chunk_id in self._deleted_ids:
            return None
        loc = self._id_map.get(chunk_id)
        if loc is None:
            return None
        return self._load_chunk(*loc)

    def _schedule_rebuild(self) -> None:
        if self._defer_rebuild_depth > 0:
            self._needs_rebuild = True
            return
        self._flush_to_disk()
        self._needs_rebuild = False

    def _ensure_built(self) -> None:
        if self._needs_rebuild or self._pending_chunks or self._deleted_ids:
            self._flush_to_disk()
            self._needs_rebuild = False

    def _ensure_searchable(self) -> None:
        """Flush only when not inside deferred_rebuild; else search soft view."""
        if self._defer_rebuild_depth > 0:
            return
        if self._needs_rebuild or self._pending_chunks or self._deleted_ids:
            self._flush_to_disk()
            self._needs_rebuild = False

    def _clear_state(self) -> None:
        self._df.clear()
        self._idf.clear()
        self._corpus_size = 0
        self._total_dl = 0
        self._id_map.clear()
        self._segment_ids.clear()
        self._segment_chunk_ids.clear()
        self._pending_chunks.clear()
        self._deleted_ids.clear()
        self._soft_view_stats_cache = None
        if self._path.exists():
            with suppress(OSError):
                shutil.rmtree(self._path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _flush_to_disk(self) -> None:
        """Rebuild segments from the live chunk set (caller holds the lock)."""
        live_chunks = [chunk for _, chunk in self._iter_live_chunks_unlocked()]
        staging = self._path.with_name(f"{self._path.name}.rebuild-{os.getpid()}")
        with suppress(OSError):
            shutil.rmtree(staging)
        meta = self._materialize_segments(staging, live_chunks, self._segment_size)
        if self._path.exists():
            with suppress(OSError):
                shutil.rmtree(self._path)
        staging.rename(self._path)
        self._apply_materialized_state(meta)
        self._pending_chunks.clear()
        self._deleted_ids.clear()
        self._soft_view_stats_cache = None

    def _soft_view_corpus_stats(self) -> tuple[dict[str, int], int, int]:
        """Return live (df, corpus_size, total_dl) including deferred mutations.

        On-disk tables stay frozen until flush. Soft-view search must undo
        deleted/replaced documents' contributions and fold in pending chunks so
        IDF/avgdl match the in-memory backend during: meth:`deferred_rebuild`.
        """
        if not self._pending_chunks and not self._deleted_ids:
            return self._df, self._corpus_size, self._total_dl
        cached = self._soft_view_stats_cache
        if cached is not None:
            return cached

        df = dict(self._df)
        size = self._corpus_size
        total_dl = self._total_dl

        # Undo every soft-deleted disk doc (pure deletes and replacements).
        for chunk_id in self._deleted_ids:
            loc = self._id_map.get(chunk_id)
            if loc is None:
                continue
            old = self._load_chunk(*loc)
            if old is None:
                continue
            toks = _tokenize(old.text)
            size = max(0, size - 1)
            total_dl = max(0, total_dl - len(toks))
            for term in _term_freqs(toks):
                remaining = df.get(term, 0) - 1
                if remaining <= 0:
                    df.pop(term, None)
                else:
                    df[term] = remaining

        # Fold pending docs (net-new and replacements) into the soft view.
        for chunk in self._pending_chunks.values():
            toks = _tokenize(chunk.text)
            size += 1
            total_dl += len(toks)
            for term in _term_freqs(toks):
                df[term] = df.get(term, 0) + 1

        self._soft_view_stats_cache = (df, size, total_dl)
        return self._soft_view_stats_cache

    @staticmethod
    def _materialize_segments(
        root: Path,
        live_chunks: list[Chunk],
        segment_size: int,
    ) -> dict[str, Any]:
        """Write a complete disk index under *root* and return in-memory metadata."""
        with suppress(OSError):
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        segments_root = root / _SEGMENTS_DIR
        segments_root.mkdir(parents=True, exist_ok=True)

        df: dict[str, int] = {}
        id_map: dict[str, tuple[int, int]] = {}
        segment_ids: list[int] = []
        segment_chunk_ids: dict[int, list[str]] = {}
        corpus_size = 0
        total_dl = 0

        writer: _SegmentWriter | None = None
        next_segment_id = 0

        def close_writer() -> None:
            nonlocal writer, next_segment_id
            if writer is None or len(writer) == 0:
                writer = None
                return
            writer.flush()
            segment_ids.append(writer.segment_id)
            ids = [c.id for c in writer.chunks]
            segment_chunk_ids[writer.segment_id] = ids
            for local_idx, seg_chunk in enumerate(writer.chunks):
                id_map[seg_chunk.id] = (writer.segment_id, local_idx)
            next_segment_id = writer.segment_id + 1
            writer = None

        for live_chunk in live_chunks:
            if writer is None:
                writer = _SegmentWriter(next_segment_id, segments_root / f"{next_segment_id:06d}")
            tokens = _tokenize(live_chunk.text)
            for term in _term_freqs(tokens):
                df[term] = df.get(term, 0) + 1
            corpus_size += 1
            total_dl += len(tokens)
            writer.add(live_chunk)
            if len(writer) >= segment_size:
                close_writer()
        close_writer()

        idf = _bm25_idf(corpus_size, df)
        _atomic_write_json(root / _DF_NAME, df)
        _atomic_write_json(root / _IDF_NAME, idf)
        _atomic_write_json(
            root / _ID_MAP_NAME,
            {cid: [seg, loc] for cid, (seg, loc) in id_map.items()},
        )
        _atomic_write_json(
            root / _MANIFEST_NAME,
            {
                "version": _INDEX_FORMAT_VERSION,
                "k1": _K1,
                "b": _B,
                "epsilon": _EPSILON,
                "corpus_size": corpus_size,
                "total_dl": total_dl,
                "avgdl": (total_dl / corpus_size) if corpus_size else 0.0,
                "segment_size": segment_size,
                "segment_ids": segment_ids,
            },
        )
        return {
            "df": df,
            "idf": idf,
            "id_map": id_map,
            "segment_ids": segment_ids,
            "segment_chunk_ids": segment_chunk_ids,
            "corpus_size": corpus_size,
            "total_dl": total_dl,
            "segment_size": segment_size,
        }

    def _apply_materialized_state(self, meta: dict[str, Any]) -> None:
        self._df = meta["df"]
        self._idf = meta["idf"]
        self._id_map = meta["id_map"]
        self._segment_ids = meta["segment_ids"]
        self._segment_chunk_ids = meta["segment_chunk_ids"]
        self._corpus_size = meta["corpus_size"]
        self._total_dl = meta["total_dl"]
        self._segment_size = meta["segment_size"]
        self._soft_view_stats_cache = None

    def _segment_dir(self, segment_id: int) -> Path:
        return self._path / _SEGMENTS_DIR / f"{segment_id:06d}"

    def _load_chunk(self, segment_id: int, local_idx: int) -> Chunk | None:
        path = self._segment_dir(segment_id) / "chunks.jsonl"
        try:
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i == local_idx:
                        return Chunk.model_validate(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(
                f"Cannot load chunk ({segment_id},{local_idx}) from {path}",
                cause=exc,
            ) from exc
        return None

    def _mmap_lengths(self, segment_id: int) -> np.ndarray:
        path = self._segment_dir(segment_id) / "lengths.npy"
        try:
            loaded: np.ndarray = np.load(path, mmap_mode="r")
            return loaded
        except OSError as exc:
            raise VectorStoreError(f"Cannot mmap lengths at {path}", cause=exc) from exc

    def _load_postings(self, segment_id: int) -> dict[str, list[list[int]]]:
        path = self._segment_dir(segment_id) / "postings.json"
        data = _read_json(path)
        if not isinstance(data, dict):
            raise VectorStoreError(f"Invalid postings at {path}")
        return data

    def _search_unlocked(
        self,
        tokens: list[str],
        top_k: int,
        *,
        filters: RetrievalFilter | None,
    ) -> list[tuple[Chunk, float]]:
        scored: dict[str, float] = defaultdict(float)

        live_df, live_size, live_total_dl = self._soft_view_corpus_stats()
        if live_size <= 0:
            return []
        live_idf = (
            self._idf
            if not self._pending_chunks and not self._deleted_ids
            else _bm25_idf(live_size, live_df)
        )
        avgdl = live_total_dl / live_size

        # Disk-backed segments (skip soft-deleted / pending replacements).
        if self._segment_ids:
            query_terms = [t for t in tokens if t in live_idf]
            for segment_id in self._segment_ids:
                lengths = self._mmap_lengths(segment_id)
                postings = self._load_postings(segment_id)
                ids = self._segment_chunk_ids.get(segment_id) or [
                    str(x) for x in _read_json(self._segment_dir(segment_id) / "ids.json")
                ]
                for term in query_terms:
                    idf = live_idf.get(term, 0.0)
                    for local_idx, tf in postings.get(term, []):
                        # pyrefly: ignore [unnecessary-type-conversion]
                        li = int(local_idx)
                        if li < 0 or li >= len(ids):
                            continue
                        chunk_id = ids[li]
                        if chunk_id in self._deleted_ids or chunk_id in self._pending_chunks:
                            continue
                        dl = float(lengths[li])
                        scored[chunk_id] += _score_term(float(tf), dl, avgdl, idf)
                del postings

        # Soft view for pending (deferred) chunks — same IDF/avgdl as disk rows.
        for chunk_id, chunk in self._pending_chunks.items():
            toks = _tokenize(chunk.text)
            tf_map = _term_freqs(toks)
            dl = float(len(toks)) or 1.0
            score = 0.0
            for term in tokens:
                score += _score_term(
                    float(tf_map.get(term, 0)),
                    dl,
                    avgdl,
                    live_idf.get(term, 0.0),
                )
            if score > 0:
                scored[chunk_id] = score

        ranked = sorted(
            ((cid, score) for cid, score in scored.items() if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[tuple[Chunk, float]] = []
        for chunk_id, score in ranked:
            maybe_chunk = self._get_by_id_unlocked(chunk_id)
            if maybe_chunk is None:
                continue
            if not chunk_matches_filter(maybe_chunk, filters):
                continue
            results.append((maybe_chunk, score))
            if len(results) >= top_k:
                break
        return results
