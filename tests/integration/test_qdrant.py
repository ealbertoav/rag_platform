"""T-013 integration tests — QdrantVectorStore (requires running Qdrant).

Run with:
    make qdrant-up # start Qdrant via Docker
    uv run pytest tests/integration/test_qdrant.py -v
"""

from __future__ import annotations

from uuid import NAMESPACE_DNS, uuid5

import pytest

from src.domain.entities.chunk import Chunk

_QDRANT_URL = "http://localhost:6333"
_COLLECTION = "test_integration"


def _chunk_id(i: int) -> str:
    """Stable UUID point IDs — Qdrant rejects arbitrary string IDs."""
    return str(uuid5(NAMESPACE_DNS, f"integ-chunk-{i:04d}"))


def _reachable() -> bool:
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=_QDRANT_URL, timeout=2, check_compatibility=False).get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="Qdrant not reachable at localhost:6333")


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=_chunk_id(i),
        document_id="integ-doc-1",
        text=f"Integration test chunk number {i}",
        embedding=[float(j + i) / 100 for j in range(4)],
        sparse_vector={i + 1: 0.9, i + 2: 0.5},
        metadata={"source": "integration_test", "index": i},
    )


@pytest.fixture(scope="module")
def store():
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    s = QdrantVectorStore(url=_QDRANT_URL, collection=_COLLECTION, dense_dim=4)
    yield s
    import contextlib

    with contextlib.suppress(OSError, ConnectionError, TimeoutError):
        s.drop_collection()


class TestQdrantIntegration:
    def test_upsert_and_count(self, store):
        chunks = [_chunk(i) for i in range(5)]
        store.upsert(chunks)
        assert store.count() >= 5

    def test_dense_search_returns_results(self, store):
        query = [0.01, 0.02, 0.03, 0.04]
        results = store.search_dense(query, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(c, Chunk) for c, _ in results)
        assert all(isinstance(s, float) for _, s in results)

    def test_sparse_search_returns_results(self, store):
        results = store.search_sparse({1: 0.9, 2: 0.5}, top_k=3)
        assert isinstance(results, list)

    def test_hybrid_search_returns_results(self, store):
        results = store.search_hybrid(
            query_vector=[0.01, 0.02, 0.03, 0.04],
            query_sparse={1: 0.9},
            alpha=0.7,
            top_k=3,
        )
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_delete_removes_chunks(self, store):
        before = store.count()
        store.delete([_chunk_id(0)])
        after = store.count()
        assert after == before - 1

    def test_chunk_payload_round_trip(self, store):
        c = _chunk(99)
        store.upsert([c])
        results = store.search_dense(c.embedding, top_k=1)  # type: ignore[arg-type]
        assert results[0][0].text == c.text
        assert results[0][0].document_id == c.document_id

    def test_upsert_preserves_feedback_metadata(self, store):
        chunk = _chunk(50)
        store.upsert([chunk])
        store.accumulate_feedback_score(chunk.id, 2.0)
        store.accumulate_feedback_score(chunk.id, 1.5)
        assert store.get_feedback_score(chunk.id) == 3.5
        assert store.get_feedback_revision(chunk.id) == 2

        reindexed = _chunk(50)
        store.upsert([reindexed])

        assert store.get_feedback_score(chunk.id) == 3.5
        assert store.get_feedback_revision(chunk.id) == 2

    def test_accumulate_feedback_score_is_linearizable(self, store):
        chunk = _chunk(51)
        store.upsert([chunk])
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda _: store.accumulate_feedback_score(chunk.id, 1.0), range(10))
            )

        assert store.get_feedback_score(chunk.id) == 10.0
        assert store.get_feedback_revision(chunk.id) == 10
        assert sorted(results) == list(range(1, 11))
