"""T-146 / T-172 scenario 5 — concurrent feedback accumulation regression.

Run with Qdrant available:
    uv run pytest tests/benchmarks/test_feedback_concurrency.py -v -s

Uses multiprocessing to simulate concurrent API pods voting on the same chunk.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from uuid import NAMESPACE_DNS, uuid5

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from src.domain.entities.chunk import Chunk

_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
_COLLECTION = "test_feedback_concurrency"
_N_WORKERS = 10
_N_VOTES = 10


def _reachable() -> bool:
    try:
        QdrantClient(url=_QDRANT_URL, timeout=2, check_compatibility=False).get_collections()
        return True
    except (OSError, TimeoutError, ResponseHandlingException):
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason=f"Qdrant not reachable at {_QDRANT_URL}",
)


def _chunk_id() -> str:
    return str(uuid5(NAMESPACE_DNS, "feedback-concurrency-chunk"))


def _chunk() -> Chunk:
    return Chunk(
        id=_chunk_id(),
        document_id="bench-doc",
        text="Concurrent feedback benchmark chunk",
        embedding=[0.1, 0.2, 0.3, 0.4],
        sparse_vector={1: 0.9},
        metadata={"source": "feedback_concurrency_benchmark"},
    )


def _worker_accumulate(_: int) -> None:
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    store = QdrantVectorStore(url=_QDRANT_URL, collection=_COLLECTION, dense_dim=4)
    store.accumulate_feedback_score(_chunk_id(), 1.0)


@pytest.fixture(scope="module")
def store():
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    s = QdrantVectorStore(url=_QDRANT_URL, collection=_COLLECTION, dense_dim=4)
    yield s
    import contextlib

    with contextlib.suppress(OSError, ConnectionError, TimeoutError):
        s.drop_collection()


class TestFeedbackConcurrencyBenchmark:
    def test_ten_concurrent_increments_no_lost_updates(self, store):
        """Simulate 10 API pods each posting 1 vote — expect score == 10."""
        chunk = _chunk()
        store.upsert([chunk])
        assert store.get_feedback_score(chunk.id) == 0.0

        with ProcessPoolExecutor(max_workers=_N_WORKERS) as pool:
            list(pool.map(_worker_accumulate, range(_N_VOTES)))

        final = store.get_feedback_score(chunk.id)
        assert final == float(_N_VOTES), f"Lost increments: expected {_N_VOTES}, got {final}"

    def test_redis_backend_concurrent_increments(self, tmp_path_factory):
        """Redis HINCRBYFLOAT path — zero-lost increments under process contention."""
        pytest.importorskip("redis")
        import redis

        client = redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )
        try:
            client.ping()
        except Exception as exc:
            pytest.skip(f"Redis not available: {exc}")

        from src.infrastructure.vectordb.feedback_store import RedisFeedbackStore

        hash_key = f"rag:feedback:test:{uuid5(NAMESPACE_DNS, 'redis-concurrency')}"
        client.delete(hash_key)
        feedback = RedisFeedbackStore(client, hash_key=hash_key)

        def _redis_vote(_: int) -> None:
            import redis as redis_mod

            worker = redis_mod.from_url(
                os.environ.get("REDIS_URL", "redis://localhost:6379"),
                decode_responses=True,
            )
            store = RedisFeedbackStore(worker, hash_key=hash_key)
            store.accumulate("chunk-a", 1.0)

        with ProcessPoolExecutor(max_workers=_N_WORKERS) as pool:
            list(pool.map(_redis_vote, range(_N_VOTES)))

        assert feedback.get_score("chunk-a") == float(_N_VOTES)
        client.delete(hash_key)
