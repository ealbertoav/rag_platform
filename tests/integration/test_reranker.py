"""T-023 integration tests — BGERerankerProvider (requires model on disk).

Run with:
    uv run pytest tests/integration/test_reranker.py -v
"""
from __future__ import annotations

import pytest

from src.core.constants import MODELS_DIR

_MODEL_PATH = MODELS_DIR / "rerankers" / "bge-reranker-v2-m3"

pytestmark = pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason=f"BGE-Reranker model not found at {_MODEL_PATH}",
)

_QUERY = "How do IAM roles work in Amazon EKS?"
_CHUNKS_TEXT = [
    "IAM roles for service accounts (IRSA) allow Kubernetes pods to assume AWS IAM roles.",
    "The quick brown fox jumps over the lazy dog.",
    "EKS node groups require IAM instance profiles with specific policies.",
    "Python is a high-level programming language.",
    "Configuring IRSA requires an OIDC provider associated with the EKS cluster.",
]


@pytest.fixture(scope="module")
def provider():
    from src.infrastructure.rerankers.bge_reranker import BGERerankerProvider

    return BGERerankerProvider(
        model_path=str(_MODEL_PATH),
        device="mps",
        batch_size=4,
    )


@pytest.fixture(scope="module")
def chunks():
    from src.domain.entities.chunk import Chunk
    return [
        Chunk(id=f"c{i}", document_id="doc", text=text)
        for i, text in enumerate(_CHUNKS_TEXT)
    ]


class TestBGERerankerIntegration:
    def test_rerank_returns_list(self, provider, chunks):
        from src.domain.entities.chunk import Chunk
        result = provider.rerank(_QUERY, chunks, top_k=3)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_top_k_respected(self, provider, chunks):
        result = provider.rerank(_QUERY, chunks, top_k=2)
        assert len(result) == 2

    def test_relevant_chunks_rank_higher(self, provider, chunks):
        result = provider.rerank(_QUERY, chunks, top_k=3)
        top_ids = {c.id for c in result}
        # IAM/EKS related chunks (c0, c2, c4) should dominate the top-3
        assert len(top_ids & {"c0", "c2", "c4"}) >= 2

    def test_mps_device_no_error(self, provider):
        assert provider.device == "mps"

    def test_batching_produces_same_result(self, provider, chunks):
        result_small_batch = provider.rerank(_QUERY, chunks, top_k=3)
        large_provider = type(provider)(
            model_path=str(_MODEL_PATH), device="mps", batch_size=32
        )
        large_provider._model = provider._model
        result_large_batch = large_provider.rerank(_QUERY, chunks, top_k=3)
        assert [c.id for c in result_small_batch] == [c.id for c in result_large_batch]
