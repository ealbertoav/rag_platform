"""T-012 integration tests — BGE-M3 provider (requires model on disk).

Run with:
    uv run pytest tests/integration/test_bge_m3.py -v

The entire module is skipped when the model directory does not exist, so CI
passes without downloaded weights.
"""

from __future__ import annotations

import pytest

from src.core.constants import MODELS_DIR

_MODEL_PATH = MODELS_DIR / "embeddings" / "bge-m3"

pytestmark = pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason=f"BGE-M3 model not found at {_MODEL_PATH}",
)

_TEXTS = [
    "How do I configure IAM Roles in EKS?",
    "What is Retrieval-Augmented Generation?",
    "Kubernetes pod scheduling with node affinity.",
]


@pytest.fixture(scope="module")
def provider():
    from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider

    return BGEM3EmbeddingProvider(
        model_path=str(_MODEL_PATH),
        device="mps",
        batch_size=4,
        normalize=True,
    )


class TestBGEM3Integration:
    def test_embed_returns_correct_shape(self, provider):
        vecs = provider.embed(_TEXTS)
        assert len(vecs) == len(_TEXTS)
        assert all(len(v) == 1024 for v in vecs)

    def test_embed_values_in_range(self, provider):
        import numpy as np

        vecs = provider.embed(_TEXTS)
        arr = np.array(vecs)
        # Normalized vectors have unit norm ≈ 1.0
        norms = np.linalg.norm(arr, axis=1)
        assert all(abs(n - 1.0) < 1e-3 for n in norms)

    def test_embed_sparse_returns_dicts(self, provider):
        result = provider.embed_sparse(_TEXTS)
        assert len(result) == len(_TEXTS)
        assert all(isinstance(d, dict) for d in result)

    def test_embed_sparse_int_keys(self, provider):
        result = provider.embed_sparse(_TEXTS)
        for d in result:
            assert all(isinstance(k, int) for k in d)

    def test_embed_sparse_positive_weights(self, provider):
        result = provider.embed_sparse(_TEXTS)
        for d in result:
            assert all(v > 0 for v in d.values())

    def test_embed_both_consistent(self, provider):
        import numpy as np

        dense_a = provider.embed(_TEXTS[:1])
        dense_b, _ = provider.embed_both(_TEXTS[:1])
        np.testing.assert_allclose(dense_a[0], dense_b[0], atol=1e-5)

    def test_different_texts_produce_different_embeddings(self, provider):
        import numpy as np

        vecs = provider.embed(_TEXTS)
        cos = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        assert cos < 0.999  # not identical

    def test_mps_device_no_error(self, provider):
        # If we got here, the model loaded on MPS without crashing.
        assert provider.device == "mps"
