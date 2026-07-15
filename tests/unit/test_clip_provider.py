"""T-251 unit tests — CLIP multimodal embedding provider (model mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.infrastructure.embeddings.clip_provider import ClipEmbeddingProvider

_TEXTS = ["A photo of a cat.", "A diagram of a network."]
_DIM = 512


def _st_mock(n: int, dim: int = _DIM) -> MagicMock:
    """Fake SentenceTransformer that returns random unit vectors."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype("float32")
    m = MagicMock()
    m.encode.return_value = vecs
    return m


@pytest.fixture
def provider() -> ClipEmbeddingProvider:
    return ClipEmbeddingProvider(model_path="fake/clip", device="cpu", batch_size=8)


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(2):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (4, 4), color=(i * 10, 0, 0)).save(p)
        paths.append(p)
    return paths


# ── Interface conformance ──────────────────────────────────────────────────────


class TestInterfaceConformance:
    def test_implements_embedding_repository(self, provider: ClipEmbeddingProvider):
        assert isinstance(provider, EmbeddingRepository)

    def test_from_settings_returns_instance(self):
        assert isinstance(ClipEmbeddingProvider.from_settings(), ClipEmbeddingProvider)


# ── embed() (text) ─────────────────────────────────────────────────────────────


class TestEmbed:
    def test_returns_one_vector_per_text(self, provider: ClipEmbeddingProvider):
        provider._model = _st_mock(len(_TEXTS))
        result = provider.embed(_TEXTS)
        assert len(result) == len(_TEXTS)
        assert all(len(v) == _DIM for v in result)

    def test_empty_input_returns_empty(self, provider: ClipEmbeddingProvider):
        assert provider.embed([]) == []

    def test_embed_sparse_returns_empty_dicts(self, provider: ClipEmbeddingProvider):
        assert provider.embed_sparse(_TEXTS) == [{}, {}]


# ── embed_image() ──────────────────────────────────────────────────────────────


class TestEmbedImage:
    def test_returns_one_vector_per_image(
        self, provider: ClipEmbeddingProvider, image_paths: list[Path]
    ):
        provider._model = _st_mock(len(image_paths))
        result = provider.embed_image(image_paths)
        assert len(result) == len(image_paths)
        assert all(len(v) == _DIM for v in result)

    def test_empty_input_returns_empty(self, provider: ClipEmbeddingProvider):
        assert provider.embed_image([]) == []

    def test_uses_same_model_instance_as_text(
        self, provider: ClipEmbeddingProvider, image_paths: list[Path]
    ):
        """embed() and embed_image() must share the same underlying model."""
        mock = _st_mock(max(len(_TEXTS), len(image_paths)))
        provider._model = mock
        provider.embed(_TEXTS)
        provider.embed_image(image_paths)
        assert mock.encode.call_count == 2

    def test_batch_size_forwarded(self, provider: ClipEmbeddingProvider, image_paths: list[Path]):
        mock = _st_mock(len(image_paths))
        provider._model = mock
        provider.embed_image(image_paths)
        _, kwargs = mock.encode.call_args
        assert kwargs["batch_size"] == provider.batch_size

    def test_missing_file_raises_embedding_error(
        self, provider: ClipEmbeddingProvider, tmp_path: Path
    ):
        provider._model = _st_mock(1)
        with pytest.raises(EmbeddingError):
            provider.embed_image([tmp_path / "missing.png"])

    def test_encode_failure_raises_embedding_error(
        self, provider: ClipEmbeddingProvider, image_paths: list[Path]
    ):
        mock = MagicMock()
        mock.encode.side_effect = RuntimeError("OOM")
        provider._model = mock
        with pytest.raises(EmbeddingError):
            provider.embed_image(image_paths)


# ── Error handling ─────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_model_load_error_raises_embedding_error(self):
        p = ClipEmbeddingProvider(model_path="bad/path")
        fake_module = MagicMock()
        fake_module.SentenceTransformer = MagicMock(side_effect=OSError("not found"))
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            pytest.raises(EmbeddingError),
        ):
            p._get_model()
