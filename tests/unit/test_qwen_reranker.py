"""T-262 unit tests — QwenRerankerProvider (model mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository
from src.infrastructure.rerankers.qwen_reranker import QwenRerankerProvider

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int, text: str = "") -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=text or f"chunk text {i}")


def _chunks(n: int) -> list[Chunk]:
    return [_chunk(i) for i in range(n)]


def _provider_and_mock(
    scores: list[float] | None = None, batch_size: int = 16
) -> tuple[QwenRerankerProvider, MagicMock]:
    p = QwenRerankerProvider(model_path="fake/path", device="cpu", batch_size=batch_size)
    model = MagicMock()
    model.predict.return_value = scores if scores is not None else [0.9, 0.3, 0.7]
    p._model = model  # type: ignore[assignment]
    return p, model


def _provider(scores: list[float] | None = None, batch_size: int = 16) -> QwenRerankerProvider:
    return _provider_and_mock(scores, batch_size)[0]


# ── QwenRerankerProvider ────────────────────────────────────────────────────────


class TestQwenRerankerProvider:
    def test_implements_reranker_repository(self):
        assert isinstance(_provider(), RerankerRepository)

    def test_returns_list_of_chunks(self):
        result = _provider([0.9, 0.5]).rerank("q", _chunks(2), top_k=2)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_top_k_respected(self):
        result = _provider([0.9, 0.5, 0.3]).rerank("q", _chunks(3), top_k=2)
        assert len(result) == 2

    def test_sorted_by_score_descending(self):
        chunks = [_chunk(0), _chunk(1), _chunk(2)]
        result = _provider([0.3, 0.9, 0.5]).rerank("q", chunks, top_k=3)
        assert result[0].id == "c1"
        assert result[1].id == "c2"
        assert result[2].id == "c0"

    def test_empty_chunks_returns_empty(self):
        assert _provider().rerank("q", [], top_k=5) == []

    def test_score_returns_chunk_score_pairs(self):
        chunks = [_chunk(0), _chunk(1)]
        scored = _provider([0.9, 0.3]).score("q", chunks)
        assert scored == [(chunks[0], 0.9), (chunks[1], 0.3)]

    def test_score_empty_chunks_returns_empty(self):
        assert _provider().score("q", []) == []

    def test_calls_predict_with_pairs(self):
        p, mock = _provider_and_mock([0.8])
        chunks = [_chunk(0, text="relevant passage")]
        p.rerank("my query", chunks, top_k=1)
        mock.predict.assert_called_once_with([("my query", "relevant passage")])

    def test_batching_splits_pairs(self):
        # batch_size=2, 5 chunks → 3 predict calls (2 + 2 + 1)
        p, mock = _provider_and_mock(batch_size=2)
        mock.predict.side_effect = [[0.9, 0.8], [0.7, 0.6], [0.5]]
        p.rerank("q", _chunks(5), top_k=5)
        assert mock.predict.call_count == 3

    def test_top_k_larger_than_chunks_returns_all(self):
        result = _provider([0.9, 0.5]).rerank("q", _chunks(2), top_k=10)
        assert len(result) == 2

    def test_scoring_error_raises_retrieval_error(self):
        p, mock = _provider_and_mock()
        mock.predict.side_effect = RuntimeError("OOM")
        with pytest.raises(RetrievalError) as exc_info:
            p.rerank("q", _chunks(2), top_k=2)
        assert exc_info.value.cause is not None

    def test_model_load_error_raises_retrieval_error(self):
        p = QwenRerankerProvider(model_path="bad/path", device="cpu")
        fake_module = MagicMock()
        fake_module.CrossEncoder.side_effect = OSError("not found")
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            pytest.raises(RetrievalError),
        ):
            p._get_model()

    def test_from_settings_returns_instance(self):
        assert isinstance(QwenRerankerProvider.from_settings(), QwenRerankerProvider)
