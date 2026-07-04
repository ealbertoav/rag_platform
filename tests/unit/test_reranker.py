"""T-023 unit tests — BGERerankerProvider and CrossEncoder (model mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository
from src.infrastructure.rerankers.bge_reranker import BGERerankerProvider
from src.rag.ranking.cross_encoder import CrossEncoder

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int, text: str = "") -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=text or f"chunk text {i}")


def _chunks(n: int) -> list[Chunk]:
    return [_chunk(i) for i in range(n)]


def _provider_and_mock(
    scores: list[float] | None = None, batch_size: int = 16
) -> tuple[BGERerankerProvider, MagicMock]:
    p = BGERerankerProvider(model_path="fake/path", device="cpu", batch_size=batch_size)
    model = MagicMock()
    model.compute_score.return_value = scores if scores is not None else [0.9, 0.3, 0.7]
    p._model = model  # type: ignore[assignment]
    return p, model


def _provider(scores: list[float] | None = None, batch_size: int = 16) -> BGERerankerProvider:
    return _provider_and_mock(scores, batch_size)[0]


# ── BGERerankerProvider ────────────────────────────────────────────────────────


class TestBGERerankerProvider:
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

    def test_calls_compute_score_with_pairs(self):
        p, mock = _provider_and_mock([0.8])
        chunks = [_chunk(0, text="relevant passage")]
        p.rerank("my query", chunks, top_k=1)
        # pairs are tuples since the stub expects list[tuple[str, str]]
        mock.compute_score.assert_called_once_with(
            [("my query", "relevant passage")], normalize=True
        )

    def test_single_score_float_handled(self):
        """compute_score may return a bare float for a single pair."""
        p, mock = _provider_and_mock()
        mock.compute_score.return_value = 0.85
        result = p.rerank("q", [_chunk(0)], top_k=1)
        assert len(result) == 1

    def test_batching_splits_pairs(self):
        # batch_size=2, 5 chunks → 3 compute_score calls (2 + 2 + 1)
        p, mock = _provider_and_mock(batch_size=2)
        mock.compute_score.side_effect = [[0.9, 0.8], [0.7, 0.6], [0.5]]
        p.rerank("q", _chunks(5), top_k=5)
        assert mock.compute_score.call_count == 3

    def test_top_k_larger_than_chunks_returns_all(self):
        result = _provider([0.9, 0.5]).rerank("q", _chunks(2), top_k=10)
        assert len(result) == 2

    def test_scoring_error_raises_retrieval_error(self):
        p, mock = _provider_and_mock()
        mock.compute_score.side_effect = RuntimeError("OOM")
        with pytest.raises(RetrievalError) as exc_info:
            p.rerank("q", _chunks(2), top_k=2)
        assert exc_info.value.cause is not None

    def test_model_load_error_raises_retrieval_error(self):
        p = BGERerankerProvider(model_path="bad/path", device="cpu")
        with (
            patch(
                "src.infrastructure.rerankers.bge_reranker.FlagReranker",
                side_effect=OSError("not found"),
                create=True,
            ),
            pytest.raises(RetrievalError),
        ):
            p._get_model()

    def test_from_settings_returns_instance(self):
        assert isinstance(BGERerankerProvider.from_settings(), BGERerankerProvider)


# ── CrossEncoder ───────────────────────────────────────────────────────────────


class TestCrossEncoder:
    @staticmethod
    def _ce(scores: list[float] | None = None, top_k: int = 5) -> CrossEncoder:
        return CrossEncoder(reranker=_provider(scores), top_k=top_k)

    def test_returns_list_of_chunks(self):
        result = self._ce([0.9, 0.5]).rerank("q", _chunks(2))
        assert isinstance(result, list)

    def test_uses_instance_top_k_by_default(self):
        ce = self._ce([0.9, 0.5, 0.3], top_k=2)
        result = ce.rerank("q", _chunks(3))
        assert len(result) == 2

    def test_per_call_top_k_overrides_default(self):
        ce = self._ce([0.9, 0.5, 0.3], top_k=2)
        result = ce.rerank("q", _chunks(3), top_k=1)
        assert len(result) == 1

    def test_empty_chunks_returns_empty(self):
        assert self._ce([0.9, 0.5]).rerank("q", []) == []

    def test_delegates_to_reranker_score(self):
        mock_reranker = MagicMock()
        chunks = _chunks(5)
        mock_reranker.score.return_value = [(chunk, 0.5) for chunk in chunks]
        ce = CrossEncoder(reranker=mock_reranker, top_k=3)
        result = ce.rerank("query", chunks)
        mock_reranker.score.assert_called_once_with("query", chunks)
        assert len(result) == 3

    def test_feedback_boost_promotes_chunk_after_rerank(self):
        from src.core.constants import FEEDBACK_SCORE_KEY

        low = Chunk(id="low", document_id="doc", text="low")
        high = Chunk(
            id="high",
            document_id="doc",
            text="high",
            metadata={FEEDBACK_SCORE_KEY: 5.0},
        )
        ce = CrossEncoder(reranker=_provider([0.9, 0.1]), top_k=2)
        result = ce.rerank("q", [low, high], boost_multiplier=0.2)
        assert result[0].id == "high"

    def test_from_settings_returns_instance(self):
        # BGERerankerProvider uses lazy model loading, so no download happens here.
        ce = CrossEncoder.from_settings()
        assert isinstance(ce, CrossEncoder)
