"""T-025 — RetrievalService and RetrievalPipeline tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.retrieval_service import RetrievalResult, RetrievalService
from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline

# ── helpers ────────────────────────────────────────────────────────────────────


_VEC = [0.1, 0.2, 0.3, 0.4]


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"relevant text {i}")


def _query(text: str = "What is EKS?") -> Query:
    return Query(text=text)


def _dense_mock() -> MagicMock:
    m = MagicMock()
    m.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": _VEC})
    return m


def _hybrid_mock(chunks: list[Chunk] | None = None) -> MagicMock:
    results = [(c, 0.9 - i * 0.1) for i, c in enumerate(chunks or [_chunk(0), _chunk(1)])]
    m = MagicMock()
    m.retrieve = AsyncMock(return_value=results)
    return m


def _service(
    n_chunks: int = 3,
    with_expander: bool = False,
    with_reranker: bool = False,
    with_compressor: bool = False,
) -> RetrievalService:
    chunks = [_chunk(i) for i in range(n_chunks)]

    expander = MagicMock() if with_expander else None
    if expander:
        expander.expand.side_effect = lambda q: q.model_copy(
            update={"expanded_texts": ["variant 1"]}
        )

    reranker = MagicMock() if with_reranker else None
    if reranker:
        reranker.rerank.return_value = chunks[:2]

    compressor = MagicMock() if with_compressor else None
    if compressor:
        compressed = [c.model_copy(update={"text": "compressed"}) for c in chunks[:2]]
        compressor.compress.return_value = compressed

    return RetrievalService(
        dense_retriever=_dense_mock(),
        hybrid_retriever=_hybrid_mock(chunks),
        query_expander=expander,
        reranker=reranker,
        compressor=compressor,
        top_k_retrieval=10,
        top_k_rerank=5,
    )


# ── RetrievalResult ────────────────────────────────────────────────────────────


class TestRetrievalResult:
    def test_fields(self):
        q = _query()
        chunks = [_chunk(0)]
        r = RetrievalResult(query=q, chunks=chunks, context="text", latency_ms=12.3)
        assert r.query is q
        assert r.chunks == chunks
        assert r.context == "text"
        assert r.latency_ms == pytest.approx(12.3)

    def test_default_latency(self):
        r = RetrievalResult(query=_query(), chunks=[], context="")
        assert r.latency_ms == 0.0


# ── RetrievalService ───────────────────────────────────────────────────────────


class TestRetrievalServiceMinimal:
    @pytest.mark.asyncio
    async def test_returns_retrieval_result(self):
        result = await _service().retrieve(_query())
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_query_embedding_populated(self):
        result = await _service().retrieve(_query())
        assert result.query.embedding == _VEC

    @pytest.mark.asyncio
    async def test_chunks_populated(self):
        result = await _service().retrieve(_query())
        assert isinstance(result.chunks, list)
        assert all(isinstance(c, Chunk) for c in result.chunks)

    @pytest.mark.asyncio
    async def test_context_is_joined_chunk_text(self):
        result = await _service(n_chunks=2).retrieve(_query())
        for chunk in result.chunks:
            assert chunk.text in result.context

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        result = await _service().retrieve(_query())
        assert result.latency_ms > 0


class TestRetrievalServiceExpansion:
    @pytest.mark.asyncio
    async def test_expander_called(self):
        svc = _service(with_expander=True)
        await svc.retrieve(_query())
        svc._expander.expand.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_no_expander_skipped(self):
        svc = _service(with_expander=False)
        result = await svc.retrieve(_query())
        assert result.query.expanded_texts == []


class TestRetrievalServiceReranking:
    @pytest.mark.asyncio
    async def test_reranker_called(self):
        svc = _service(with_reranker=True)
        await svc.retrieve(_query())
        svc._reranker.rerank.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_no_reranker_uses_hybrid_results(self):
        svc = _service(n_chunks=3, with_reranker=False)
        result = await svc.retrieve(_query())
        assert len(result.chunks) == 3

    @pytest.mark.asyncio
    async def test_reranker_top_k_applied(self):
        svc = _service(n_chunks=3, with_reranker=True)
        await svc.retrieve(_query())
        _, kwargs = svc._reranker.rerank.call_args  # type: ignore[union-attr]
        assert kwargs["top_k"] == svc._top_k_rerank


class TestRetrievalServiceCompression:
    @pytest.mark.asyncio
    async def test_compressor_called(self):
        svc = _service(with_compressor=True)
        await svc.retrieve(_query())
        svc._compressor.compress.assert_called_once()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_compressed_text_in_context(self):
        svc = _service(n_chunks=2, with_compressor=True)
        result = await svc.retrieve(_query())
        assert "compressed" in result.context

    @pytest.mark.asyncio
    async def test_no_compressor_uses_reranked_chunks(self):
        svc = _service(n_chunks=2, with_compressor=False)
        result = await svc.retrieve(_query())
        assert result.context != ""


# ── RetrievalPipeline ──────────────────────────────────────────────────────────


class TestRetrievalPipeline:
    @pytest.mark.asyncio
    async def test_retrieve_returns_result(self):
        pipeline = RetrievalPipeline(service=_service())
        result = await pipeline.retrieve(_query())
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_otel_span_does_not_raise(self):
        # OTel is a no-op without a configured provider — must not crash.
        pipeline = RetrievalPipeline(service=_service())
        await pipeline.retrieve(_query())

    def test_retrieve_sync(self):
        pipeline = RetrievalPipeline(service=_service())
        result = pipeline.retrieve_sync(_query())
        assert isinstance(result, RetrievalResult)
