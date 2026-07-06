"""T-025 — RetrievalService and RetrievalPipeline tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
        expander.expand.side_effect = lambda q, n_variants=None: q.model_copy(
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

    @pytest.mark.asyncio
    async def test_multi_query_fusion_calls_hybrid_per_variant(self):
        svc = _service(with_expander=True, n_chunks=2)
        await svc.retrieve(_query())
        # original + 1 variant → 2 hybrid calls
        assert svc.hybrid.retrieve.call_count == 2  # type: ignore[union-attr]


class TestRetrievalServiceTopKFinal:
    @pytest.mark.asyncio
    async def test_top_k_final_limits_output_chunks(self):
        svc = RetrievalService(
            dense_retriever=_dense_mock(),
            hybrid_retriever=_hybrid_mock([_chunk(i) for i in range(10)]),
            top_k_retrieval=10,
            top_k_rerank=10,
            top_k_final=2,
        )
        result = await svc.retrieve(_query())
        assert len(result.chunks) == 2


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


class TestRetrievalServiceRSE:
    @pytest.mark.asyncio
    async def test_rse_disabled_by_default(self):
        from src.core.constants import MERGED_CHUNK_IDS_KEY

        chunks = [
            Chunk(
                id="c0",
                document_id="doc",
                text="part one",
                metadata={"chunk_index": 0},
            ),
            Chunk(
                id="c1",
                document_id="doc",
                text="part two",
                metadata={"chunk_index": 1},
            ),
        ]
        svc = RetrievalService(
            dense_retriever=_dense_mock(),
            hybrid_retriever=_hybrid_mock(chunks),
            top_k_retrieval=10,
        )
        result = await svc.retrieve(_query())
        assert len(result.chunks) == 2
        assert MERGED_CHUNK_IDS_KEY not in result.chunks[0].metadata

    @pytest.mark.asyncio
    async def test_rse_enabled_merges_adjacent_chunks(self):
        from src.core.constants import MERGED_CHUNK_IDS_KEY

        chunks = [
            Chunk(
                id="c0",
                document_id="doc",
                text="part one",
                metadata={"chunk_index": 0},
            ),
            Chunk(
                id="c1",
                document_id="doc",
                text="part two",
                metadata={"chunk_index": 1},
            ),
        ]
        svc = RetrievalService(
            dense_retriever=_dense_mock(),
            hybrid_retriever=_hybrid_mock(chunks),
            top_k_retrieval=10,
            rse_enabled=True,
            rse_max_segment_tokens=500,
        )
        result = await svc.retrieve(_query())
        assert len(result.chunks) == 1
        assert result.chunks[0].metadata[MERGED_CHUNK_IDS_KEY] == ["c0", "c1"]


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

    def test_service_property(self):
        svc = _service()
        pipeline = RetrievalPipeline(service=svc)
        assert pipeline.service is svc


class TestRetrievalPipelineFromSettings:
    def test_from_settings_builds_pipeline(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.rag.retrieval.bm25_retriever.BM25Retriever"),
            patch("src.rag.retrieval.dense_retriever.DenseRetriever"),
            patch("src.rag.retrieval.hybrid_retriever.HybridRetriever"),
            patch("src.rag.ranking.cross_encoder.CrossEncoder.from_settings"),
        ):
            mock_settings.retrieval = MagicMock(
                hybrid_alpha=0.7,
                top_k_dense=10,
                top_k_final=5,
                hybrid_fusion="rrf",
                rse=MagicMock(enabled=False, max_segment_tokens=1500),
            )
            mock_settings.neo4j = MagicMock(enabled=False)
            mock_settings.reranker = MagicMock(top_k=5)
            mock_settings.query_expansion = MagicMock(enabled=False)
            mock_settings.compression = MagicMock(enabled=False)
            mock_llm.return_value = MagicMock()
            pipeline = RetrievalPipeline.from_settings()

        assert isinstance(pipeline, RetrievalPipeline)
        assert isinstance(pipeline.service, RetrievalService)

    def test_from_settings_with_expansion_and_compression(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.rag.retrieval.bm25_retriever.BM25Retriever"),
            patch("src.rag.retrieval.dense_retriever.DenseRetriever"),
            patch("src.rag.retrieval.hybrid_retriever.HybridRetriever"),
            patch("src.rag.retrieval.query_expansion.QueryExpander.from_settings"),
            patch("src.rag.compression.contextual_compression.ContextualCompressor.from_settings"),
            patch("src.rag.ranking.cross_encoder.CrossEncoder.from_settings"),
        ):
            mock_settings.retrieval = MagicMock(
                hybrid_alpha=0.5,
                top_k_dense=20,
                top_k_final=5,
                hybrid_fusion="rrf",
                rse=MagicMock(enabled=False, max_segment_tokens=1500),
            )
            mock_settings.neo4j = MagicMock(enabled=False)
            mock_settings.reranker = MagicMock(top_k=5)
            mock_settings.query_expansion = MagicMock(enabled=True)
            mock_settings.compression = MagicMock(enabled=True)
            mock_llm.return_value = MagicMock()
            pipeline = RetrievalPipeline.from_settings(llm=mock_llm.return_value)

        assert isinstance(pipeline, RetrievalPipeline)


class TestBuildGraphRetriever:
    def test_returns_none_when_neo4j_disabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_graph_retriever

        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.neo4j = MagicMock(enabled=False)
            assert _build_graph_retriever(MagicMock(), MagicMock()) is None

    def test_returns_retriever_when_enabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_graph_retriever

        graph_mock = MagicMock()
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.graph_retriever.GraphRetriever.from_settings",
                return_value=graph_mock,
            ),
        ):
            mock_settings.neo4j = MagicMock(enabled=True)
            result = _build_graph_retriever(MagicMock(), MagicMock())
        assert result is graph_mock

    def test_returns_none_on_failure(self, caplog):
        import logging

        from src.rag.pipelines.retrieval_pipeline import _build_graph_retriever

        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.graph_retriever.GraphRetriever.from_settings",
                side_effect=RuntimeError("neo4j down"),
            ),
            caplog.at_level(logging.WARNING, logger="src.rag.pipelines.retrieval_pipeline"),
        ):
            mock_settings.neo4j = MagicMock(enabled=True)
            assert _build_graph_retriever(MagicMock(), MagicMock()) is None
        assert "Graph retriever unavailable" in caplog.text

    def test_from_settings_wires_graph_when_neo4j_enabled(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.rag.retrieval.bm25_retriever.BM25Retriever"),
            patch("src.rag.retrieval.dense_retriever.DenseRetriever"),
            patch("src.rag.retrieval.hybrid_retriever.HybridRetriever") as mock_hybrid,
            patch("src.rag.ranking.cross_encoder.CrossEncoder.from_settings"),
            patch(
                "src.rag.pipelines.retrieval_pipeline._build_graph_retriever",
                return_value=MagicMock(),
            ),
        ):
            mock_settings.retrieval = MagicMock(
                hybrid_alpha=0.7,
                top_k_dense=10,
                top_k_final=5,
                hybrid_fusion="rrf",
                rse=MagicMock(enabled=False, max_segment_tokens=1500),
            )
            mock_settings.neo4j = MagicMock(enabled=True)
            mock_settings.reranker = MagicMock(top_k=5)
            mock_settings.query_expansion = MagicMock(enabled=False)
            mock_settings.compression = MagicMock(enabled=False)
            mock_llm.return_value = MagicMock()
            RetrievalPipeline.from_settings()
            _, kwargs = mock_hybrid.call_args
            assert kwargs["graph_retriever"] is not None


class TestBuildHyPERetriever:
    def test_returns_none_when_hype_disabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_hype_retriever

        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.retrieval = MagicMock(hype=MagicMock(enabled=False))
            assert _build_hype_retriever(MagicMock(), MagicMock(), MagicMock()) is None

    def test_returns_retriever_when_enabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_hype_retriever

        embedder = MagicMock()
        vector_store = MagicMock()
        bm25 = MagicMock()
        hype_mock = MagicMock()
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.hype_retriever.HyPERetriever",
                return_value=hype_mock,
            ) as mock_cls,
        ):
            mock_settings.retrieval = MagicMock(hype=MagicMock(enabled=True))
            result = _build_hype_retriever(embedder, vector_store, bm25)

        assert result is hype_mock
        mock_cls.assert_called_once_with(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=bm25,
        )

    def test_returns_none_on_failure(self, caplog):
        import logging

        from src.rag.pipelines.retrieval_pipeline import _build_hype_retriever

        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.hype_retriever.HyPERetriever",
                side_effect=RuntimeError("hype down"),
            ),
            caplog.at_level(logging.WARNING, logger="src.rag.pipelines.retrieval_pipeline"),
        ):
            mock_settings.retrieval = MagicMock(hype=MagicMock(enabled=True))
            assert _build_hype_retriever(MagicMock(), MagicMock(), MagicMock()) is None
        assert "HyPE retriever unavailable" in caplog.text


class TestBuildHyDERetriever:
    def test_returns_none_when_hyde_disabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_hyde_retriever

        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.retrieval = MagicMock(hyde=MagicMock(enabled=False))
            assert _build_hyde_retriever(MagicMock(), MagicMock(), MagicMock()) is None

    def test_returns_retriever_when_enabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_hyde_retriever

        llm = MagicMock()
        embedder = MagicMock()
        vector_store = MagicMock()
        hyde_mock = MagicMock()
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.hyde_retriever.HyDERetriever",
                return_value=hyde_mock,
            ) as mock_cls,
        ):
            mock_settings.retrieval = MagicMock(hyde=MagicMock(enabled=True))
            result = _build_hyde_retriever(llm, embedder, vector_store)

        assert result is hyde_mock
        mock_cls.assert_called_once_with(
            llm=llm,
            embedder=embedder,
            vector_store=vector_store,
        )

    def test_returns_none_on_failure(self, caplog):
        import logging

        from src.rag.pipelines.retrieval_pipeline import _build_hyde_retriever

        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.hyde_retriever.HyDERetriever",
                side_effect=RuntimeError("hyde down"),
            ),
            caplog.at_level(logging.WARNING, logger="src.rag.pipelines.retrieval_pipeline"),
        ):
            mock_settings.retrieval = MagicMock(hyde=MagicMock(enabled=True))
            assert _build_hyde_retriever(MagicMock(), MagicMock(), MagicMock()) is None
        assert "HyDE retriever unavailable" in caplog.text


class TestBuildHierarchicalRetriever:
    def test_returns_none_when_hierarchical_disabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_hierarchical_retriever

        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.chunking = MagicMock(hierarchical=MagicMock(enabled=False))
            assert _build_hierarchical_retriever(MagicMock(), MagicMock()) is None

    def test_returns_retriever_when_enabled(self):
        from src.rag.pipelines.retrieval_pipeline import _build_hierarchical_retriever

        embedder = MagicMock()
        vector_store = MagicMock()
        hierarchical_mock = MagicMock()
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.hierarchical_retriever.HierarchicalRetriever",
                return_value=hierarchical_mock,
            ) as mock_cls,
        ):
            mock_settings.chunking = MagicMock(
                hierarchical=MagicMock(enabled=True, summary_top_k=4),
            )
            result = _build_hierarchical_retriever(embedder, vector_store)

        assert result is hierarchical_mock
        mock_cls.assert_called_once_with(
            embedder=embedder,
            vector_store=vector_store,
            summary_top_k=4,
        )

    def test_returns_none_on_failure(self, caplog):
        import logging

        from src.rag.pipelines.retrieval_pipeline import _build_hierarchical_retriever

        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.rag.retrieval.hierarchical_retriever.HierarchicalRetriever",
                side_effect=RuntimeError("hierarchical down"),
            ),
            caplog.at_level(logging.WARNING, logger="src.rag.pipelines.retrieval_pipeline"),
        ):
            mock_settings.chunking = MagicMock(hierarchical=MagicMock(enabled=True))
            assert _build_hierarchical_retriever(MagicMock(), MagicMock()) is None
        assert "Hierarchical retriever unavailable" in caplog.text

    def test_from_settings_wires_hierarchical_when_enabled(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.rag.retrieval.bm25_retriever.BM25Retriever"),
            patch("src.rag.retrieval.dense_retriever.DenseRetriever"),
            patch("src.rag.retrieval.hybrid_retriever.HybridRetriever") as mock_hybrid,
            patch("src.rag.ranking.cross_encoder.CrossEncoder.from_settings"),
            patch(
                "src.rag.pipelines.retrieval_pipeline._build_hierarchical_retriever",
                return_value=MagicMock(),
            ),
        ):
            mock_settings.retrieval = MagicMock(
                hybrid_alpha=0.7,
                top_k_dense=10,
                top_k_final=5,
                hybrid_fusion="rrf",
                rse=MagicMock(enabled=False, max_segment_tokens=1500),
                parent_context=MagicMock(enabled=False),
            )
            mock_settings.chunking = MagicMock(strategy="recursive")
            mock_settings.neo4j = MagicMock(enabled=False)
            mock_settings.reranker = MagicMock(top_k=5)
            mock_settings.query_expansion = MagicMock(enabled=False)
            mock_settings.compression = MagicMock(enabled=False)
            mock_llm.return_value = MagicMock()
            RetrievalPipeline.from_settings()
            _, kwargs = mock_hybrid.call_args
            assert kwargs["hierarchical_retriever"] is not None
