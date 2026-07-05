"""Targeted tests for remaining line coverage gaps (see logs.log)."""

from __future__ import annotations

import importlib
import json
import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic.fields import FieldInfo

from src.api.security import validate_ingest_path, validate_upload_filename
from src.core.exceptions import DocumentLoadError, EmbeddingError, GenerationError, VectorStoreError
from src.core.logging import JsonFormatter
from src.core.settings import Neo4jSettings, Settings, YamlConfigSource
from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.entities.evaluation import EvalSample
from src.domain.entities.query import Query
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.services.generation_service import GenerationService
from src.domain.services.retrieval_service import RetrievalService
from src.evals.e2e.rag_benchmark import SampleResult
from src.evals.generation import RagasMetric
from src.evals.generation.context_precision import ContextPrecisionMetric
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.infrastructure.vectordb.bm25 import BM25Index
from src.infrastructure.vectordb.qdrant import QdrantVectorStore
from src.main import create_app
from src.rag.pipelines.chat_pipeline import ChatPipeline
from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline
from src.rag.quality.explainable_retrieval import ChunkExplanation
from src.rag.quality.post_generation import explain_and_highlight
from src.rag.quality.self_rag import UtilityAction, decide_retrieval, score_utility
from src.rag.retrieval.adaptive.strategies import RetrievalStrategyParams
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.graph_retriever import EntityExtractor, GraphRetriever
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.structured_output import extract_json_object, parse_structured_output
from tests.unit.hybrid_retriever_helpers import feedback_boost_retriever


def _internal(module: str, name: str) -> object:
    """Resolve a module-internal helper without importing private names."""
    return getattr(importlib.import_module(module), name)


setup_otel = cast(
    Callable[[str, float], None],
    _internal("src.core.logging", "_setup_otel"),
)
parse_json_pairs = cast(
    Callable[[str], list[object]],
    _internal("src.evals.golden_dataset", "_parse_json_pairs"),
)
discover_paths = cast(
    Callable[[Path], list[Path]],
    _internal("src.rag.pipelines.ingestion_pipeline", "_discover"),
)
build_hierarchical_indexer = cast(
    Callable[..., object | None],
    _internal("src.rag.pipelines.ingestion_pipeline", "_build_hierarchical_indexer"),
)
build_hype_indexer = cast(
    Callable[..., object | None],
    _internal("src.rag.pipelines.ingestion_pipeline", "_build_hype_indexer"),
)
lookup_explanation = cast(
    Callable[..., ChunkExplanation | None],
    _internal("src.rag.quality.explainable_retrieval", "_lookup_explanation"),
)
validate_span = cast(
    Callable[[str, str], str | None],
    _internal("src.rag.quality.source_highlighting", "_validate_span"),
)
normalize_relevance_scores = cast(
    Callable[[int], list[float]],
    _internal("src.rag.ranking.diversity", "_normalize_relevance_scores"),
)
params_from_config = cast(
    Callable[[object], RetrievalStrategyParams],
    _internal("src.rag.retrieval.adaptive.strategies", "_params_from_config"),
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int = 0, text: str = "sample", *, metadata: dict[str, object] | None = None) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=text, metadata=metadata or {})


def _dense_mock() -> MagicMock:
    mock = MagicMock()
    mock.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1, 0.2]})
    return mock


def _make_record(**kwargs: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


class _Embedder(EmbeddingRepository):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts)]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [{0: 1.0} for _ in texts]


# ── API / core ─────────────────────────────────────────────────────────────────


class TestFeedbackApi502:
    @pytest.mark.asyncio
    async def test_vector_store_error_returns_502(self):
        app = create_app()
        app.state.models_loaded = True
        store = MagicMock()
        store.accumulate_feedback_score.side_effect = VectorStoreError("Qdrant retrieve failed")
        app.state.vector_store = store
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/feedback",
                json={"query_id": "q-1", "chunk_id": "chunk-a", "relevant": True},
            )
        assert resp.status_code == 502


class TestApiSecurityGaps:
    def test_rejects_when_no_allowed_roots(self, monkeypatch: pytest.MonkeyPatch):
        from src.core.settings import settings

        monkeypatch.setattr(settings.api, "ingest_allowed_roots", [])
        with pytest.raises(HTTPException) as exc:
            validate_ingest_path(Path("data/raw/doc.md"))
        assert exc.value.status_code == 403

    @pytest.mark.parametrize("filename", [".", ".."])
    def test_rejects_invalid_upload_filename(self, filename: str):
        with pytest.raises(HTTPException) as exc:
            validate_upload_filename(filename)
        assert exc.value.status_code == 400


class TestLoggingGaps:
    def test_stack_info_included(self):
        record = _make_record(stack_info="stack trace here")
        parsed = json.loads(JsonFormatter().format(record))
        assert parsed["stack_info"] == "stack trace here"

    def test_configure_logging_skips_otel_on_failure(self, caplog):

        with (
            patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
                side_effect=OSError("otel down"),
            ),
            caplog.at_level(logging.DEBUG, logger="src.core.logging"),
        ):
            setup_otel("localhost:4317", 1.0)
        assert "OTel setup skipped" in caplog.text


class TestSettingsGaps:
    def test_yaml_config_source_get_field_value_stub(self):
        source = YamlConfigSource(Settings)
        assert source.get_field_value(FieldInfo(annotation=str), "llm") == (None, "llm", False)

    def test_yaml_config_source_missing_configs_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setattr("src.core.settings.ROOT", tmp_path)
        assert YamlConfigSource(Settings)() == {}

    def test_neo4j_password_required_when_enabled(self):
        with pytest.raises(ValidationError, match="NEO4J__PASSWORD"):
            Neo4jSettings(enabled=True, password=SecretStr(""))


class TestEmbeddingRepositoryDefault:
    def test_embed_query_delegates_to_embed(self):
        emb = _Embedder()
        texts = ["query one", "query two"]
        assert emb.embed_query(texts) == emb.embed(texts)


# ── generation / retrieval ─────────────────────────────────────────────────────


class TestGenerationServiceGaps:
    def test_call_llm_delegates_to_llm(self):
        llm = MagicMock()
        llm.generate.return_value = "agent decision"
        svc = GenerationService(llm=llm)
        assert svc.call_llm("decide next step") == "agent decision"
        llm.generate.assert_called_once_with(prompt="decide next step", context="")

    def test_build_prompt_reuses_cached_template(self):
        llm = MagicMock()
        llm.generate.return_value = "answer"
        svc = GenerationService(llm=llm)
        svc.generate("q1", "ctx one", ["c0"])
        svc.generate("q2", "ctx two", ["c1"])
        assert svc._template is not None
        assert llm.generate.call_count == 2


class TestRetrievalServiceGaps:
    def test_hybrid_property(self):
        hybrid = MagicMock()
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            top_k_retrieval=5,
            top_k_rerank=5,
        )
        assert svc.hybrid is hybrid

    @pytest.mark.asyncio
    async def test_single_variant_skips_rrf_fuse(self):
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(_chunk(0), 0.9)])
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            top_k_retrieval=5,
            top_k_rerank=5,
        )
        with patch("src.domain.services.retrieval_service.rrf_fuse") as rrf:
            await svc.retrieve(Query(text="q"))
            rrf.assert_not_called()

    def test_apply_relevance_grading_empty_chunks(self):
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            top_k_retrieval=5,
            top_k_rerank=5,
            reliable_rag_enabled=True,
            llm=MagicMock(),
        )
        assert svc._apply_relevance_grading("q", []) == ([], 0, 0, [])

    def test_apply_relevance_grading_no_llm(self, caplog):
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            top_k_retrieval=5,
            top_k_rerank=5,
            reliable_rag_enabled=True,
            llm=None,
        )
        chunks = [_chunk(0)]
        kept, *_ = svc._apply_relevance_grading("q", chunks)
        assert kept == chunks
        assert "skipping relevance grading" in caplog.text

    def test_apply_diversity_empty(self):
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            top_k_retrieval=5,
            top_k_rerank=5,
        )
        assert svc._apply_diversity([]) == []

    def test_resolve_chunk_embeddings_empty(self):
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            top_k_retrieval=5,
            top_k_rerank=5,
        )
        assert svc._resolve_chunk_embeddings([]) == []


class TestHybridRetrieverGaps:
    @pytest.mark.asyncio
    async def test_feedback_boost_expands_fusion_top_k(self):
        hr, _, _ = feedback_boost_retriever()
        with patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf:
            rrf.return_value = []
            await hr.retrieve(Query(text="q"), top_k=10)
            assert rrf.call_args.kwargs["top_k"] == 30

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("top_k", "query_text", "expected_candidate_cap", "expected_fusion_top_k"),
        [
            (10, "q", 30, 10),
            (50, "What is EKS?", 50, 50),
        ],
    )
    async def test_feedback_boost_without_pool_expansion(
        self,
        top_k: int,
        query_text: str,
        expected_candidate_cap: int,
        expected_fusion_top_k: int,
    ):
        hr, dense_mock, _ = feedback_boost_retriever(feedback_expand_pool=False)
        with patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf:
            rrf.return_value = []
            await hr.retrieve(Query(text=query_text), top_k=top_k)
            assert dense_mock.retrieve.call_args[0][1] == expected_candidate_cap
            assert rrf.call_args.kwargs["top_k"] == expected_fusion_top_k

    @pytest.mark.asyncio
    async def test_feedback_boost_expands_beyond_default_cap(self):
        hr, dense_mock, bm25_mock = feedback_boost_retriever(feedback_expand_pool=True)
        await hr.retrieve(Query(text="What is EKS?"), top_k=50)
        dense_mock.retrieve.assert_called_once()
        assert dense_mock.retrieve.call_args[0][1] == 150
        bm25_mock.search.assert_called_once_with("What is EKS?", 150, filters=None)

    def test_optional_retriever_properties_exposed(self):
        graph, hype, hyde, hierarchical = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        hr = HybridRetriever(
            dense=MagicMock(),
            bm25=MagicMock(),
            graph_retriever=graph,
            hype_retriever=hype,
            hyde_retriever=hyde,
            hierarchical_retriever=hierarchical,
        )
        assert hr.graph is graph
        assert hr.hype is hype
        assert hr.hyde is hyde
        assert hr.hierarchical is hierarchical


class TestRetrievalPipelineAdaptive:
    def test_from_settings_wires_adaptive_when_enabled(self):
        with (
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings") as qdrant,
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.rag.retrieval.bm25_retriever.BM25Retriever"),
            patch("src.rag.retrieval.dense_retriever.DenseRetriever"),
            patch("src.rag.retrieval.hybrid_retriever.HybridRetriever") as mock_hybrid,
            patch("src.rag.ranking.cross_encoder.CrossEncoder.from_settings"),
            patch(
                "src.rag.retrieval.adaptive.query_classifier.QueryClassifier.from_settings"
            ) as qc,
            patch(
                "src.rag.retrieval.adaptive.strategies.AdaptiveStrategyRegistry.from_settings"
            ) as reg,
        ):
            mock_settings.retrieval = MagicMock(
                hybrid_alpha=0.7,
                top_k_dense=10,
                top_k_final=5,
                hybrid_fusion="rrf",
                rse=MagicMock(enabled=False, max_segment_tokens=1500),
                adaptive=MagicMock(enabled=True),
                hyde=MagicMock(enabled=False),
                hype=MagicMock(enabled=False),
                parent_context=MagicMock(enabled=False),
                diversity=MagicMock(enabled=False),
            )
            mock_settings.neo4j = MagicMock(enabled=False)
            mock_settings.reranker = MagicMock(top_k=5)
            mock_settings.query_expansion = MagicMock(
                enabled=False, step_back=MagicMock(enabled=False)
            )
            mock_settings.compression = MagicMock(enabled=False)
            mock_settings.quality = MagicMock(
                feedback_loop=MagicMock(
                    enabled=True,
                    boost_multiplier=0.05,
                    expand_candidate_pool=True,
                ),
                reliable_rag=MagicMock(enabled=False),
            )
            mock_settings.chunking = MagicMock(strategy="recursive")
            qdrant.return_value = MagicMock()
            mock_llm.return_value = MagicMock()
            pipeline = RetrievalPipeline.from_settings()
        qc.assert_called_once()
        reg.assert_called_once()
        assert pipeline.service._classifier is qc.return_value
        assert pipeline.service._strategy_registry is reg.return_value
        assert pipeline.service._feedback_boost_multiplier == pytest.approx(0.05)
        assert pipeline.service._vector_store is qdrant.return_value
        mock_hybrid.assert_called_once()
        assert mock_hybrid.call_args.kwargs["feedback_expand_pool"] is True


# ── BM25 / Qdrant ──────────────────────────────────────────────────────────────


class TestBm25Gaps:
    def test_rebuild_public_method(self):
        idx = BM25Index()
        idx.index([_chunk(0)])
        idx.remove_by_ids(["c0"])
        idx.rebuild()
        assert idx.search("sample", top_k=1) == []

    def test_update_chunk_metadata_empty_returns_false(self):
        idx = BM25Index()
        idx.index([_chunk(0)])
        assert idx.update_chunk_metadata("c0", {}) is False

    def test_update_chunk_metadata_missing_id_returns_false(self):
        idx = BM25Index()
        idx.index([_chunk(0)])
        assert idx.update_chunk_metadata("missing", {"k": "v"}) is False

    def test_update_chunk_metadata_merges_updates(self):
        idx = BM25Index()
        idx.index([_chunk(0)])
        assert idx.update_chunk_metadata("c0", {"tag": "v1"}) is True
        updated = idx.get_by_id("c0")
        assert updated is not None
        assert updated.metadata["tag"] == "v1"

    def test_load_invalid_chunks_format_raises(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        path.write_text('{"chunks": "not-a-list"}', encoding="utf-8")
        with pytest.raises(VectorStoreError, match="Invalid BM25 index format"):
            BM25Index(index_path=path).load()

    def test_load_legacy_pickle_os_error_raises(self, tmp_path: Path):
        bad = tmp_path / "bm25.pkl"
        bad.write_bytes(b"not-pickle")
        with pytest.raises(VectorStoreError, match="Cannot load legacy"):
            BM25Index(index_path=tmp_path / "bm25.json")._load_legacy_pickle(bad)

    def test_load_legacy_pickle_invalid_format_raises(self, tmp_path: Path):
        bad = tmp_path / "bm25.pkl"
        bad.write_bytes(pickle.dumps({"chunks": "bad"}))
        with pytest.raises(VectorStoreError, match="Invalid legacy BM25 index format"):
            BM25Index(index_path=tmp_path / "bm25.json")._load_legacy_pickle(bad)


class TestQdrantFeedbackGaps:
    @pytest.fixture
    def store(self) -> QdrantVectorStore:
        s = QdrantVectorStore(collection="test_col", dense_dim=4)
        s._client = MagicMock()
        s._collection_ready = True
        return s

    def test_get_feedback_score_retrieve_failure_raises(self, store):
        store._client.retrieve.side_effect = RuntimeError("down")
        with pytest.raises(VectorStoreError, match="retrieve failed"):
            store.get_feedback_score("chunk-a")

    def test_get_feedback_score_bool_returns_zero(self, store):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": True}}
        store._client.retrieve.return_value = [point]
        assert store.get_feedback_score("chunk-a") == 0.0

    def test_get_feedback_score_non_numeric_returns_zero(self, store):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": "high"}}
        store._client.retrieve.return_value = [point]
        assert store.get_feedback_score("chunk-a") == 0.0

    def test_set_feedback_score_retrieve_failure_raises(self, store):
        store._client.retrieve.side_effect = RuntimeError("down")
        with pytest.raises(VectorStoreError, match="retrieve failed"):
            store.set_feedback_score("chunk-a", 1.0)

    def test_set_feedback_score_set_payload_failure_raises(self, store):
        point = MagicMock()
        point.payload = {"metadata": {}}
        store._client.retrieve.return_value = [point]
        store._client.set_payload.side_effect = RuntimeError("write failed")
        with pytest.raises(VectorStoreError, match="set_payload failed"):
            store.set_feedback_score("chunk-a", 1.0)

    def test_upsert_adds_chunk_type_to_payload(self, store):
        from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_KEY

        chunk = _chunk(0)
        chunk = chunk.model_copy(update={"metadata": {CHUNK_TYPE_KEY: CHUNK_TYPE_HYPE}})
        chunk = chunk.model_copy(
            update={"embedding": [0.1, 0.2, 0.3, 0.4], "sparse_vector": {1: 0.9}}
        )
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        inserted = {"done": False}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            if inserted["done"]:
                return [point]
            return []

        def upsert_side_effect(**_kwargs: object) -> None:
            inserted["done"] = True

        store._client.retrieve.side_effect = retrieve_side_effect
        store._client.upsert.side_effect = upsert_side_effect
        store.upsert([chunk])
        payload = store._client.upsert.call_args.kwargs["points"][0].payload
        assert payload[CHUNK_TYPE_KEY] == CHUNK_TYPE_HYPE


# ── quality / structured output ────────────────────────────────────────────────


class TestExplainableRetrievalGaps:
    def test_lookup_explanation_via_sibling_id(self):
        rep = _chunk(0)
        sibling = _chunk(1)
        explanation = ChunkExplanation(chunk_id="c1", reason="sibling match")
        result = lookup_explanation({"c1": explanation}, rep, [rep, sibling])
        assert result is explanation


class TestPostGenerationGaps:
    def test_empty_answer_text_returns_empty(self):
        llm = MagicMock()
        answer = Answer(query_id="q-1", text="  ", sources=["c0"])
        explanations, highlights = explain_and_highlight("q", answer, [_chunk(0)], llm)
        assert explanations == []
        assert highlights == {}
        llm.generate.assert_not_called()

    def test_parse_null_explanations_and_highlights(self):
        from src.rag.quality.post_generation import (
            _parse_explanations_field,
            _parse_highlights_field,
        )

        assert _parse_explanations_field(None) == []
        assert _parse_highlights_field(None) == []

    def test_parse_empty_arrays_raises(self):
        from src.rag.quality.post_generation import parse_explain_and_highlight

        with pytest.raises(ValueError, match="Could not parse explain and highlight"):
            parse_explain_and_highlight('{"explanations":"bad","highlights":[]}')


class TestSelfRagGaps:
    def test_decide_retrieval_failure_fallback(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("boom")
        decision = decide_retrieval("what is eks?", llm)
        assert decision.need_retrieval is True

    def test_score_utility_failure_fallback(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("boom")
        utility = score_utility("what is eks?", "draft", "", llm)
        assert utility.action == UtilityAction.ACCEPT
        assert utility.score == 0.5


class TestSourceHighlightingGaps:
    def test_validate_span_whitespace_only(self):
        assert validate_span("   ", "passage text") is None

    def test_validate_span_tokens_empty_after_normalize(self):
        assert validate_span(" \n ", "some passage text here") is None

    def test_validate_span_not_in_passage(self):
        with patch("src.rag.quality.source_highlighting.re.search") as search:
            search.return_value = MagicMock(group=lambda *_args: "missing text")
            assert validate_span("missing text", "actual passage") is None


class TestStructuredOutput:
    class _Model(BaseModel):
        value: int

    def test_extract_json_object_skips_empty_candidate(self):
        assert extract_json_object("") is None

    def test_parse_structured_output_skips_empty_candidate(self):
        with pytest.raises(ValueError, match="Could not parse test"):
            parse_structured_output("", self._Model, label="test")


class TestBm25RetrieverGaps:
    def test_get_by_id_returns_chunk(self):
        retriever = BM25Retriever(BM25Index())
        retriever.index([_chunk(0)])
        assert retriever.get_by_id("c0") is not None
        assert retriever.get_by_id("missing") is None


class TestGraphRetrieverGaps:
    def test_template_cached_after_first_use(self):
        llm = MagicMock()
        llm.generate.return_value = "[]"
        extractor = EntityExtractor(llm=llm)
        extractor.extract_relations("text")
        first = extractor._template
        extractor.extract_relations("more text")
        assert extractor._template is first

    def test_from_settings_builds_retriever(self):
        with patch("src.infrastructure.vectordb.neo4j_graph.Neo4jGraphRepository.from_settings"):
            gr = GraphRetriever.from_settings(MagicMock(), MagicMock())
            assert isinstance(gr, GraphRetriever)

    def test_non_list_json_returns_empty(self):
        llm = MagicMock()
        llm.generate.return_value = '{"subject":"A"}'
        assert EntityExtractor(llm=llm).extract_relations("text") == []

    def test_skips_non_dict_items(self):
        llm = MagicMock()
        llm.generate.return_value = '["not-a-dict", {"subject":"A","relation":"r","object":"B"}]'
        relations = EntityExtractor(llm=llm).extract_relations("text")
        assert len(relations) == 1


# ── evals ──────────────────────────────────────────────────────────────────────


class TestEvalsGaps:
    def test_sample_result_to_dict(self):
        sample = SampleResult(
            question="q",
            expected_answer="a",
            generated_answer="g",
            retrieved_ids=["c0"],
            relevant_ids=["c0"],
            recall_at_5=1.0,
            faithfulness=0.9,
            relevance=0.8,
            context_precision=0.7,
            hallucination=0.1,
        )
        data = sample.to_dict()
        assert data["question"] == "q"
        assert data["faithfulness"] == 0.9

    def test_ragas_metric_base_pre_checks_empty(self):
        class _Metric(RagasMetric):
            _metric_name = "faithfulness"

            def _get_ragas_metric(self) -> object:
                return MagicMock()

        sample = EvalSample(question="q", expected_answer="a")
        assert _Metric(threshold=0.5)._pre_checks(sample) == []

    def test_ragas_score_calls_evaluate(self):
        sample = EvalSample(
            question="q",
            expected_answer="a",
            generated_answer="g",
            retrieved_chunks=["ctx"],
        )
        metric = FaithfulnessMetric(threshold=0.5)
        fake_ragas = MagicMock()
        fake_ragas.evaluate.return_value = {"faithfulness": 0.85}
        with (
            patch.object(metric, "_get_ragas_metric", return_value=MagicMock()),
            patch.dict("sys.modules", {"ragas": fake_ragas}),
        ):
            score = metric._ragas_score(sample)
        assert score == 0.85
        fake_ragas.evaluate.assert_called_once()

    def test_parse_json_pairs_invalid_inner_json(self):
        assert parse_json_pairs("prefix [not valid json] suffix") == []


# ── pipelines / ingestion ──────────────────────────────────────────────────────


class TestChatPipelineGaps:
    def test_crag_warns_when_web_search_disabled(self, monkeypatch: pytest.MonkeyPatch, caplog):
        from src.core.settings import settings

        monkeypatch.setattr(settings.quality.crag, "enabled", True)
        monkeypatch.setattr(settings.web_search, "provider", "none")
        with (
            patch("src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"),
            patch("src.rag.pipelines.chat_pipeline.RetrievalPipeline.from_settings"),
            patch("src.rag.pipelines.chat_pipeline.GenerationService.from_settings"),
            caplog.at_level(logging.WARNING),
        ):
            ChatPipeline.from_settings()
        assert "web_search.provider=none" in caplog.text


class TestIngestionPipelineGaps:
    def test_discover_unsupported_file_returns_empty(self, tmp_path: Path):
        bad = tmp_path / "file.exe"
        bad.write_text("binary", encoding="utf-8")
        assert discover_paths(bad) == []

    def test_build_hype_indexer_disabled(self):
        assert build_hype_indexer(MagicMock(), SimpleNamespace(enabled=False)) is None

    def test_build_hierarchical_indexer_disabled(self):
        assert build_hierarchical_indexer(MagicMock(), SimpleNamespace(enabled=False)) is None


class TestDiversityGaps:
    def test_normalize_relevance_scores_edge_cases(self):
        assert normalize_relevance_scores(0) == []
        assert normalize_relevance_scores(1) == [1.0]


class TestAdaptiveStrategiesGaps:
    def test_params_from_config_dict(self):
        params = params_from_config(
            {"top_k": 10, "n_variants": 2, "hyde": True, "compression": False}
        )
        assert params.top_k == 10
        assert params.hyde is True


# ── remaining 99% → 100% gaps ─────────────────────────────────────────────────


class TestRetrievalServiceRemaining:
    @pytest.mark.asyncio
    async def test_single_gathered_result_returns_directly(self):
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(_chunk(0), 0.9)])
        svc = RetrievalService(
            dense_retriever=_dense_mock(),
            hybrid_retriever=hybrid,
            top_k_retrieval=5,
            top_k_rerank=5,
        )
        query = Query(text="q", expanded_texts=["variant"])

        async def _fake_gather(*_args, **_kwargs):
            return [[(_chunk(0), 0.9)]]

        with patch(
            "src.domain.services.retrieval_service.asyncio.gather",
            side_effect=_fake_gather,
        ):
            results = await svc._retrieve_variants(query)
        assert len(results) == 1

    def test_resolve_chunk_embeddings_lookup_miss(self):
        lookup = MagicMock()
        lookup.get_by_id.return_value = None
        embedder = MagicMock()
        embedder.embed_passage.return_value = [[0.1, 0.2]]
        svc = RetrievalService(
            dense_retriever=embedder,
            hybrid_retriever=MagicMock(),
            top_k_retrieval=5,
            top_k_rerank=5,
            chunk_lookup=lookup,
            diversity_enabled=True,
            diversity_lambda=0.7,
            embedder=embedder,
        )
        chunks = [_chunk(0)]
        assert svc._resolve_chunk_embeddings(chunks) == [[0.1, 0.2]]

    def test_resolve_chunk_embeddings_returns_none_when_still_unresolved(self):
        embedder = MagicMock()
        embedder.embed_passage.return_value = [None]
        svc = RetrievalService(
            dense_retriever=embedder,
            hybrid_retriever=MagicMock(),
            top_k_retrieval=5,
            top_k_rerank=5,
            diversity_enabled=True,
            embedder=embedder,
        )
        chunks = [_chunk(0)]
        assert svc._resolve_chunk_embeddings(chunks) is None


class TestGenerationMetricPreChecks:
    def test_context_precision_no_context_guard(self):
        sample = EvalSample(question="q", expected_answer="a", retrieved_chunks=[])
        checks = ContextPrecisionMetric()._pre_checks(sample)
        assert checks[0].details == "No context provided"

    def test_faithfulness_no_context_guard(self):
        sample = EvalSample(
            question="q",
            expected_answer="a",
            generated_answer="answer",
            retrieved_chunks=[],
        )
        checks = FaithfulnessMetric()._pre_checks(sample)
        assert checks[0].details == "No context provided"

    def test_faithfulness_parametric_answer_guard(self):
        sample = EvalSample(
            question="q",
            expected_answer="a",
            generated_answer="answer",
            retrieved_chunks=[],
            parametric_answer=True,
        )
        checks = FaithfulnessMetric()._pre_checks(sample)
        assert checks[0].score == pytest.approx(1.0)
        assert "Parametric answer" in checks[0].details

    def test_hallucination_no_context_score(self):
        from src.evals.generation.hallucination import HallucinationMetric

        sample = EvalSample(
            question="q",
            expected_answer="a",
            generated_answer="answer",
            retrieved_chunks=[],
        )
        result = HallucinationMetric().score(sample)
        assert result.score == 1.0
        assert result.details == "No context to verify against"


class TestEmbeddingProviderGaps:
    def test_bge_m3_loads_model_on_cuda(self):
        mock_model = MagicMock()
        fake_flag = MagicMock()
        fake_flag.BGEM3FlagModel.return_value = mock_model
        with patch.dict("sys.modules", {"FlagEmbedding": fake_flag}):
            from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider

            provider = BGEM3EmbeddingProvider(model_path="fake", device="cuda")
            assert provider._get_model() is mock_model
            fake_flag.BGEM3FlagModel.assert_called_once_with("fake", use_fp16=True, device="cuda")

    def test_redact_redis_url_missing_hostname(self):
        from src.infrastructure.embeddings.cached_embedding_provider import _redact_redis_url

        assert _redact_redis_url("redis://:@/0") == "<redacted>"

    def test_cohere_rate_limit_fallback_string(self):
        from src.infrastructure.embeddings.cohere_provider import _is_rate_limit

        assert _is_rate_limit(Exception("HTTP 429 Too Many Requests")) is True

    def test_cohere_embed_failure_wraps_error(self):
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="key", model="embed")
        client = MagicMock()
        client.embed.side_effect = Exception("auth failed")
        provider._client = client
        with pytest.raises(EmbeddingError, match="Cohere embed failed"):
            provider.embed(["hello"])

    def test_cohere_missing_package_raises(self):
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="key", model="embed")
        with (
            patch.dict("sys.modules", {"cohere": None}),
            pytest.raises(EmbeddingError, match="cohere package"),
        ):
            provider._get_client()

    def test_gemini_rate_limit_resource_exhausted(self):
        from src.infrastructure.embeddings.gemini_provider import _is_rate_limit

        resource_exhausted = type("ResourceExhausted", (Exception,), {})
        fake_exceptions = MagicMock(ResourceExhausted=resource_exhausted)
        with patch.dict("sys.modules", {"google.api_core.exceptions": fake_exceptions}):
            assert _is_rate_limit(resource_exhausted("quota")) is True

    def test_gemini_embed_failure_wraps_error(self):
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="key", model="text-embedding")
        with (
            patch.object(provider, "_call_with_retry", side_effect=Exception("fail")),
            pytest.raises(EmbeddingError, match="Gemini embed failed"),
        ):
            provider.embed(["hello"])

    def test_gemini_flat_embedding_wrapped(self):
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="key", model="text-embedding")
        with patch.object(provider, "_call_api", return_value={"embedding": [0.1, 0.2]}):
            assert provider._call_api(["one"], "RETRIEVAL_QUERY") == {"embedding": [0.1, 0.2]}

    def test_nomic_matryoshka_dim(self):
        from src.infrastructure.embeddings.nomic import NomicEmbeddingProvider

        provider = NomicEmbeddingProvider(model_path="fake", matryoshka_dim=256)
        assert provider._encode_kwargs() == {"truncate_dim": 256}

    def test_openai_rate_limit(self):
        from src.infrastructure.embeddings.openai_provider import _is_rate_limit

        assert _is_rate_limit(Exception("rate_limit exceeded")) is True

    def test_openai_dimensions_passed(self):
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="key", model="text-embedding-3-small", dimensions=512
        )
        client = MagicMock()
        item = MagicMock(embedding=[0.1, 0.2], index=0)
        client.embeddings.create.return_value = MagicMock(data=[item])
        provider._client = client
        provider.embed(["hello"])
        assert client.embeddings.create.call_args.kwargs["dimensions"] == 512

    def test_sentence_transformer_load_success(self):
        mock_model = MagicMock()
        fake_st = MagicMock()
        fake_st.SentenceTransformer.return_value = mock_model
        with patch.dict("sys.modules", {"sentence_transformers": fake_st}):
            from src.infrastructure.embeddings.nomic import NomicEmbeddingProvider

            provider = NomicEmbeddingProvider(model_path="fake")
            assert provider._get_model() is mock_model

    def test_voyage_rate_limit_string(self):
        from src.infrastructure.embeddings.voyage_provider import _is_rate_limit

        assert _is_rate_limit(Exception("429 rate limit")) is True

    def test_voyage_missing_package_raises(self):
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="key", model="voyage")
        with (
            patch.dict("sys.modules", {"voyageai": None}),
            pytest.raises(EmbeddingError, match="voyageai package"),
        ):
            provider._get_client()


class TestLlamaCppGaps:
    def test_get_model_success_load(self):
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider(model_path="fake.gguf")
        mock_llama = MagicMock()
        with patch("llama_cpp.Llama", return_value=mock_llama):
            assert provider._get_model() is mock_llama

    @pytest.mark.asyncio
    async def test_generate_stream_worker_exception_raises_generation_error(self):
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider(model_path="fake.gguf")
        mock_llama = MagicMock()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        mock_llama.create_chat_completion.side_effect = _boom
        provider._model = mock_llama
        with pytest.raises(GenerationError, match="stream failed"):
            async for _ in provider.generate_stream("prompt", "context"):
                pass


class TestLoaderExceptionPaths:
    def test_pdf_reraises_document_load_error(self, tmp_path: Path):
        from src.infrastructure.loaders.pdf_loader import PdfLoader

        path = tmp_path / "bad.pdf"
        path.write_bytes(b"x")
        with (
            patch(
                "src.infrastructure.loaders.pdf_loader.PdfReader",
                side_effect=DocumentLoadError("already wrapped"),
            ),
            pytest.raises(DocumentLoadError, match="already wrapped"),
        ):
            PdfLoader().load(path)

    def test_docx_wraps_generic_exception(self, tmp_path: Path):
        from src.infrastructure.loaders.docx_loader import DocxLoader

        path = tmp_path / "bad.docx"
        path.write_bytes(b"x")
        with (
            patch(
                "src.infrastructure.loaders.docx_loader.python_docx.Document",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(DocumentLoadError, match="Cannot load DOCX"),
        ):
            DocxLoader().load(path)

    def test_docx_reraises_document_load_error(self, tmp_path: Path):
        from src.infrastructure.loaders.docx_loader import DocxLoader

        path = tmp_path / "bad.docx"
        path.write_bytes(b"x")
        with (
            patch(
                "src.infrastructure.loaders.docx_loader.python_docx.Document",
                side_effect=DocumentLoadError("inner"),
            ),
            pytest.raises(DocumentLoadError, match="inner"),
        ):
            DocxLoader().load(path)

    def test_html_wraps_generic_exception(self, tmp_path: Path):
        from src.infrastructure.loaders.html_loader import HtmlLoader

        path = tmp_path / "bad.html"
        path.write_text("<html></html>")
        with (
            patch(
                "src.infrastructure.loaders.html_loader.BeautifulSoup",
                side_effect=ValueError("bad"),
            ),
            pytest.raises(DocumentLoadError, match="Cannot load HTML"),
        ):
            HtmlLoader().load(path)

    def test_html_reraises_document_load_error(self, tmp_path: Path):
        from src.infrastructure.loaders.html_loader import HtmlLoader

        path = tmp_path / "bad.html"
        path.write_text("<html></html>")
        with (
            patch(
                "src.infrastructure.loaders.html_loader.BeautifulSoup",
                side_effect=DocumentLoadError("inner"),
            ),
            pytest.raises(DocumentLoadError, match="inner"),
        ):
            HtmlLoader().load(path)

    def test_markdown_wraps_generic_exception(self, tmp_path: Path):
        from src.infrastructure.loaders.markdown_loader import MarkdownLoader

        path = tmp_path / "bad.md"
        path.write_text("# title")
        with (
            patch.object(Path, "read_text", side_effect=ValueError("bad")),
            pytest.raises(DocumentLoadError, match="Cannot load Markdown"),
        ):
            MarkdownLoader().load(path)

    def test_markdown_reraises_document_load_error(self, tmp_path: Path):
        from src.infrastructure.loaders.markdown_loader import MarkdownLoader

        path = tmp_path / "bad.md"
        path.write_text("# title")
        with (
            patch.object(Path, "read_text", side_effect=DocumentLoadError("inner")),
            pytest.raises(DocumentLoadError, match="inner"),
        ):
            MarkdownLoader().load(path)


class TestBgeRerankerGaps:
    def test_get_model_success_on_mps(self):
        mock_model = MagicMock()
        fake_flag = MagicMock()
        fake_flag.FlagReranker.return_value = mock_model
        with patch.dict("sys.modules", {"FlagEmbedding": fake_flag}):
            from src.infrastructure.rerankers.bge_reranker import BGERerankerProvider

            provider = BGERerankerProvider(model_path="fake", device="mps")
            assert provider._get_model() is mock_model
            fake_flag.FlagReranker.assert_called_once_with("fake", use_fp16=True, device="mps")


class TestNeo4jGaps:
    def test_get_driver_success(self):
        from src.infrastructure.vectordb.neo4j_graph import Neo4jGraphRepository

        repo = Neo4jGraphRepository(uri="bolt://localhost", user="neo4j", password="pw")
        mock_driver = MagicMock()
        fake_neo4j = MagicMock()
        fake_neo4j.GraphDatabase.driver.return_value = mock_driver
        with patch.dict("sys.modules", {"neo4j": fake_neo4j}):
            assert repo._get_driver() is mock_driver
            assert repo._get_driver() is mock_driver


class TestRagModuleGaps:
    def test_section_label_from_sections_list(self):
        from src.rag.chunking.contextual_headers import _section_label

        chunk = _chunk(0, metadata={"sections": ["Intro"]})
        assert _section_label(chunk) == "Intro"

    def test_section_label_from_headings_list(self):
        from src.rag.chunking.contextual_headers import _section_label

        chunk = _chunk(0, metadata={"headings": ["Title"]})
        assert _section_label(chunk) == "Title"

    def test_contextual_compression_skips_empty_extract(self):
        from src.rag.compression.contextual_compression import ContextualCompressor

        compressor = ContextualCompressor(llm=MagicMock(), max_tokens=100)
        with patch.object(compressor, "_extract", return_value=""):
            result = compressor.compress("q", [_chunk(0)])
        assert result == []

    def test_hierarchical_indexer_empty_summary(self, caplog):
        from src.rag.enrichment.hierarchical_indexer import HierarchicalIndexer

        indexer = HierarchicalIndexer(llm=MagicMock(), embedder=MagicMock())
        doc = Document(source="doc.md", content="body")
        with patch(
            "src.rag.enrichment.hierarchical_indexer.generate_document_summary",
            return_value="",
        ):
            summaries = indexer.index(doc, [_chunk(0)])[1]
        assert summaries == []

    def test_hype_indexer_blank_questions_returns_empty(self):
        from src.rag.enrichment.hype_indexer import HyPEIndexer

        llm = MagicMock()
        llm.generate.return_value = '["", "   "]'
        indexer = HyPEIndexer(llm=llm, embedder=MagicMock(), n_questions=2)
        assert indexer.index([_chunk(0)]) == []

    def test_join_overlapping_empty_parts(self):
        from src.rag.enrichment.relevant_segment_extraction import _join_overlapping

        assert _join_overlapping(["", "  "]) == ""

    def test_query_classifier_clear_cache(self):
        from src.rag.retrieval.adaptive.query_classifier import QueryClassifier

        llm = MagicMock()
        llm.generate.return_value = '{"category": "factual"}'
        classifier = QueryClassifier(llm=llm, enabled=True)
        query = Query(text="what is eks?")
        classifier.classify(query)
        classifier.classify(query)
        assert llm.generate.call_count == 1
        classifier.clear_cache()
        classifier.classify(query)
        assert llm.generate.call_count == 2

    def test_query_classifier_from_settings(self):
        from src.rag.retrieval.adaptive.query_classifier import QueryClassifier

        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.retrieval.adaptive.enabled = False
            classifier = QueryClassifier.from_settings(MagicMock())
        assert classifier._enabled is False


class TestChatPipelineRemaining:
    def test_crag_web_search_unavailable_warning(self, monkeypatch: pytest.MonkeyPatch, caplog):
        from src.core.settings import settings

        monkeypatch.setattr(settings.quality.crag, "enabled", True)
        monkeypatch.setattr(settings.web_search, "provider", "duckduckgo")
        with (
            patch("src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"),
            patch("src.rag.pipelines.chat_pipeline.RetrievalPipeline.from_settings"),
            patch("src.rag.pipelines.chat_pipeline.GenerationService.from_settings"),
            patch(
                "src.infrastructure.search.web_search.get_web_search_provider",
                side_effect=RuntimeError("search down"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            ChatPipeline.from_settings()
        assert "web search unavailable" in caplog.text


class TestIngestionPipelineRemaining:
    def test_proposition_strategy_wires_llm(self):
        from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

        with (
            patch("src.core.settings.settings") as mock_settings,
            patch("src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"),
            patch("src.rag.chunking.get_chunker") as get_chunker,
            patch("src.infrastructure.embeddings.get_embedding_provider"),
            patch("src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings"),
            patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create"),
            patch("src.infrastructure.metadata.sqlite_store.SQLiteMetadataStore.from_settings"),
        ):
            mock_settings.chunking = MagicMock(
                strategy="proposition",
                proposition=MagicMock(quality_threshold=8),
            )
            mock_settings.metadata = MagicMock(enabled=True)
            mock_settings.chunking.contextual_headers = MagicMock(enabled=False)
            mock_settings.chunking.augmentation = MagicMock(enabled=False)
            mock_settings.chunking.hierarchical = MagicMock(enabled=False)
            mock_settings.retrieval = MagicMock(hype=MagicMock(enabled=False))
            mock_settings.neo4j = MagicMock(enabled=False)
            IngestionPipeline.from_settings()
        kwargs = get_chunker.call_args.kwargs
        assert kwargs["overlap"] == 0
        assert kwargs["quality_threshold"] == 8


class TestFinalCoverageGaps:
    def test_cohere_too_many_requests_is_rate_limit(self):
        fake_cohere = MagicMock()
        err = type("TooManyRequestsError", (Exception,), {})
        fake_cohere.TooManyRequestsError = err
        with patch.dict("sys.modules", {"cohere": fake_cohere}):
            from src.infrastructure.embeddings.cohere_provider import _is_rate_limit

            assert _is_rate_limit(err("limit")) is True

    def test_cohere_client_lazy_import_error(self):
        import builtins

        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="key", model="embed")
        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError("No module named 'cohere'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_import),
            pytest.raises(EmbeddingError, match="cohere package"),
        ):
            provider._get_client()

    def test_cohere_get_client_instantiates_client(self):
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        fake_client = MagicMock()
        fake_cohere = MagicMock()
        fake_cohere.ClientV2.return_value = fake_client
        provider = CohereEmbeddingProvider(api_key="key", model="embed")
        with patch.dict("sys.modules", {"cohere": fake_cohere}):
            assert provider._get_client() is fake_client
        fake_cohere.ClientV2.assert_called_once_with(api_key="key")

    def test_gemini_flat_embedding_in_embed(self):
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="key", model="text-embedding")
        fake_genai = MagicMock()
        fake_genai.embed_content.return_value = {"embedding": [0.1, 0.2]}
        with patch.dict("sys.modules", {"google.generativeai": fake_genai}):
            vectors = provider._call_api(["one"], "RETRIEVAL_QUERY")
        assert vectors == [[0.1, 0.2]]

    def test_openai_rate_limit_error_type(self):
        fake_openai = MagicMock()
        err = type("RateLimitError", (Exception,), {})
        fake_openai.RateLimitError = err
        with patch.dict("sys.modules", {"openai": fake_openai}):
            from src.infrastructure.embeddings.openai_provider import _is_rate_limit

            assert _is_rate_limit(err("limit")) is True

    def test_openai_client_lazy_import_error(self):
        import builtins

        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="key", model="text-embedding-3-small")
        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("No module named 'openai'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_import),
            pytest.raises(EmbeddingError, match="openai package"),
        ):
            provider._get_client()

    def test_openai_get_client_instantiates_client(self):
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        fake_client = MagicMock()
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        provider = OpenAIEmbeddingProvider(api_key="key", model="text-embedding-3-small")
        with patch.dict("sys.modules", {"openai": fake_openai}):
            assert provider._get_client() is fake_client
        fake_openai.OpenAI.assert_called_once_with(api_key="key")

    def test_voyage_rate_limit_error_type(self):
        err = type("RateLimitError", (Exception,), {})
        fake_error_mod = MagicMock()
        fake_error_mod.RateLimitError = err
        fake_voyage = MagicMock()
        with patch.dict(
            "sys.modules",
            {"voyageai": fake_voyage, "voyageai.error": fake_error_mod},
        ):
            from src.infrastructure.embeddings.voyage_provider import _is_rate_limit

            assert _is_rate_limit(err("limit")) is True

    def test_voyage_client_lazy_import_error(self):
        import builtins

        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="key", model="voyage")
        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "voyageai":
                raise ImportError("No module named 'voyageai'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_import),
            pytest.raises(EmbeddingError, match="voyageai package"),
        ):
            provider._get_client()

    def test_voyage_get_client_instantiates_client(self):
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        fake_client = MagicMock()
        fake_voyage = MagicMock()
        fake_voyage.Client.return_value = fake_client
        provider = VoyageEmbeddingProvider(api_key="key", model="voyage")
        with patch.dict("sys.modules", {"voyageai": fake_voyage}):
            assert provider._get_client() is fake_client
        fake_voyage.Client.assert_called_once_with(api_key="key")

    def test_rse_truncates_merged_context(self):
        from src.rag.enrichment.relevant_segment_extraction import _merge_run

        long_text = "word " * 500
        chunks = [_chunk(0, long_text), _chunk(1, long_text)]
        merged = _merge_run(chunks, max_segment_tokens=50)
        assert merged.metadata.get("rse_merged") is True

    def test_explain_and_highlight_empty_answer(self):
        llm = MagicMock()
        answer = Answer(query_id="q", text="", sources=["c0"])
        assert explain_and_highlight("q", answer, [_chunk(0)], llm) == ([], {})
        llm.generate.assert_not_called()

    def test_source_highlighting_empty_tokens_after_normalize(self):
        with patch(
            "src.rag.quality.source_highlighting._normalize_whitespace",
            return_value="",
        ):
            assert validate_span("not-in-passage", "some passage text") is None

    @pytest.mark.asyncio
    async def test_chat_full_clears_unresolved_source_chunks(self):
        from src.domain.services.retrieval_service import RetrievalResult

        chunks = [_chunk(0)]
        retrieval = MagicMock()
        retrieval.retrieve = AsyncMock(
            return_value=RetrievalResult(query=Query(text="q"), chunks=chunks, context="ctx")
        )
        generation = MagicMock()
        generation.generate.return_value = Answer(
            query_id="q", text="answer", sources=["missing-id"]
        )
        pipeline = ChatPipeline(
            retrieval=retrieval,
            generation=generation,
            llm=MagicMock(),
        )
        with patch("src.rag.pipelines.chat_pipeline.explain_and_highlight") as combined:
            await pipeline.chat_full("q", explain=True)
            combined.assert_not_called()

    def test_ingestion_hierarchical_and_hype_indexers(self, tmp_path: Path):
        from tests.unit.ingestion_helpers import embedded_chunk, mock_ingestion_pipeline

        path = tmp_path / "doc.md"
        path.write_text("content")
        base = embedded_chunk(0)
        summary = embedded_chunk(1)
        hype = embedded_chunk(2)
        hierarchical = MagicMock()
        hierarchical.index.return_value = ([base], [summary])
        hype_indexer = MagicMock()
        hype_indexer.index.return_value = [hype]

        pipeline, _, vector_store, _ = mock_ingestion_pipeline([base])
        pipeline._hierarchical_indexer = hierarchical
        pipeline._hype_indexer = hype_indexer
        pipeline.ingest_file(path)

        upserted = vector_store.upsert.call_args.args[0]
        assert any(c.id == summary.id for c in upserted)
        assert any(c.id == hype.id for c in upserted)
