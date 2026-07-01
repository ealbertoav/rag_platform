import pytest
from pydantic import ValidationError

from src.core.settings import (
    APISettings,
    ChunkingSettings,
    CompressionSettings,
    EmbeddingSettings,
    LLMSettings,
    LoggingSettings,
    MetadataSettings,
    Neo4jSettings,
    QdrantSettings,
    QueryExpansionSettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
    settings,
)


class TestSettingsSingleton:
    def test_importable(self):
        assert settings is not None

    def test_has_all_sections(self):
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.embeddings, EmbeddingSettings)
        assert isinstance(settings.reranker, RerankerSettings)
        assert isinstance(settings.qdrant, QdrantSettings)
        assert isinstance(settings.retrieval, RetrievalSettings)
        assert isinstance(settings.query_expansion, QueryExpansionSettings)
        assert isinstance(settings.compression, CompressionSettings)
        assert isinstance(settings.api, APISettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.neo4j, Neo4jSettings)
        assert isinstance(settings.metadata, MetadataSettings)
        assert isinstance(settings.chunking, ChunkingSettings)


class TestYamlDefaults:
    def test_llm_defaults_from_yaml(self):
        assert settings.llm.provider == "llama_cpp"
        assert settings.llm.context_size == 32768
        assert settings.llm.temperature == 0.1
        assert settings.llm.max_tokens == 2048
        assert "<|im_end|>" in settings.llm.stop_tokens

    def test_embedding_defaults_from_yaml(self):
        assert settings.embeddings.provider == "bge_m3"
        assert settings.embeddings.batch_size == 32
        assert settings.embeddings.dense_dim == 1024
        assert settings.embeddings.sparse_dim == 30522

    def test_retrieval_defaults_from_yaml(self):
        assert settings.retrieval.top_k_dense == 50
        assert settings.retrieval.top_k_final == 5
        assert settings.retrieval.hybrid_alpha == pytest.approx(0.7)
        assert settings.retrieval.hybrid_fusion == "rrf"
        assert settings.retrieval.rse.enabled is False
        assert settings.retrieval.rse.max_segment_tokens == 1500

    def test_neo4j_defaults_from_yaml(self):
        assert settings.neo4j.enabled is False

    def test_metadata_defaults_from_yaml(self):
        assert settings.metadata.enabled is True

    def test_reranker_defaults_from_yaml(self):
        assert settings.reranker.provider == "bge_reranker"
        assert settings.reranker.top_k == 10

    def test_query_expansion_defaults_from_yaml(self):
        assert settings.query_expansion.enabled is True
        assert settings.query_expansion.n_variants == 3
        assert settings.query_expansion.step_back.enabled is False

    def test_compression_defaults_from_yaml(self):
        assert settings.compression.enabled is True
        assert settings.compression.max_tokens == 1500

    def test_api_defaults_from_yaml(self):
        assert settings.api.port == 8000

    def test_logging_defaults_from_yaml(self):
        assert settings.logging.level == "INFO"
        assert settings.logging.format == "json"

    def test_contextual_headers_defaults_from_yaml(self):
        assert settings.chunking.contextual_headers.enabled is False
        assert settings.chunking.contextual_headers.exclude_from_llm_context is True

    def test_augmentation_defaults_from_yaml(self):
        assert settings.chunking.augmentation.enabled is False
        assert settings.chunking.augmentation.n_questions == 3

    def test_hype_defaults_from_yaml(self):
        assert settings.retrieval.hype.enabled is False
        assert settings.retrieval.hype.n_questions == 3

    def test_hyde_defaults_from_yaml(self):
        assert settings.retrieval.hyde.enabled is False

    def test_adaptive_defaults_from_yaml(self):
        assert settings.retrieval.adaptive.enabled is False
        factual = settings.retrieval.adaptive.strategies["factual"]
        assert factual.top_k == 30
        assert factual.n_variants == 1
        analytical = settings.retrieval.adaptive.strategies["analytical"]
        assert analytical.top_k == 50
        assert analytical.hyde is True

    def test_rse_defaults_from_yaml(self):
        assert settings.retrieval.rse.enabled is False
        assert settings.retrieval.rse.max_segment_tokens == 1500

    def test_parent_context_defaults_from_yaml(self):
        assert settings.retrieval.parent_context.enabled is False

    def test_diversity_defaults_from_yaml(self):
        assert settings.retrieval.diversity.enabled is False
        assert settings.retrieval.diversity.lambda_ == pytest.approx(0.7)

    def test_reliable_rag_defaults_from_yaml(self):
        assert settings.quality.reliable_rag.enabled is False
        assert settings.quality.reliable_rag.min_score == pytest.approx(0.5)

    def test_hierarchical_defaults_from_yaml(self):
        assert settings.chunking.hierarchical.enabled is False
        assert settings.chunking.hierarchical.summary_top_k == 3

    def test_proposition_defaults_from_yaml(self):
        assert settings.chunking.proposition.quality_threshold == 7


class TestEnvVarOverride:
    def test_llm_provider_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM__PROVIDER", "ollama")
        s = Settings()
        assert s.llm.provider == "ollama"

    def test_embedding_batch_size_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EMBEDDINGS__BATCH_SIZE", "64")
        s = Settings()
        assert s.embeddings.batch_size == 64

    def test_qdrant_url_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QDRANT__URL", "http://remote-qdrant:6333")
        s = Settings()
        assert s.qdrant.url == "http://remote-qdrant:6333"

    def test_api_port_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("API__PORT", "9000")
        s = Settings()
        assert s.api.port == 9000

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LOGGING__LEVEL", "DEBUG")
        s = Settings()
        assert s.logging.level == "DEBUG"

    def test_hybrid_alpha_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RETRIEVAL__HYBRID_ALPHA", "0.5")
        s = Settings()
        assert s.retrieval.hybrid_alpha == pytest.approx(0.5)


class TestValidation:
    def test_hybrid_alpha_out_of_range(self):
        with pytest.raises(ValidationError):
            RetrievalSettings(hybrid_alpha=1.5)

    def test_api_port_out_of_range(self):
        with pytest.raises(ValidationError):
            APISettings(port=99999)

    def test_invalid_llm_provider(self):
        with pytest.raises(ValidationError):
            LLMSettings(provider="unknown_provider")  # type: ignore[arg-type]

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            LoggingSettings(level="VERBOSE")  # type: ignore[arg-type]

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            EmbeddingSettings(device="tpu")  # type: ignore[arg-type]

    def test_compression_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            CompressionSettings(max_tokens=0)
