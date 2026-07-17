import pytest
from pydantic import ValidationError

from src.core.settings import (
    APISettings,
    ChunkingSettings,
    CompressionSettings,
    EmbeddingSettings,
    GenerationSettings,
    LLMSettings,
    LoggingSettings,
    MetadataSettings,
    Neo4jSettings,
    ParsingSettings,
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
        assert isinstance(settings.parsing, ParsingSettings)
        assert isinstance(settings.generation, GenerationSettings)


class TestYamlDefaults:
    def test_llm_defaults_from_yaml(self):
        assert settings.llm.provider == "llama_cpp"
        assert settings.llm.context_size == 32768
        assert settings.llm.temperature == 0.1
        assert settings.llm.max_tokens == 2048
        assert settings.llm.disable_disk_cache is False
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
        assert settings.retrieval.bm25.backend == "memory"
        assert settings.retrieval.bm25.disk_path == "data/processed/bm25_disk"
        assert settings.retrieval.bm25.segment_size == 10_000

    def test_rrf_weights_default_from_yaml(self):
        weights = settings.retrieval.rrf_weights
        assert weights.dense == pytest.approx(1.0)
        assert weights.bm25 == pytest.approx(1.0)
        assert weights.graph == pytest.approx(1.0)
        assert weights.hype == pytest.approx(1.0)
        assert weights.hyde == pytest.approx(1.0)
        assert weights.hierarchical == pytest.approx(1.0)
        assert weights.image == pytest.approx(1.0)

    def test_neo4j_defaults_from_yaml(self):
        assert settings.neo4j.enabled is False

    def test_metadata_defaults_from_yaml(self):
        assert settings.metadata.enabled is True

    def test_reranker_defaults_from_yaml(self):
        assert settings.reranker.provider == "bge_reranker"
        assert settings.reranker.top_k == 10
        assert settings.reranker.modality_boost == 0.0

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

    def test_self_rag_defaults_from_yaml(self):
        assert settings.quality.self_rag.enabled is False

    def test_crag_defaults_from_yaml(self):
        assert settings.quality.crag.enabled is False
        assert settings.quality.crag.lower_threshold == pytest.approx(0.3)
        assert settings.quality.crag.upper_threshold == pytest.approx(0.7)

    def test_source_highlighting_defaults_from_yaml(self):
        assert settings.quality.source_highlighting.enabled is False

    def test_web_search_defaults_from_yaml(self):
        assert settings.web_search.provider == "none"
        assert settings.web_search.max_results == 5

    def test_hierarchical_defaults_from_yaml(self):
        assert settings.chunking.hierarchical.enabled is False
        assert settings.chunking.hierarchical.summary_top_k == 3

    def test_proposition_defaults_from_yaml(self):
        assert settings.chunking.proposition.quality_threshold == 7

    def test_generation_defaults_from_yaml(self):
        assert settings.generation.multimodal_prompt.enabled is False
        assert settings.generation.vision_generation.enabled is False

    def test_parsing_defaults_from_yaml(self):
        assert settings.parsing.layout_parser.enabled is False
        assert settings.parsing.layout_parser.provider == "docling"
        assert settings.parsing.ocr.enabled is False
        assert settings.parsing.ocr.provider == "tesseract"
        assert settings.parsing.ocr.min_chars == 50
        assert settings.parsing.ocr.azure_di.endpoint == ""
        assert settings.parsing.ocr.azure_di.api_key.get_secret_value() == ""
        assert settings.parsing.ocr.azure_di.api_version == "2024-11-30"
        assert settings.parsing.ocr.azure_di.model_id == "prebuilt-read"
        assert settings.parsing.ocr.azure_di.timeout_seconds == 120.0
        assert settings.parsing.ocr.azure_di.poll_interval_seconds == 1.0
        assert settings.parsing.table_chunks.enabled is False
        assert settings.parsing.caption_chunks.enabled is False
        assert settings.parsing.figure_assets.enabled is False
        assert settings.parsing.figure_assets.store_dir == "data/assets"
        assert settings.parsing.figure_captions.enabled is False
        assert settings.parsing.figure_captions.provider == "openai"
        assert settings.parsing.figure_captions.openai.api_key.get_secret_value() == ""
        assert settings.parsing.figure_captions.openai.model == "gpt-4o-mini"
        assert settings.parsing.figure_captions.gemini.api_key.get_secret_value() == ""
        assert settings.parsing.figure_captions.gemini.model == "gemini-2.0-flash"


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

    def test_rrf_weights_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RETRIEVAL__RRF_WEIGHTS__DENSE", "2.0")
        monkeypatch.setenv("RETRIEVAL__RRF_WEIGHTS__BM25", "0.5")
        s = Settings()
        assert s.retrieval.rrf_weights.dense == pytest.approx(2.0)
        assert s.retrieval.rrf_weights.bm25 == pytest.approx(0.5)
        assert s.retrieval.rrf_weights.graph == pytest.approx(1.0)

    def test_disable_disk_cache_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM__DISABLE_DISK_CACHE", "true")
        s = Settings()
        assert s.llm.disable_disk_cache is True

    def test_parsing_layout_parser_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PARSING__LAYOUT_PARSER__ENABLED", "true")
        monkeypatch.setenv("PARSING__LAYOUT_PARSER__PROVIDER", "custom")
        s = Settings()
        assert s.parsing.layout_parser.enabled is True
        assert s.parsing.layout_parser.provider == "custom"

    def test_parsing_ocr_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PARSING__OCR__ENABLED", "true")
        monkeypatch.setenv("PARSING__OCR__PROVIDER", "azure_di")
        monkeypatch.setenv("PARSING__OCR__MIN_CHARS", "100")
        monkeypatch.setenv(
            "PARSING__OCR__AZURE_DI__ENDPOINT",
            "https://example.cognitiveservices.azure.com",
        )
        monkeypatch.setenv("PARSING__OCR__AZURE_DI__API_KEY", "secret-key")
        monkeypatch.setenv("PARSING__OCR__AZURE_DI__MODEL_ID", "prebuilt-layout")
        s = Settings()
        assert s.parsing.ocr.enabled is True
        assert s.parsing.ocr.provider == "azure_di"
        assert s.parsing.ocr.min_chars == 100
        assert s.parsing.ocr.azure_di.endpoint == "https://example.cognitiveservices.azure.com"
        assert s.parsing.ocr.azure_di.api_key.get_secret_value() == "secret-key"
        assert s.parsing.ocr.azure_di.model_id == "prebuilt-layout"

    def test_parsing_table_chunks_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PARSING__TABLE_CHUNKS__ENABLED", "true")
        s = Settings()
        assert s.parsing.table_chunks.enabled is True

    def test_parsing_caption_chunks_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PARSING__CAPTION_CHUNKS__ENABLED", "true")
        s = Settings()
        assert s.parsing.caption_chunks.enabled is True

    def test_parsing_figure_assets_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PARSING__FIGURE_ASSETS__ENABLED", "true")
        monkeypatch.setenv("PARSING__FIGURE_ASSETS__STORE_DIR", "/tmp/figures")
        s = Settings()
        assert s.parsing.figure_assets.enabled is True
        assert s.parsing.figure_assets.store_dir == "/tmp/figures"

    def test_parsing_figure_captions_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PARSING__FIGURE_CAPTIONS__ENABLED", "true")
        monkeypatch.setenv("PARSING__FIGURE_CAPTIONS__PROVIDER", "gemini")
        monkeypatch.setenv("PARSING__FIGURE_CAPTIONS__OPENAI__API_KEY", "sk-test")
        monkeypatch.setenv("PARSING__FIGURE_CAPTIONS__OPENAI__MODEL", "gpt-4o")
        monkeypatch.setenv("PARSING__FIGURE_CAPTIONS__GEMINI__API_KEY", "gemini-secret")
        monkeypatch.setenv("PARSING__FIGURE_CAPTIONS__GEMINI__MODEL", "gemini-1.5-flash")
        s = Settings()
        assert s.parsing.figure_captions.enabled is True
        assert s.parsing.figure_captions.provider == "gemini"
        assert s.parsing.figure_captions.openai.api_key.get_secret_value() == "sk-test"
        assert s.parsing.figure_captions.openai.model == "gpt-4o"
        assert s.parsing.figure_captions.gemini.api_key.get_secret_value() == "gemini-secret"
        assert s.parsing.figure_captions.gemini.model == "gemini-1.5-flash"

    def test_generation_multimodal_prompt_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GENERATION__MULTIMODAL_PROMPT__ENABLED", "true")
        s = Settings()
        assert s.generation.multimodal_prompt.enabled is True

    def test_generation_vision_generation_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GENERATION__VISION_GENERATION__ENABLED", "true")
        s = Settings()
        assert s.generation.vision_generation.enabled is True


class TestValidation:
    def test_hybrid_alpha_out_of_range(self):
        with pytest.raises(ValidationError):
            RetrievalSettings(hybrid_alpha=1.5)

    def test_rrf_weight_negative_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalSettings(rrf_weights={"dense": -1.0})

    def test_api_port_out_of_range(self):
        with pytest.raises(ValidationError):
            APISettings.model_validate({"port": 99999})

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
            CompressionSettings.model_validate({"max_tokens": 0})
