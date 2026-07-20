from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, override

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class YamlConfigSource(PydanticBaseSettingsSource):
    """Merges all configs/*.yaml files into a single settings dict.

    Lower priority than env vars — loaded last in the source chain, so env
    vars and .env file always win.
    """

    @override
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Not used — we override __call__ directly.
        return None, field_name, False

    @override
    def __call__(self) -> dict[str, Any]:
        configs_dir = ROOT / "configs"
        if not configs_dir.exists():
            return {}
        merged: dict[str, Any] = {}
        for path in sorted(configs_dir.glob("*.yaml")):
            with path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            merged.update(data)
        return merged


# ── Nested config models ───────────────────────────────────────────────────────


class NvidiaNIMConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "meta/llama-3.1-8b-instruct"
    base_url: str = "https://integrate.api.nvidia.com/v1"


class LLMSettings(BaseModel):
    provider: Literal["llama_cpp", "ollama", "mlx", "vllm", "nvidia_nim"] = "llama_cpp"
    model_path: str = "models/llm/qwen3-30b.gguf"
    context_size: int = 32768
    n_gpu_layers: int = -1
    temperature: float = 0.1
    max_tokens: int = 2048
    stop_tokens: list[str] = Field(default_factory=lambda: ["<|im_end|>"])
    disable_disk_cache: bool = False

    # Per-provider API config (populated from env vars or YAML)
    nvidia_nim: NvidiaNIMConfig = Field(default_factory=NvidiaNIMConfig)


# ── API embedding provider config blocks (all optional) ───────────────────────


class OpenAIEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "text-embedding-3-large"
    dimensions: int = 3072  # text-embedding-3 supports truncation via this param


class VoyageEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "voyage-large-2"
    dimensions: int = 1536
    multimodal_model: str = "voyage-multimodal-3"  # used by embed_image() (T-251)
    multimodal_dimensions: int = 1024  # voyage-multimodal-3 output dim; sizes image_dense (T-252)


class CohereEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "embed-english-v3.0"
    dimensions: int = 1024


class GeminiEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "text-embedding-004"
    dimensions: int = 768


class NvidiaNIMEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    # nvidia/llama-3.2-nv-embedqa-1b-v2 reached end-of-life on 2026-05-18 (HTTP 410);
    # this is its successor in the NIM catalog, confirmed live against /v1/embeddings.
    model: str = "nvidia/llama-nemotron-embed-1b-v2"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    dimensions: int = 2048  # confirmed against a real embedding response (#79 smoke test)


class EmbeddingCacheSettings(BaseModel):
    enabled: bool = False  # opt-in: avoids Redis round-trips for self-hosted providers
    ttl_seconds: int = 604800  # 7 days


class EmbeddingSettings(BaseModel):
    provider: Literal[
        "bge_m3",
        "nomic",
        "qwen_embedding",
        "clip",  # self-hosted (clip is text+image, T-251)
        "openai",
        "voyage",
        "cohere",
        "gemini",  # API-based
        "nvidia_nim",
    ] = "bge_m3"
    model_path: str = "models/embeddings/bge-m3"
    batch_size: int = 32
    device: Literal["mps", "cuda", "cpu"] = "mps"
    normalize: bool = True
    dense_dim: int = 1024
    sparse_dim: int = 30522

    # Per-provider API config (populated from env vars or YAML)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    voyage: VoyageEmbeddingConfig = Field(default_factory=VoyageEmbeddingConfig)
    cohere: CohereEmbeddingConfig = Field(default_factory=CohereEmbeddingConfig)
    gemini: GeminiEmbeddingConfig = Field(default_factory=GeminiEmbeddingConfig)
    nvidia_nim: NvidiaNIMEmbeddingConfig = Field(default_factory=NvidiaNIMEmbeddingConfig)

    cache: EmbeddingCacheSettings = Field(default_factory=EmbeddingCacheSettings)


class NvidiaNIMRerankerConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    # nvidia/llama-3.2-nv-rerankqa-1b-v2 never resolved (HTTP 404); this is the
    # correct current model, confirmed live against the reranking invoke URL.
    model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    # NIM reranking NIMs live on a different host/path than chat + embeddings —
    # the invoke URL is "{base_url}/retrieval/{model}/reranking", not "{base_url}/ranking".
    base_url: str = "https://ai.api.nvidia.com/v1"


class RerankerSettings(BaseModel):
    provider: Literal["bge_reranker", "qwen_reranker", "nvidia_nim"] = "bge_reranker"
    model_path: str = "models/rerankers/bge-reranker-v2-m3"
    top_k: int = 10
    batch_size: int = 16
    # T-262: additive boost applied to table/caption chunk scores after cross-encoder
    # scoring, since cross-encoders trained on prose pairs tend to under-score
    # structured content. 0.0 (default) disables the boost — no behavior change.
    modality_boost: float = 0.0

    # Per-provider API config (populated from env vars or YAML)
    nvidia_nim: NvidiaNIMRerankerConfig = Field(default_factory=NvidiaNIMRerankerConfig)


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "rag_documents"
    api_key: str = ""


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379"
    password: SecretStr = SecretStr("")


class HyPESettings(BaseModel):
    enabled: bool = False
    n_questions: int = Field(default=3, gt=0)


class HyDESettings(BaseModel):
    enabled: bool = False


class CategoryStrategySettings(BaseModel):
    top_k: int = Field(gt=0)
    n_variants: int = Field(default=1, ge=1, le=10)
    hyde: bool = False
    compression: bool = True


def _default_adaptive_strategies() -> dict[str, CategoryStrategySettings]:
    return {
        "factual": CategoryStrategySettings(top_k=30, n_variants=1, hyde=False, compression=True),
        "analytical": CategoryStrategySettings(top_k=50, n_variants=3, hyde=True, compression=True),
        "opinion": CategoryStrategySettings(top_k=20, n_variants=2, hyde=False, compression=False),
        "contextual": CategoryStrategySettings(
            top_k=40, n_variants=2, hyde=False, compression=True
        ),
    }


class AdaptiveSettings(BaseModel):
    enabled: bool = False
    strategies: dict[str, CategoryStrategySettings] = Field(
        default_factory=_default_adaptive_strategies,
    )


class RSESettings(BaseModel):
    enabled: bool = False
    max_segment_tokens: int = Field(default=1500, gt=0)


class ParentContextSettings(BaseModel):
    enabled: bool = False


class DiversitySettings(BaseModel):
    enabled: bool = False
    lambda_: float = Field(default=0.7, ge=0.0, le=1.0, alias="lambda")


class BM25Settings(BaseModel):
    """Lexical BM25 backend selection (T-165).

    "memory" (default) keeps the full Okapi model in RAM — fine for typical
    corpora.  "disk" stores postings and chunk payloads as memmapped segments
    so search memory stays bounded for 100K–1M+ chunk corpora.
    """

    backend: Literal["memory", "disk"] = "memory"
    disk_path: str = "data/processed/bm25_disk"
    # Max chunks per on-disk segment (disk backend only).
    segment_size: int = Field(default=10_000, ge=1)


class ReliableRAGSettings(BaseModel):
    enabled: bool = False
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)


class SelfRAGSettings(BaseModel):
    enabled: bool = False


class CRAGSettings(BaseModel):
    enabled: bool = False
    lower_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    upper_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def thresholds_ordered(self) -> CRAGSettings:
        if self.lower_threshold > self.upper_threshold:
            msg = "quality.crag.lower_threshold must be <= upper_threshold"
            raise ValueError(msg)
        return self


class TavilySearchConfig(BaseModel):
    api_key: SecretStr = SecretStr("")


class WebSearchSettings(BaseModel):
    provider: Literal["none", "duckduckgo", "tavily"] = "none"
    max_results: int = Field(default=5, ge=1, le=20)
    tavily: TavilySearchConfig = Field(default_factory=TavilySearchConfig)


class SourceHighlightingSettings(BaseModel):
    enabled: bool = False


class SourceReferencesSettings(BaseModel):
    """Structured multimodal citations on /chat/full (T-272)."""

    enabled: bool = False


class ChunkLookupSettings(BaseModel):
    """Direct chunk lookup via GET /chunks/{chunk_id} (T-273)."""

    enabled: bool = False


class FeedbackLoopSettings(BaseModel):
    enabled: bool = False
    boost_multiplier: float = Field(default=0.05, ge=0.0)
    expand_candidate_pool: bool = True
    backend: Literal["qdrant", "redis", "postgres"] = "qdrant"
    postgres_url: str = ""


class MultimodalPromptSettings(BaseModel):
    """Mixed-modality system prompt for generation (T-270)."""

    enabled: bool = False


class VisionGenerationSettings(BaseModel):
    """Query-time vision-LLM figure descriptions (T-271).

    Reuses the `parsing.figure_captions` provider/credentials, but this flag is
    independent of `parsing.figure_captions.enabled` — ingest-time captioning
    (T-231) and generation-time vision description toggle separately.
    """

    enabled: bool = False


class GenerationSettings(BaseModel):
    multimodal_prompt: MultimodalPromptSettings = Field(default_factory=MultimodalPromptSettings)
    vision_generation: VisionGenerationSettings = Field(default_factory=VisionGenerationSettings)


class QualitySettings(BaseModel):
    reliable_rag: ReliableRAGSettings = Field(default_factory=ReliableRAGSettings)
    self_rag: SelfRAGSettings = Field(default_factory=SelfRAGSettings)
    crag: CRAGSettings = Field(default_factory=CRAGSettings)
    source_highlighting: SourceHighlightingSettings = Field(
        default_factory=SourceHighlightingSettings,
    )
    source_references: SourceReferencesSettings = Field(
        default_factory=SourceReferencesSettings,
    )
    chunk_lookup: ChunkLookupSettings = Field(default_factory=ChunkLookupSettings)
    feedback_loop: FeedbackLoopSettings = Field(default_factory=FeedbackLoopSettings)


class RRFWeightsSettings(BaseModel):
    """Per-leg RRF weight multipliers (T-263). 1.0 = unweighted (pre-T-263 default)."""

    dense: float = Field(default=1.0, ge=0.0)
    bm25: float = Field(default=1.0, ge=0.0)
    graph: float = Field(default=1.0, ge=0.0)
    hype: float = Field(default=1.0, ge=0.0)
    hyde: float = Field(default=1.0, ge=0.0)
    hierarchical: float = Field(default=1.0, ge=0.0)
    image: float = Field(default=1.0, ge=0.0)


class RetrievalSettings(BaseModel):
    top_k_dense: int = 50
    top_k_final: int = 5
    # Used only when hybrid_fusion=weighted_linear. RRF (default) ignores this value.
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    hybrid_fusion: Literal["rrf", "weighted_linear"] = "rrf"
    # Per-leg weight override for RRF fusion (T-263); all-1.0 default is unweighted RRF.
    rrf_weights: RRFWeightsSettings = Field(default_factory=RRFWeightsSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    hype: HyPESettings = Field(default_factory=HyPESettings)
    hyde: HyDESettings = Field(default_factory=HyDESettings)
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)
    rse: RSESettings = Field(default_factory=RSESettings)
    parent_context: ParentContextSettings = Field(default_factory=ParentContextSettings)
    diversity: DiversitySettings = Field(default_factory=DiversitySettings)

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)


class Neo4jSettings(BaseModel):
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("")
    database: str = "neo4j"
    max_hops: int = Field(default=2, ge=1, le=5)
    max_connection_pool_size: int = Field(default=100, ge=1)
    extract_entities_on_ingest: bool = True

    @model_validator(mode="after")
    def password_required_when_enabled(self) -> Neo4jSettings:
        if self.enabled and not self.password.get_secret_value():
            msg = "NEO4J__PASSWORD is required when NEO4J__ENABLED=true"
            raise ValueError(msg)
        return self


class MetadataSettings(BaseModel):
    db_path: str = "data/processed/metadata.db"
    enabled: bool = True


class StepBackSettings(BaseModel):
    enabled: bool = False


class QueryExpansionSettings(BaseModel):
    enabled: bool = True
    n_variants: int = Field(default=3, ge=1, le=10)
    step_back: StepBackSettings = Field(default_factory=StepBackSettings)


class CompressionSettings(BaseModel):
    enabled: bool = True
    max_tokens: int = Field(default=1500, gt=0)


class APIRateLimitSettings(BaseModel):
    enabled: bool = False
    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=0)


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key: SecretStr = SecretStr("")
    max_upload_bytes: int = Field(default=10_485_760, gt=0)  # 10 MiB
    ingest_allowed_roots: list[str] = Field(default_factory=lambda: [str(ROOT / "data" / "raw")])
    rate_limit: APIRateLimitSettings = Field(default_factory=APIRateLimitSettings)


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    otel_endpoint: str = "http://localhost:4317"
    trace_sampling_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class ContextualHeadersSettings(BaseModel):
    enabled: bool = False
    exclude_from_llm_context: bool = True


class AugmentationSettings(BaseModel):
    enabled: bool = False
    n_questions: int = Field(default=3, gt=0)


class HierarchicalSettings(BaseModel):
    enabled: bool = False
    summary_top_k: int = Field(default=3, ge=1, le=10)


class PropositionSettings(BaseModel):
    quality_threshold: int = Field(default=7, ge=1, le=10)


class ChunkingSettings(BaseModel):
    strategy: Literal["recursive", "semantic", "parent_child", "proposition", "section", "page"] = (
        "recursive"
    )
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
    # SemanticChunker: split when cosine distance between adjacent sentences > threshold
    similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # ParentChildChunker
    parent_chunk_size: int = Field(default=1500, gt=0)
    child_chunk_size: int = Field(default=400, gt=0)
    contextual_headers: ContextualHeadersSettings = Field(default_factory=ContextualHeadersSettings)
    augmentation: AugmentationSettings = Field(default_factory=AugmentationSettings)
    hierarchical: HierarchicalSettings = Field(default_factory=HierarchicalSettings)
    proposition: PropositionSettings = Field(default_factory=PropositionSettings)


class LayoutParserSettings(BaseModel):
    enabled: bool = False
    provider: str = "docling"


class AzureDiOcrConfig(BaseModel):
    """Azure Document Intelligence credentials (T-222).

    Required when "parsing.ocr.provider=azure_di" and OCR are enabled.
    """

    endpoint: str = ""
    api_key: SecretStr = SecretStr("")
    api_version: str = "2024-11-30"
    model_id: str = "prebuilt-read"
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    poll_interval_seconds: float = Field(default=1.0, gt=0.0)


class OcrSettings(BaseModel):
    enabled: bool = False
    provider: str = "tesseract"
    # Whole-file OCR when extractable text has fewer than this many
    # non-whitespace chars (per page when metadata["pages"] is present;
    # otherwise overall content).
    min_chars: int = Field(default=50, ge=0)
    azure_di: AzureDiOcrConfig = Field(default_factory=AzureDiOcrConfig)


class TableChunkSettings(BaseModel):
    enabled: bool = False


class CaptionChunkSettings(BaseModel):
    """Index type=caption chunks from figures[].caption (T-232)."""

    enabled: bool = False


class FigureAssetSettings(BaseModel):
    """Persist extracted figure bytes under a local asset store (T-230)."""

    enabled: bool = False
    store_dir: str = "data/assets"


class FigureChunkSettings(BaseModel):
    """Index type=figure chunks from figures[].asset_path (T-253)."""

    enabled: bool = False


class OpenAIVisionConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "gpt-4o-mini"


class GeminiVisionConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "gemini-2.0-flash"


class FigureCaptionSettings(BaseModel):
    """VLM captions for stored figure assets at ingesting (T-231)."""

    enabled: bool = False
    provider: str = "openai"  # openai | gemini
    openai: OpenAIVisionConfig = Field(default_factory=OpenAIVisionConfig)
    gemini: GeminiVisionConfig = Field(default_factory=GeminiVisionConfig)


class ParsingSettings(BaseModel):
    layout_parser: LayoutParserSettings = Field(default_factory=LayoutParserSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    table_chunks: TableChunkSettings = Field(default_factory=TableChunkSettings)
    caption_chunks: CaptionChunkSettings = Field(default_factory=CaptionChunkSettings)
    figure_assets: FigureAssetSettings = Field(default_factory=FigureAssetSettings)
    figure_captions: FigureCaptionSettings = Field(default_factory=FigureCaptionSettings)
    figure_chunks: FigureChunkSettings = Field(default_factory=FigureChunkSettings)


# ── Root settings ──────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Single source of truth for all configuration.

    Priority (highest → lowest):
      1. Code / init kwargs
      2. Environment variables (use __ as nested delimiter: LLM__PROVIDER)
      3. .env file
      4. configs/*.yaml files
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    query_expansion: QueryExpansionSettings = Field(default_factory=QueryExpansionSettings)
    compression: CompressionSettings = Field(default_factory=CompressionSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSource(settings_cls),
            file_secret_settings,
        )


settings: Settings = Settings()
