from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class YamlConfigSource(PydanticBaseSettingsSource):
    """Merges all configs/*.yaml files into a single settings dict.

    Lower priority than env vars — loaded last in the source chain, so env
    vars and .env file always win.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Not used — we override __call__ directly.
        return None, field_name, False

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


class LLMSettings(BaseModel):
    provider: Literal["llama_cpp", "ollama", "mlx", "vllm"] = "llama_cpp"
    model_path: str = "models/llm/qwen3-30b.gguf"
    context_size: int = 32768
    n_gpu_layers: int = -1
    temperature: float = 0.1
    max_tokens: int = 2048
    stop_tokens: list[str] = Field(default_factory=lambda: ["<|im_end|>"])


# ── API embedding provider config blocks (all optional) ───────────────────────


class OpenAIEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "text-embedding-3-large"
    dimensions: int = 3072  # text-embedding-3 supports truncation via this param


class VoyageEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "voyage-large-2"
    dimensions: int = 1536


class CohereEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "embed-english-v3.0"
    dimensions: int = 1024


class GeminiEmbeddingConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    model: str = "text-embedding-004"
    dimensions: int = 768


class EmbeddingCacheSettings(BaseModel):
    enabled: bool = False  # opt-in: avoids Redis round-trips for self-hosted providers
    ttl_seconds: int = 604800  # 7 days


class EmbeddingSettings(BaseModel):
    provider: Literal[
        "bge_m3",
        "nomic",
        "qwen_embedding",  # self-hosted
        "openai",
        "voyage",
        "cohere",
        "gemini",  # API-based
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

    cache: EmbeddingCacheSettings = Field(default_factory=EmbeddingCacheSettings)


class RerankerSettings(BaseModel):
    provider: Literal["bge_reranker", "qwen_reranker"] = "bge_reranker"
    model_path: str = "models/rerankers/bge-reranker-v2-m3"
    top_k: int = 10
    batch_size: int = 16


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


class RSESettings(BaseModel):
    enabled: bool = False
    max_segment_tokens: int = Field(default=1500, gt=0)


class RetrievalSettings(BaseModel):
    top_k_dense: int = 50
    top_k_final: int = 5
    # Used only when hybrid_fusion=weighted_linear. RRF (default) ignores this value.
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    hybrid_fusion: Literal["rrf", "weighted_linear"] = "rrf"
    hype: HyPESettings = Field(default_factory=HyPESettings)
    rse: RSESettings = Field(default_factory=RSESettings)


class Neo4jSettings(BaseModel):
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("")
    database: str = "neo4j"
    max_hops: int = Field(default=2, ge=1, le=5)
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


class QueryExpansionSettings(BaseModel):
    enabled: bool = True
    n_variants: int = Field(default=3, ge=1, le=10)


class CompressionSettings(BaseModel):
    enabled: bool = True
    max_tokens: int = Field(default=1500, gt=0)


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key: SecretStr = SecretStr("")
    max_upload_bytes: int = Field(default=10_485_760, gt=0)  # 10 MiB
    ingest_allowed_roots: list[str] = Field(default_factory=lambda: [str(ROOT / "data" / "raw")])


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


class ChunkingSettings(BaseModel):
    strategy: Literal["recursive", "semantic", "parent_child"] = "recursive"
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
    # SemanticChunker: split when cosine distance between adjacent sentences > threshold
    similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # ParentChildChunker
    parent_chunk_size: int = Field(default=1500, gt=0)
    child_chunk_size: int = Field(default=400, gt=0)
    contextual_headers: ContextualHeadersSettings = Field(default_factory=ContextualHeadersSettings)
    augmentation: AugmentationSettings = Field(default_factory=AugmentationSettings)


# ── Root settings ──────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Single source of truth for all configuration.

    Priority (highest → lowest):
      1. Code / init kwargs
      2. Environment variables (use __ as nested delimiter: LLM__PROVIDER)
      3. .env file
      4. configs/*.yaml files
    """

    model_config = SettingsConfigDict(
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
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
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
