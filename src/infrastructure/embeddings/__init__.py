from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

from src.core.constants import (
    API_EMBEDDING_PROVIDERS,
    SELF_HOSTED_EMBEDDING_DEFAULT_DIMS,
    SELF_HOSTED_EMBEDDING_MODEL_PATHS,
)
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
from src.infrastructure.embeddings.nomic import NomicEmbeddingProvider
from src.infrastructure.embeddings.qwen_embedding import QwenEmbeddingProvider

if TYPE_CHECKING:
    from src.core.settings import Settings

# API providers — imported lazily inside get_embedding_provider() to avoid
# importing optional libraries at module load time.

# ENV_VAR names shown in ConfigurationError messages
_API_KEY_ENV: dict[str, str] = {
    "openai": "EMBEDDINGS__OPENAI__API_KEY",
    "voyage": "EMBEDDINGS__VOYAGE__API_KEY",
    "cohere": "EMBEDDINGS__COHERE__API_KEY",
    "gemini": "EMBEDDINGS__GEMINI__API_KEY",
}


def create_embedding_provider(name: str, settings: Settings) -> EmbeddingRepository:
    """Construct a provider by name with explicit settings, without cache wrapping.

    Unlike get_embedding_provider(), this does not read from the global settings
    singleton and does not apply the Redis cache decorator.  Use this when explicit
    control over which provider is instantiated is needed (e.g. benchmarking scripts
    that iterate over multiple providers without mutating global state).
    """
    return _create_provider(name, settings)


def get_embedding_provider() -> EmbeddingRepository:
    """Return the configured embedding provider based on "settings.embeddings.provider".

    Supported values (set in configs/embeddings.yaml or via env var
    "EMBEDDINGS__PROVIDER"):

    Self-hosted (no API key required):
        bge_m3         — BGE-M3 1024-dim dense and sparse (default)
        nomic          — Nomic-Embed-Text v1.5, 768-dim dense only
        qwen_embedding — Qwen3-Embedding-0.6B, 1024-dim dense only

    API-based (require API key and "uv sync --extra api-embeddings"):
        openai         — text-embedding-3-large/small/ada-002, dense only
        voyage         — voyage-large-2 / voyage-code-2, dense only
        cohere         — embed-english-v3.0 / embed-multilingual-v3.0, dense only
        gemini         — text-embedding-004 (768-dim), dense only

    When switching providers, update "embeddings.dense_dim" to match the new
    model and run "python scripts/rebuild_embeddings.py --recreate-collection".

    When "embeddings.cache.enabled = true", the returned provider is
    wrapped in a Redis cache decorator (disabled by default).
    """
    from src.core.settings import settings

    provider_name = settings.embeddings.provider
    provider = _create_provider(provider_name, settings)

    if settings.embeddings.cache.enabled:
        provider = _wrap_with_cache(provider, provider_name, settings)

    return provider


def _self_hosted_model_path(name: str, settings: Settings) -> str:
    """Return the model path for a self-hosted provider.

    Uses the configured "model_path" when it matches the active provider;
    otherwise falls back to each provider's canonical default so benchmarking
    scripts can instantiate alternate providers without mutating global config.
    """
    cfg = settings.embeddings
    if name == cfg.provider:
        return cfg.model_path
    return SELF_HOSTED_EMBEDDING_MODEL_PATHS[name]


def provider_dense_dim(name: str, settings: Settings) -> int:
    """Return the dense vector dimension for *name*."""
    if name in API_EMBEDDING_PROVIDERS:
        emb = settings.embeddings
        match name:
            case "openai":
                from src.infrastructure.embeddings.openai_provider import openai_effective_dense_dim

                return openai_effective_dense_dim(emb.openai.model, emb.openai.dimensions)
            case "voyage":
                return emb.voyage.dimensions
            case "cohere":
                return emb.cohere.dimensions
            case "gemini":
                return emb.gemini.dimensions
    return SELF_HOSTED_EMBEDDING_DEFAULT_DIMS[name]


def embedding_model_identifier(provider_name: str, settings: Settings) -> str:
    """Return a stable string that uniquely identifies the active model.

    Includes both the provider name and the specific model name/path so that
    switching models within the same provider (e.g. text-embedding-3-large →
    text-embedding-3-small) produces different cache keys and Qdrant payloads.

    API providers also append the effective dense dimension (``@512``) so that
    OpenAI text-embedding-3 truncation and other dimension overrides produce
    distinct identifiers.
    """
    emb = settings.embeddings
    api_model: str | None = None
    if provider_name == "openai":
        api_model = emb.openai.model
    elif provider_name == "voyage":
        api_model = emb.voyage.model
    elif provider_name == "cohere":
        api_model = emb.cohere.model
    elif provider_name == "gemini":
        api_model = emb.gemini.model
    model = api_model if api_model is not None else _self_hosted_model_path(provider_name, settings)
    base = f"{provider_name}:{model}"
    if provider_name in API_EMBEDDING_PROVIDERS:
        return f"{base}@{provider_dense_dim(provider_name, settings)}"
    return base


def _create_provider(name: str, settings: Settings) -> EmbeddingRepository:
    match name:
        case "bge_m3":
            cfg = settings.embeddings
            return BGEM3EmbeddingProvider(
                model_path=_self_hosted_model_path(name, settings),
                device=cfg.device,
                batch_size=cfg.batch_size,
                normalize=cfg.normalize,
            )
        case "nomic":
            cfg = settings.embeddings
            return NomicEmbeddingProvider(
                model_path=_self_hosted_model_path(name, settings),
                device=cfg.device,
                batch_size=cfg.batch_size,
                normalize=cfg.normalize,
            )
        case "qwen_embedding":
            cfg = settings.embeddings
            return QwenEmbeddingProvider(
                model_path=_self_hosted_model_path(name, settings),
                device=cfg.device,
                batch_size=cfg.batch_size,
                normalize=cfg.normalize,
            )
        case "openai":
            openai_cfg = settings.embeddings.openai
            _require_api_key(name, openai_cfg.api_key)
            from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(
                api_key=openai_cfg.api_key.get_secret_value(),
                model=openai_cfg.model,
                dimensions=openai_cfg.dimensions,
            )
        case "voyage":
            voyage_cfg = settings.embeddings.voyage
            _require_api_key(name, voyage_cfg.api_key)
            from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

            return VoyageEmbeddingProvider(
                api_key=voyage_cfg.api_key.get_secret_value(),
                model=voyage_cfg.model,
            )
        case "cohere":
            cohere_cfg = settings.embeddings.cohere
            _require_api_key(name, cohere_cfg.api_key)
            from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

            return CohereEmbeddingProvider(
                api_key=cohere_cfg.api_key.get_secret_value(),
                model=cohere_cfg.model,
            )
        case "gemini":
            gemini_cfg = settings.embeddings.gemini
            _require_api_key(name, gemini_cfg.api_key)
            from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

            return GeminiEmbeddingProvider(
                api_key=gemini_cfg.api_key.get_secret_value(),
                model=gemini_cfg.model,
            )
        case _:
            raise ValueError(f"Unknown embedding provider: {name!r}")


def _require_api_key(provider: str, api_key: SecretStr) -> None:
    raw = api_key.get_secret_value()
    if not raw:
        from src.core.exceptions import ConfigurationError

        env_var = _API_KEY_ENV.get(provider, f"EMBEDDINGS__{provider.upper()}__API_KEY")
        raise ConfigurationError(
            f"Provider '{provider}' requires an API key. "
            f"Set {env_var} in your environment or .env file."
        )


def _wrap_with_cache(
    inner: EmbeddingRepository, provider_name: str, settings: Settings
) -> EmbeddingRepository:
    from src.infrastructure.embeddings.cached_embedding_provider import CachedEmbeddingProvider

    return CachedEmbeddingProvider(
        inner=inner,
        redis_url=settings.redis.url,
        redis_password=settings.redis.password.get_secret_value(),
        ttl_seconds=settings.embeddings.cache.ttl_seconds,
        model_identifier=embedding_model_identifier(provider_name, settings),
    )


__all__ = [
    "BGEM3EmbeddingProvider",
    "NomicEmbeddingProvider",
    "QwenEmbeddingProvider",
    "create_embedding_provider",
    "embedding_model_identifier",
    "get_embedding_provider",
    "provider_dense_dim",
]
