from __future__ import annotations

from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
from src.infrastructure.embeddings.nomic import NomicEmbeddingProvider
from src.infrastructure.embeddings.qwen_embedding import QwenEmbeddingProvider

# API providers — imported lazily inside get_embedding_provider() to avoid
# importing optional libraries at module load time.

_API_PROVIDERS = {"openai", "voyage", "cohere", "gemini"}

# ENV_VAR names shown in ConfigurationError messages
_API_KEY_ENV: dict[str, str] = {
    "openai": "EMBEDDINGS__OPENAI__API_KEY",
    "voyage": "EMBEDDINGS__VOYAGE__API_KEY",
    "cohere": "EMBEDDINGS__COHERE__API_KEY",
    "gemini": "EMBEDDINGS__GEMINI__API_KEY",
}


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

    When "embeddings.cache.enabled = true" (default), the returned provider is
    wrapped in a Redis cache decorator.
    """
    from src.core.settings import settings

    provider_name = settings.embeddings.provider
    provider = _create_provider(provider_name, settings)

    if settings.embeddings.cache.enabled:
        provider = _wrap_with_cache(provider, provider_name, settings)

    return provider


def _create_provider(name: str, settings: object) -> EmbeddingRepository:
    from src.core.settings import Settings

    s: Settings = settings  # type: ignore[assignment]

    match name:
        case "bge_m3":
            return BGEM3EmbeddingProvider.from_settings()
        case "nomic":
            return NomicEmbeddingProvider.from_settings()
        case "qwen_embedding":
            return QwenEmbeddingProvider.from_settings()
        case "openai":
            _require_api_key(name, s.embeddings.openai.api_key)
            from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider.from_settings()
        case "voyage":
            _require_api_key(name, s.embeddings.voyage.api_key)
            from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

            return VoyageEmbeddingProvider.from_settings()
        case "cohere":
            _require_api_key(name, s.embeddings.cohere.api_key)
            from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

            return CohereEmbeddingProvider.from_settings()
        case "gemini":
            _require_api_key(name, s.embeddings.gemini.api_key)
            from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

            return GeminiEmbeddingProvider.from_settings()
        case _:
            raise ValueError(f"Unknown embedding provider: {name!r}")


def _require_api_key(provider: str, api_key: str) -> None:
    if not api_key:
        from src.core.exceptions import ConfigurationError

        env_var = _API_KEY_ENV.get(provider, f"EMBEDDINGS__{provider.upper()}__API_KEY")
        raise ConfigurationError(
            f"Provider '{provider}' requires an API key. "
            f"Set {env_var} in your environment or .env file."
        )


def _wrap_with_cache(
    inner: EmbeddingRepository, provider_name: str, settings: object
) -> EmbeddingRepository:
    from src.core.settings import Settings

    s: Settings = settings  # type: ignore[assignment]
    from src.infrastructure.embeddings.cached_embedding_provider import CachedEmbeddingProvider

    return CachedEmbeddingProvider(
        inner=inner,
        redis_url=s.redis.url,
        ttl_seconds=s.embeddings.cache.ttl_seconds,
        model_identifier=provider_name,
    )


__all__ = [
    "BGEM3EmbeddingProvider",
    "NomicEmbeddingProvider",
    "QwenEmbeddingProvider",
    "get_embedding_provider",
]
