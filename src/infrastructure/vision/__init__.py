"""Vision provider factory for VLM figure captions (T-231).

Feature-flagged via "parsing.figure_captions.enabled". Returns "None" when
disabled. OpenAI and Gemini share the "api-embeddings" optional extra.
"""

from __future__ import annotations

from collections.abc import Hashable

from src.core.exceptions import ConfigurationError
from src.core.settings import FigureCaptionSettings, Settings
from src.domain.repositories.vision_repository import VisionRepository
from src.infrastructure.provider_factory import EnabledProviderCache, load_settings
from src.infrastructure.vision.gemini_vision_provider import GeminiVisionProvider
from src.infrastructure.vision.openai_vision_provider import OpenAIVisionProvider

__all__ = [
    "GeminiVisionProvider",
    "OpenAIVisionProvider",
    "clear_vision_provider_cache",
    "get_generation_vision_provider",
    "get_vision_provider",
]

_settings = load_settings
_cache: EnabledProviderCache[VisionRepository] = EnabledProviderCache()
_generation_cache: EnabledProviderCache[VisionRepository] = EnabledProviderCache()


def clear_vision_provider_cache() -> None:
    """Reset the cached vision provider instances (for tests and settings reloads)."""
    _cache.clear()
    _generation_cache.clear()


def _vision_provider_identity(cfg: FigureCaptionSettings) -> Hashable:
    """Fingerprint credentials/model captured at provider construction."""
    if cfg.provider == "gemini":
        gemini = cfg.gemini
        return gemini.api_key.get_secret_value(), gemini.model
    openai = cfg.openai
    return openai.api_key.get_secret_value(), openai.model


def _create_vision_provider(cfg: FigureCaptionSettings, provider: str) -> VisionRepository:
    match provider:
        case "openai":
            return OpenAIVisionProvider(
                api_key=cfg.openai.api_key.get_secret_value(),
                model=cfg.openai.model,
            )
        case "gemini":
            return GeminiVisionProvider(
                api_key=cfg.gemini.api_key.get_secret_value(),
                model=cfg.gemini.model,
            )
        case _:
            raise ConfigurationError(f"Unknown vision provider: {provider!r}")


def get_vision_provider(app_settings: Settings | None = None) -> VisionRepository | None:
    """Return the configured vision provider or None when captions are disabled.

    The cache key is "(enabled, provider, identity)" so credential and model
    rotations rebuild the client without an explicit cache clear.
    """
    if app_settings is None:
        app_settings = _settings()

    caption_cfg = app_settings.parsing.figure_captions
    return _cache.get(
        caption_cfg,
        lambda provider: _create_vision_provider(caption_cfg, provider),
        identity=_vision_provider_identity(caption_cfg),
    )


def get_generation_vision_provider(app_settings: Settings | None = None) -> VisionRepository | None:
    """Return the vision provider for query-time figure descriptions (T-271).

    Reuses "parsing.figure_captions" credentials/provider, but is gated on
    "generation.vision_generation.enabled" instead — ingest-time captioning
    (T-231) and generation-time vision description toggle independently even
    though they share the same provider configuration.
    """
    if app_settings is None:
        app_settings = _settings()

    caption_cfg = app_settings.parsing.figure_captions
    effective_cfg = caption_cfg.model_copy(
        update={"enabled": app_settings.generation.vision_generation.enabled}
    )
    return _generation_cache.get(
        effective_cfg,
        lambda provider: _create_vision_provider(effective_cfg, provider),
        identity=_vision_provider_identity(effective_cfg),
    )
