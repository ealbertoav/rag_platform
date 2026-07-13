"""Vision provider factory for VLM figure captions (T-231).

Feature-flagged via ``parsing.figure_captions.enabled``. Returns ``None`` when
disabled. OpenAI and Gemini share the ``api-embeddings`` optional extra.
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
    "get_vision_provider",
]

_settings = load_settings
_cache: EnabledProviderCache[VisionRepository] = EnabledProviderCache()


def clear_vision_provider_cache() -> None:
    """Reset the cached vision provider instance (for tests and settings reloads)."""
    _cache.clear()


def _vision_provider_identity(cfg: FigureCaptionSettings) -> Hashable:
    """Fingerprint credentials/model captured at provider construction."""
    if cfg.provider == "gemini":
        gemini = cfg.gemini
        return (gemini.api_key.get_secret_value(), gemini.model)
    openai = cfg.openai
    return (openai.api_key.get_secret_value(), openai.model)


def get_vision_provider(app_settings: Settings | None = None) -> VisionRepository | None:
    """Return the configured vision provider, or None when captions are disabled.

    Cache key is ``(enabled, provider, identity)`` so credential and model
    rotations rebuild the client without an explicit cache clear.
    """
    if app_settings is None:
        app_settings = _settings()

    caption_cfg = app_settings.parsing.figure_captions

    def _create(provider: str) -> VisionRepository:
        match provider:
            case "openai":
                return OpenAIVisionProvider(
                    api_key=caption_cfg.openai.api_key.get_secret_value(),
                    model=caption_cfg.openai.model,
                )
            case "gemini":
                return GeminiVisionProvider(
                    api_key=caption_cfg.gemini.api_key.get_secret_value(),
                    model=caption_cfg.gemini.model,
                )
            case _:
                raise ConfigurationError(f"Unknown vision provider: {provider!r}")

    return _cache.get(caption_cfg, _create, identity=_vision_provider_identity(caption_cfg))
