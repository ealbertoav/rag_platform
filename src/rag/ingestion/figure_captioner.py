"""VLM figure captioning at ingest (T-231).

When ``parsing.figure_captions.enabled`` is true, call the configured OpenAI or
Gemini vision provider for each ``figures[]`` entry that has an ``asset_path``,
and write the returned text into ``figures[].caption``.

Depends on stored figure assets from T-230. Soft-fails when the VLM is
unavailable or a single figure fails so ingest continues. Existing Docling
captions are overwritten only when the VLM returns non-empty text.
Caption chunk indexing is T-232.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.constants import ASSET_PATH_KEY, FIGURE_CAPTION_KEY, FIGURE_ID_KEY
from src.core.exceptions import ConfigurationError, GenerationError
from src.core.settings import FigureCaptionSettings, Settings
from src.domain.entities.document import Document
from src.domain.repositories.vision_repository import VisionRepository

logger = logging.getLogger(__name__)


def apply_figure_captions(
    document: Document,
    *,
    app_settings: Settings | None = None,
    vision_provider: VisionRepository | None = None,
) -> Document:
    """Caption stored figure assets and attach text onto ``figures[]`` metadata.

    No-op when figure captions are disabled or no figures have an asset path.
    Soft-fails (logs and returns the best-effort document) on configuration /
    generation errors so ingest continues.
    """
    cfg = _figure_caption_settings(app_settings)
    if not cfg.enabled and vision_provider is None:
        return document

    figures = document.metadata.get("figures")
    if not isinstance(figures, list) or not figures:
        return document

    provider = vision_provider
    if provider is None:
        provider = _resolve_vision_provider(app_settings)
        if provider is None:
            return document

    updated_figures: list[Any] = []
    changed = False
    for index, raw_entry in enumerate(figures):
        if not isinstance(raw_entry, dict):
            updated_figures.append(raw_entry)
            continue

        entry = dict(raw_entry)
        asset_path = entry.get(ASSET_PATH_KEY) or entry.get("asset_path")
        figure_id = entry.get(FIGURE_ID_KEY) or f"index-{index}"
        if not asset_path:
            updated_figures.append(entry)
            continue

        path = Path(str(asset_path))
        if not path.is_file():
            logger.warning(
                "Figure caption skipped for %s: asset missing at %s",
                figure_id,
                path,
            )
            updated_figures.append(entry)
            continue

        try:
            caption = provider.caption_image(path).strip()
        except GenerationError as exc:
            logger.warning(
                "Figure caption failed for %s: %s — keeping existing caption",
                figure_id,
                exc,
            )
            updated_figures.append(entry)
            continue
        except Exception as exc:
            logger.warning(
                "Unexpected figure caption error for %s: %s — keeping existing caption",
                figure_id,
                exc,
            )
            updated_figures.append(entry)
            continue

        if not caption:
            logger.warning(
                "Figure caption empty for %s — keeping existing caption",
                figure_id,
            )
            updated_figures.append(entry)
            continue

        entry[FIGURE_CAPTION_KEY] = caption
        changed = True
        updated_figures.append(entry)

    if not changed:
        return document

    metadata = dict(document.metadata)
    metadata["figures"] = updated_figures
    return document.model_copy(update={"metadata": metadata})


def _resolve_vision_provider(app_settings: Settings | None) -> VisionRepository | None:
    """Return the configured vision provider, or None when disabled / misconfigured."""
    from src.infrastructure.vision import get_vision_provider

    try:
        return get_vision_provider(app_settings)
    except ConfigurationError as exc:
        logger.warning(
            "Vision caption provider misconfigured; continuing without captions: %s",
            exc,
        )
        return None


def _figure_caption_settings(app_settings: Settings | None) -> FigureCaptionSettings:
    if app_settings is not None:
        return app_settings.parsing.figure_captions
    from src.core.settings import settings

    return settings.parsing.figure_captions
