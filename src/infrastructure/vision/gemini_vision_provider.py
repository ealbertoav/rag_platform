"""Google Gemini vision provider for figure captions (T-231)."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import override

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import ConfigurationError, GenerationError
from src.domain.repositories.vision_repository import VisionRepository

logger = logging.getLogger(__name__)

_DEFAULT_CAPTION_PROMPT = (
    "Describe this figure concisely for document retrieval. "
    "Focus on the main subject, labels, and any visible text."
)


def _is_rate_limit(exc: BaseException) -> bool:
    try:
        from google.api_core.exceptions import ResourceExhausted

        return isinstance(exc, ResourceExhausted)
    except ImportError:
        msg = str(exc).lower()
        keywords = ("429", "quota", "resource_exhausted", "rate_limit", "too many requests")
        return any(kw in msg for kw in keywords)


def _mime_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


class GeminiVisionProvider(VisionRepository):
    """Gemini multimodal captions via google.generativeai.

    ``genai.configure()`` runs immediately before each API request so multiple
    instances with different API keys do not share stale global state.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ConfigurationError(
                "Vision provider 'gemini' requires parsing.figure_captions.gemini.api_key"
            )
        self.api_key: str = normalized_key
        self.model: str = model

    @override
    def caption_image(self, path: Path, *, prompt: str | None = None) -> str:
        try:
            return self._call_with_retry(path, prompt or _DEFAULT_CAPTION_PROMPT).strip()
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                f"Gemini vision caption failed for {path.name}", cause=exc
            ) from exc

    @classmethod
    def from_settings(cls) -> GeminiVisionProvider:
        from src.core.settings import settings

        cfg = settings.parsing.figure_captions.gemini
        return cls(api_key=cfg.api_key.get_secret_value(), model=cfg.model)

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry(self, path: Path, prompt: str) -> str:
        return self._call_api(path, prompt)

    def _call_api(self, path: Path, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise GenerationError(
                "google-generativeai package is not installed. Run: uv sync --extra api-embeddings"
            ) from exc

        image_bytes = path.read_bytes()
        if not image_bytes:
            raise GenerationError(f"Figure asset is empty: {path}")

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            [
                prompt,
                {"mime_type": _mime_type_for(path), "data": image_bytes},
            ]
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise GenerationError(f"Gemini vision returned non-text content for {path.name}")
        return text
