"""OpenAI chat-completions vision provider for figure captions (T-231)."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, override

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
        from openai import RateLimitError

        return isinstance(exc, RateLimitError)
    except ImportError:
        msg = str(exc).lower()
        return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))


def _mime_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


class OpenAIVisionProvider(VisionRepository):
    """OpenAI vision captions via chat.completions with a base64 data URL."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ConfigurationError(
                "Vision provider 'openai' requires parsing.figure_captions.openai.api_key"
            )
        self.api_key: str = normalized_key
        self.model: str = model
        self._client: Any | None = None

    @override
    def caption_image(self, path: Path, *, prompt: str | None = None) -> str:
        try:
            return self._call_with_retry(path, prompt or _DEFAULT_CAPTION_PROMPT).strip()
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                f"OpenAI vision caption failed for {path.name}", cause=exc
            ) from exc

    @classmethod
    def from_settings(cls) -> OpenAIVisionProvider:
        from src.core.settings import settings

        cfg = settings.parsing.figure_captions.openai
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
        client = self._get_client()

        image_bytes = path.read_bytes()
        if not image_bytes:
            raise GenerationError(f"Figure asset is empty: {path}")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = _mime_type_for(path)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise GenerationError(f"OpenAI vision returned non-text content for {path.name}")
        return content

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise GenerationError(
                    "openai package is not installed. Run: uv sync --extra api-embeddings"
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client
