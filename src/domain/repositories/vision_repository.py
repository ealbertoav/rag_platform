from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VisionRepository(ABC):
    """Contract for vision-language models that caption local image assets."""

    @abstractmethod
    def caption_image(self, path: Path, *, prompt: str | None = None) -> str:
        """Return a natural-language caption for the image at *path*."""
