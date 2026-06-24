from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMRepository(ABC):
    """Contract for generating text from a language model."""

    @abstractmethod
    def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
        """Return the full completion as a single string (blocking)."""

    @abstractmethod
    def generate_stream(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        """Return an async iterator that yields tokens as they are produced.

        Callers iterate with:
            async for token in repo.generate_stream(prompt, context):
                ...
        """
