"""NVIDIA NIM LLM provider (OpenAI-compatible hosted/self-hosted chat API)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, override

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import GenerationError
from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)


def _is_rate_limit(exc: BaseException) -> bool:
    try:
        from openai import RateLimitError

        return isinstance(exc, RateLimitError)
    except ImportError:
        msg = str(exc).lower()
        return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))


class NvidiaNimProvider(LLMRepository):
    """LLMRepository backed by NVIDIA NIM's OpenAI-compatible chat completions API.

    Unlike LlamaCppProvider, "prompt" and "context" map to separate "system"
    and "user" chat messages (ADR-0002) rather than being flattened into one
    "user" message — NIM's catalog models expect a real system/user split.

    "generate" retries on HTTP 429 with exponential backoff via tenacity;
    "generate_stream" does not, since a partially-yielded stream cannot be
    safely retried from scratch.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.1-8b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.base_url: str = base_url
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self._client: Any | None = None
        self._async_client: Any | None = None

    # ── LLMRepository interface ────────────────────────────────────────────────

    @override
    def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
        """Return the full completion as a single string (blocking)."""
        try:
            return self._call_with_retry(prompt, context, **kwargs)
        except Exception as exc:
            raise GenerationError("NVIDIA NIM generate() failed", cause=exc) from exc

    @override
    async def generate_stream(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        """Return an async iterator that yields tokens as they are produced."""
        client = self._get_async_client()
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=_build_messages(prompt, context),
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise GenerationError("NVIDIA NIM generate_stream() failed", cause=exc) from exc

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> NvidiaNimProvider:
        from src.core.settings import settings

        cfg = settings.llm
        nim_cfg = cfg.nvidia_nim
        return cls(
            api_key=nim_cfg.api_key.get_secret_value(),
            model=nim_cfg.model,
            base_url=nim_cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry(self, prompt: str, context: str, **kwargs: Any) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=_build_messages(prompt, context),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=False,
        )
        return str(response.choices[0].message.content)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise GenerationError(
                    "openai package is not installed. Run: uv sync --extra api-llm"
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise GenerationError(
                    "openai package is not installed. Run: uv sync --extra api-llm"
                ) from exc
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._async_client


def _build_messages(prompt: str, context: str) -> list[dict[str, str]]:
    """Map (prompt, context) to chat messages.

    When "context" is empty (GenerationService.generate_direct/call_llm), "prompt"
    is the raw user-facing text, not a system template — it must go in a "user"
    message or the request has no user turn at all. Only the RAG path (non-empty
    "context") has a real system template, so that's the only case with both roles.
    """
    if not context:
        return [{"role": "user", "content": prompt}]
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]
