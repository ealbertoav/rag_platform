from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import AsyncIterator
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any

from src.core.exceptions import GenerationError
from src.domain.repositories.llm_repository import LLMRepository

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)

_SENTINEL = object()


class LlamaCppProvider(LLMRepository):
    """LLMRepository backed by llama-cpp-python (GGUF models, Metal/CPU).

    The "Llama" instance is created lazily on the first call and reused for
    all later requests, so model weights are loaded only once.

    llama.cpp model instances are not thread-safe. A process-wide lock serializes
    "generate" and streaming completions so concurrent retrieval paths (HyDE,
    multi-query fusion) and overlapping API requests cannot corrupt inference.
    """

    def __init__(
        self,
        model_path: str,
        context_size: int = 32768,
        n_gpu_layers: int = -1,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop_tokens: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_tokens: list[str] = stop_tokens or ["<|im_end|>"]
        self._model: Llama | None = None
        self._lock = Lock()

    # ── LLMRepository interface ────────────────────────────────────────────────

    def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
        """Return the full completion as a single string (blocking)."""
        with self._lock:
            model = self._get_model()
            try:
                output = model.create_chat_completion(
                    messages=[{"role": "user", "content": _join(prompt, context)}],  # type: ignore[arg-type]
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    stop=self.stop_tokens,
                    stream=False,
                )
                return str(output["choices"][0]["message"]["content"])  # type: ignore[index]
            except Exception as exc:
                raise GenerationError("llama.cpp generate() failed", cause=exc) from exc

    def generate_stream(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        """Return an async iterator that yields tokens as they are produced.

        The synchronous llama.cpp generator runs in a background thread; tokens
        are forwarded via a queue, so the event loop is never blocked.
        """
        return self._stream_tokens(prompt, context, **kwargs)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> LlamaCppProvider:
        from src.core.settings import settings

        cfg = settings.llm
        return cls(
            model_path=cfg.model_path,
            context_size=cfg.context_size,
            n_gpu_layers=cfg.n_gpu_layers,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stop_tokens=list(cfg.stop_tokens),
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_model(self) -> Llama:
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama  # lazy import

            llama = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            logger.info(
                "llama.cpp model loaded: %s (ctx=%d, gpu_layers=%d)",
                self.model_path,
                self.context_size,
                self.n_gpu_layers,
            )
            self._model = llama
            return llama
        except (ImportError, OSError, ValueError) as exc:
            raise GenerationError(
                f"Cannot load llama.cpp model from {self.model_path!r}", cause=exc
            ) from exc

    async def _stream_tokens(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        """Async generator: yields tokens from a sync llama.cpp stream via a thread."""
        full_prompt = _join(prompt, context)
        token_queue: queue.Queue[object] = queue.Queue()

        def _run() -> None:
            try:
                with self._lock:
                    model = self._get_model()
                    for chunk in model.create_chat_completion(
                        messages=[{"role": "user", "content": full_prompt}],  # type: ignore[arg-type]
                        max_tokens=kwargs.get("max_tokens", self.max_tokens),
                        temperature=kwargs.get("temperature", self.temperature),
                        stop=self.stop_tokens,
                        stream=True,
                    ):
                        choices = chunk["choices"]  # type: ignore[union-attr,index]
                        delta = str(choices[0]["delta"].get("content", ""))
                        if delta:
                            token_queue.put(delta)
            except Exception as exc:
                token_queue.put(exc)
            finally:
                token_queue.put(_SENTINEL)

        loop = asyncio.get_event_loop()
        thread = Thread(target=_run, daemon=True)
        thread.start()

        try:
            while True:
                item = await loop.run_in_executor(None, lambda: token_queue.get())  # type: ignore[misc]
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise GenerationError("llama.cpp stream failed", cause=item) from item
                if isinstance(item, str):
                    yield item
        finally:
            thread.join(timeout=5)


def _join(prompt: str, context: str) -> str:
    return f"{prompt}\n\n{context}" if context else prompt
