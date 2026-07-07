from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from threading import Lock
from typing import TYPE_CHECKING, Any

from src.core.exceptions import GenerationError
from src.domain.repositories.llm_repository import LLMRepository
from src.observability.tracing import get_tracer

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)
_tracer = get_tracer("rag-platform.llm")

_SENTINEL = object()
# Backpressure when many concurrent streams fill the bridge queue.
_STREAM_QUEUE_MAXSIZE = 256


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
        disable_disk_cache: bool = False,
    ) -> None:
        self.model_path = model_path
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_tokens: list[str] = stop_tokens or ["<|im_end|>"]
        self.disable_disk_cache = disable_disk_cache
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

        Synchronous llama.cpp streaming runs in "asyncio.to_thread"; tokens
        cross into the event loop via a bounded "asyncio.Queue" so consumers
        await natively without "run_in_executor" polling.
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
            disable_disk_cache=cfg.disable_disk_cache,
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
            self._apply_prompt_cache_policy(llama)
            logger.info(
                "llama.cpp model loaded: %s (ctx=%d, gpu_layers=%d, disable_disk_cache=%s)",
                self.model_path,
                self.context_size,
                self.n_gpu_layers,
                self.disable_disk_cache,
            )
            self._model = llama
            return llama
        except (ImportError, OSError, ValueError) as exc:
            raise GenerationError(
                f"Cannot load llama.cpp model from {self.model_path!r}", cause=exc
            ) from exc

    def _apply_prompt_cache_policy(self, llama: Llama) -> None:
        """Configure llama.cpp prompt cache — never disk-backed when disabled (T-162)."""
        if self.disable_disk_cache:
            llama.set_cache(None)
            return
        from llama_cpp.llama_cache import LlamaRAMCache

        llama.set_cache(LlamaRAMCache())

    async def _stream_tokens(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        """Async generator: yields tokens from a sync llama.cpp stream via a thread."""
        full_prompt = _join(prompt, context)
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

        def _enqueue(item: object) -> None:
            asyncio.run_coroutine_threadsafe(token_queue.put(item), loop).result()

        def _producer() -> None:
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
                            _enqueue(delta)
            except Exception as exc:
                _enqueue(exc)
            finally:
                _enqueue(_SENTINEL)

        worker = asyncio.create_task(asyncio.to_thread(_producer))
        queue_wait_ms = 0.0

        with _tracer.start_as_current_span("llm.stream") as span:
            try:
                while True:
                    t0 = time.monotonic()
                    item = await token_queue.get()
                    queue_wait_ms += (time.monotonic() - t0) * 1000

                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise GenerationError("llama.cpp stream failed", cause=item) from item
                    if isinstance(item, str):
                        yield item
            finally:
                span.set_attribute("queue_wait_ms", round(queue_wait_ms, 1))
                await worker


def _join(prompt: str, context: str) -> str:
    return f"{prompt}\n\n{context}" if context else prompt
