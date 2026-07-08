from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import TypeVar

T = TypeVar("T")

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _background_loop() -> asyncio.AbstractEventLoop:
    """Persistent loop for sync callers (CLI, FastAPI handlers on the main loop)."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None:
            loop = asyncio.new_event_loop()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            _bg_thread = threading.Thread(
                target=_runner,
                name="async-bridge",
                daemon=True,
            )
            _bg_thread.start()
            _bg_loop = loop
        return _bg_loop


def run_async(coro: Coroutine[object, object, T]) -> T:
    """Run *coro* from sync code, including when the caller already has a running loop."""
    loop = _background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
