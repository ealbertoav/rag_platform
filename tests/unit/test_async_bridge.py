"""Unit tests for src/core/async_bridge.py."""

from __future__ import annotations

import pytest

from src.core.async_bridge import run_async


class TestRunAsync:
    def test_run_async_from_sync_context(self):
        async def _work() -> int:
            return 42

        assert run_async(_work()) == 42

    @pytest.mark.asyncio
    async def test_run_async_from_running_loop(self):
        async def _work() -> str:
            return "ok"

        assert run_async(_work()) == "ok"
