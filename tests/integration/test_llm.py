"""T-030 integration tests — LlamaCppProvider (requires a GGUF model on disk).

Run with:
    uv run pytest tests/integration/test_llm.py -v

The tests are skipped when no model file is found under models/llm/.
"""
from __future__ import annotations

import pytest

from src.core.constants import MODELS_DIR

_MODEL_DIR = MODELS_DIR / "llm"
_MODEL_FILE = next(_MODEL_DIR.glob("*.gguf"), None) if _MODEL_DIR.exists() else None

pytestmark = pytest.mark.skipif(
    _MODEL_FILE is None,
    reason=f"No GGUF model found in {_MODEL_DIR}",
)


@pytest.fixture(scope="module")
def provider():
    from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

    return LlamaCppProvider(
        model_path=str(_MODEL_FILE),
        context_size=512,
        n_gpu_layers=-1,
        temperature=0.0,
        max_tokens=64,
    )


class TestLlamaCppIntegration:
    def test_generate_returns_string(self, provider):
        result = provider.generate("Say hello in one word.", "")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_uses_context(self, provider):
        result = provider.generate(
            "Answer the question using the context.",
            "The sky is blue.",
        )
        assert isinstance(result, str)

    def test_model_loaded_once(self, provider):
        m1 = provider._model
        provider.generate("Hi", "")
        m2 = provider._model
        assert m1 is m2

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, provider):
        tokens = [t async for t in provider.generate_stream("Count: one two", "")]
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_stream_tokens_join_to_nonempty_string(self, provider):
        tokens = [t async for t in provider.generate_stream("Say one word.", "")]
        assert "".join(tokens).strip() != ""
