"""T-030 unit tests — LlamaCppProvider (Llama model mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import GenerationError
from src.domain.repositories.llm_repository import LLMRepository
from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider


def _provider(model: MagicMock | None = None) -> LlamaCppProvider:
    p = LlamaCppProvider(
        model_path="fake/model.gguf",
        context_size=512,
        n_gpu_layers=0,
        temperature=0.0,
        max_tokens=64,
    )
    if model is not None:
        p._model = model  # type: ignore[assignment]
    return p


def _llama_mock(response: str = "test response") -> MagicMock:
    m = MagicMock()
    m.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response}}]
    }
    return m


def _stream_mock(tokens: list[str]) -> MagicMock:
    m = MagicMock()
    chunks = [{"choices": [{"delta": {"content": t}}]} for t in tokens]
    m.create_chat_completion.return_value = iter(chunks)
    return m


# ── prompt + context joining (tested through generate) ────────────────────────


class TestPromptJoining:
    def test_no_context_sends_prompt_only(self):
        mock = _llama_mock()
        _provider(mock).generate("only prompt", "")
        _, kwargs = mock.create_chat_completion.call_args
        assert kwargs["messages"][0]["content"] == "only prompt"

    def test_context_appended_with_blank_line(self):
        mock = _llama_mock()
        _provider(mock).generate("prompt", "context")
        _, kwargs = mock.create_chat_completion.call_args
        content = kwargs["messages"][0]["content"]
        assert content == "prompt\n\ncontext"

    def test_empty_both_sends_empty_string(self):
        mock = _llama_mock()
        _provider(mock).generate("", "")
        _, kwargs = mock.create_chat_completion.call_args
        assert kwargs["messages"][0]["content"] == ""


# ── LLMRepository interface ────────────────────────────────────────────────────


class TestInterface:
    def test_implements_llm_repository(self):
        assert isinstance(_provider(), LLMRepository)

    def test_from_settings_returns_instance(self):
        p = LlamaCppProvider.from_settings()
        assert isinstance(p, LlamaCppProvider)


# ── generate ──────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_string(self):
        p = _provider(_llama_mock("hello world"))
        assert isinstance(p.generate("prompt", "ctx"), str)

    def test_returns_model_response(self):
        p = _provider(_llama_mock("expected answer"))
        assert p.generate("q", "") == "expected answer"

    def test_calls_create_chat_completion(self):
        mock = _llama_mock()
        p = _provider(mock)
        p.generate("my prompt", "my context")
        mock.create_chat_completion.assert_called_once()

    def test_model_error_raises_generation_error(self):
        mock = MagicMock()
        mock.create_chat_completion.side_effect = RuntimeError("OOM")
        p = _provider(mock)
        with pytest.raises(GenerationError) as exc_info:
            p.generate("q", "")
        assert exc_info.value.cause is not None

    def test_model_load_error_raises_generation_error(self):
        p = LlamaCppProvider(model_path="bad.gguf")
        with patch("src.infrastructure.llm.llama_cpp_provider.Llama",
                   side_effect=OSError("not found"), create=True), pytest.raises(GenerationError):
            p._get_model()

    def test_model_loaded_once(self):
        mock = _llama_mock()
        p = _provider(mock)
        p.generate("a", "")
        p.generate("b", "")
        # _model already set — no re-loading; create_chat_completion called twice
        assert mock.create_chat_completion.call_count == 2

    def test_temperature_forwarded(self):
        mock = _llama_mock()
        p = _provider(mock)
        p.generate("q", "", temperature=0.9)
        _, kwargs = mock.create_chat_completion.call_args
        assert kwargs["temperature"] == pytest.approx(0.9)

    def test_stream_false_for_generate(self):
        mock = _llama_mock()
        p = _provider(mock)
        p.generate("q", "")
        _, kwargs = mock.create_chat_completion.call_args
        assert kwargs["stream"] is False


# ── generate_stream ────────────────────────────────────────────────────────────


class TestGenerateStream:
    def test_returns_async_iterator(self):
        p = _provider(_stream_mock(["hello"]))
        stream = p.generate_stream("q", "")
        assert hasattr(stream, "__aiter__")

    @pytest.mark.asyncio
    async def test_yields_tokens(self):
        p = _provider(_stream_mock(["tok1", "tok2", "tok3"]))
        tokens = [t async for t in p.generate_stream("q", "")]
        assert tokens == ["tok1", "tok2", "tok3"]

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        p = _provider(_stream_mock([]))
        tokens = [t async for t in p.generate_stream("q", "")]
        assert tokens == []

    @pytest.mark.asyncio
    async def test_skips_empty_delta(self):
        mock = MagicMock()
        mock.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": ""}}]},
            {"choices": [{"delta": {"content": "real"}}]},
            {"choices": [{"delta": {}}]},
        ])
        p = _provider(mock)
        tokens = [t async for t in p.generate_stream("q", "")]
        assert tokens == ["real"]
