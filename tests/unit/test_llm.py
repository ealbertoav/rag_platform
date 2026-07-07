"""T-030 unit tests — LlamaCppProvider (Llama model mocked)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    m.create_chat_completion.return_value = {"choices": [{"message": {"content": response}}]}
    return m


def _stream_mock(tokens: list[str]) -> MagicMock:
    m = MagicMock()
    chunks = [{"choices": [{"delta": {"content": t}}]} for t in tokens]
    m.create_chat_completion.return_value = iter(chunks)
    return m


# ── prompt + context joining (tested through generating) ────────────────────────


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

    def test_from_settings_forwards_disable_disk_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM__DISABLE_DISK_CACHE", "true")
        from src.core.settings import Settings

        with patch("src.core.settings.settings", Settings()):
            p = LlamaCppProvider.from_settings()
        assert p.disable_disk_cache is True


class TestPromptCachePolicy:
    def test_disable_disk_cache_clears_prompt_cache(self):
        llama = MagicMock()
        p = LlamaCppProvider(model_path="fake/model.gguf", disable_disk_cache=True)
        p._apply_prompt_cache_policy(llama)
        llama.set_cache.assert_called_once_with(None)

    def test_disk_cache_enabled_uses_ram_cache(self):
        llama = MagicMock()
        p = LlamaCppProvider(model_path="fake/model.gguf", disable_disk_cache=False)
        with patch("llama_cpp.llama_cache.LlamaRAMCache") as ram_cache_cls:
            ram_cache = MagicMock()
            ram_cache_cls.return_value = ram_cache
            p._apply_prompt_cache_policy(llama)
        llama.set_cache.assert_called_once_with(ram_cache)

    def test_get_model_applies_cache_policy(self):
        llama = MagicMock()
        p = LlamaCppProvider(model_path="fake/model.gguf", disable_disk_cache=True)
        with (
            patch("llama_cpp.Llama", return_value=llama),
            patch.object(p, "_apply_prompt_cache_policy") as apply_policy,
        ):
            loaded = p._get_model()
        assert loaded is llama
        apply_policy.assert_called_once_with(llama)


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
        with (
            patch(
                "src.infrastructure.llm.llama_cpp_provider.Llama",
                side_effect=OSError("not found"),
                create=True,
            ),
            pytest.raises(GenerationError),
        ):
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

    def test_concurrent_generate_serializes_on_shared_model(self):
        mock = _llama_mock()
        active = 0
        peak = 0

        def _track(**_: object) -> dict[str, object]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            active -= 1
            return {"choices": [{"message": {"content": "ok"}}]}

        mock.create_chat_completion.side_effect = _track
        p = _provider(mock)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: p.generate(f"prompt-{i}", ""), range(16)))

        assert len(results) == 16
        assert all(r == "ok" for r in results)
        assert peak == 1
        assert mock.create_chat_completion.call_count == 16


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
        mock.create_chat_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": ""}}]},
                {"choices": [{"delta": {"content": "real"}}]},
                {"choices": [{"delta": {}}]},
            ]
        )
        p = _provider(mock)
        tokens = [t async for t in p.generate_stream("q", "")]
        assert tokens == ["real"]
