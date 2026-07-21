"""Unit tests for src/evals/generation/nim_judge.py (#104)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from src.evals.generation.nim_judge import NimJudgeConfigError, build_nim_judge_llm
from src.infrastructure.llm.nvidia_nim_provider import NvidiaNimProvider


@pytest.fixture(autouse=True)
def _clear_judge_cache():
    build_nim_judge_llm.cache_clear()
    yield
    build_nim_judge_llm.cache_clear()


def _configure_valid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh import, not a module-level `from src.core.settings import settings` —
    # other tests (e.g. test_docling_parser.py's temporary_config) reload
    # src.core.settings at runtime, which rebinds the singleton to a new object.
    # A reference captured at collection time would silently go stale.
    from src.core.settings import settings

    monkeypatch.setattr(settings.llm.nvidia_nim, "api_key", SecretStr("llm-key"))
    monkeypatch.setattr(settings.llm.nvidia_nim, "model", "meta/llama-3.1-8b-instruct")
    monkeypatch.setattr(settings.llm.nvidia_nim, "base_url", "https://integrate.api.nvidia.com/v1")


class TestBuildNimJudgeLlm:
    def test_returns_nvidia_nim_provider_from_settings(self, monkeypatch: pytest.MonkeyPatch):
        _configure_valid_settings(monkeypatch)

        judge = build_nim_judge_llm()

        assert isinstance(judge, NvidiaNimProvider)
        assert judge.model == "meta/llama-3.1-8b-instruct"
        assert judge.api_key == "llm-key"
        assert judge.base_url == "https://integrate.api.nvidia.com/v1"

    def test_raises_clear_error_when_api_key_missing(self, monkeypatch: pytest.MonkeyPatch):
        from src.core.settings import settings

        _configure_valid_settings(monkeypatch)
        monkeypatch.setattr(settings.llm.nvidia_nim, "api_key", SecretStr(""))

        with pytest.raises(NimJudgeConfigError, match="LLM__NVIDIA_NIM__API_KEY"):
            build_nim_judge_llm()

    def test_is_constructed_once_and_cached(self, monkeypatch: pytest.MonkeyPatch):
        _configure_valid_settings(monkeypatch)

        first = build_nim_judge_llm()
        second = build_nim_judge_llm()

        assert first is second

    def test_ignores_active_provider_setting(self, monkeypatch: pytest.MonkeyPatch):
        """The judge is NVIDIA-NIM-backed regardless of settings.llm.provider (#103)."""
        from src.core.settings import settings

        _configure_valid_settings(monkeypatch)
        monkeypatch.setattr(settings.llm, "provider", "llama_cpp")

        judge = build_nim_judge_llm()

        assert isinstance(judge, NvidiaNimProvider)
