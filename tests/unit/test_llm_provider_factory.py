"""Unit tests for src/infrastructure/llm/__init__.py's get_llm_provider() (#96 follow-up).

Shared by ChatPipeline.from_settings() and scripts/run_evals.py so provider
selection (nvidia_nim vs. self-hosted llama.cpp) lives in one place.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.infrastructure.llm import get_llm_provider


class TestGetLlmProvider:
    def test_uses_nvidia_nim_when_configured(self):
        settings = MagicMock()
        settings.llm.provider = "nvidia_nim"
        with patch(
            "src.infrastructure.llm.nvidia_nim_provider.NvidiaNimProvider.from_settings",
            return_value="nim-llm",
        ) as mock_nim:
            result = get_llm_provider(settings)
        mock_nim.assert_called_once()
        assert result == "nim-llm"

    def test_defaults_to_llama_cpp(self):
        settings = MagicMock()
        settings.llm.provider = "llama_cpp"
        with patch(
            "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings",
            return_value="llama-cpp-llm",
        ) as mock_llama_cpp:
            result = get_llm_provider(settings)
        mock_llama_cpp.assert_called_once()
        assert result == "llama-cpp-llm"

    def test_reads_global_settings_when_omitted(self):
        settings = MagicMock()
        settings.llm.provider = "nvidia_nim"
        with (
            patch("src.core.settings.settings", settings),
            patch(
                "src.infrastructure.llm.nvidia_nim_provider.NvidiaNimProvider.from_settings",
                return_value="nim-llm",
            ) as mock_nim,
        ):
            result = get_llm_provider()
        mock_nim.assert_called_once()
        assert result == "nim-llm"
