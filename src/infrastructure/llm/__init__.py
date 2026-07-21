from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.repositories.llm_repository import LLMRepository

if TYPE_CHECKING:
    from src.core.settings import Settings


def get_llm_provider(settings: Settings | None = None) -> LLMRepository:
    """Build the LLM provider selected by settings.llm.provider.

    Shared by ChatPipeline.from_settings() and scripts/run_evals.py so
    provider selection (nvidia_nim vs. self-hosted llama.cpp) lives in one
    place. Providers are imported lazily to avoid loading optional
    dependencies (llama-cpp-python, the NIM client) at module load time.
    """
    if settings is None:
        from src.core.settings import settings as _settings

        settings = _settings

    if settings.llm.provider == "nvidia_nim":
        from src.infrastructure.llm.nvidia_nim_provider import NvidiaNimProvider

        return NvidiaNimProvider.from_settings()

    from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

    return LlamaCppProvider.from_settings()
