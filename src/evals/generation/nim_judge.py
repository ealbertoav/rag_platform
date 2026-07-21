"""NVIDIA NIM as the eval judge LLM (#103/#104).

ragas turned out to be unusable in this project's dependency environment —
ragas.llms.base unconditionally imports langchain_community.chat_models.vertexai
.ChatVertexAI (removed from any langchain-community version compatible with
this project's numpy>=2.4.6) and ragas.metrics imports langchain_core.pydantic_v1
(removed from modern langchain-core). Both are upstream ragas packaging bugs,
confirmed across ragas 0.1.22 and the latest 0.4.3 — not fixable by pinning.

Faithfulness/Relevance/ContextPrecision instead call NVIDIA NIM directly via
the same LLMRepository interface already used for live generation, with no
ragas/langchain dependency at all. Judging is offline batch work — not the
hot path ADR-0003 measured NIM as slower for. The judge always uses
settings.llm.nvidia_nim directly, independent of whichever provider is
active for live generation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.repositories.llm_repository import LLMRepository


class NimJudgeConfigError(RuntimeError):
    """Raised when NVIDIA NIM credentials needed for eval judging are missing."""


@lru_cache(maxsize=1)
def build_nim_judge_llm() -> LLMRepository:
    """Build the shared NVIDIA-NIM-backed judge LLM.

    Constructed once per process and cached — repeated calls across many
    samples reuse the same client instead of rebuilding it every score().
    Tests must call `build_nim_judge_llm.cache_clear()` between cases.
    """
    from src.core.settings import settings
    from src.infrastructure.llm.nvidia_nim_provider import NvidiaNimProvider

    cfg = settings.llm.nvidia_nim
    api_key = cfg.api_key.get_secret_value()
    if not api_key:
        raise NimJudgeConfigError(
            "settings.llm.nvidia_nim.api_key is empty — set LLM__NVIDIA_NIM__API_KEY "
            "to use NVIDIA NIM as the eval judge."
        )
    return NvidiaNimProvider(api_key=api_key, model=cfg.model, base_url=cfg.base_url)
