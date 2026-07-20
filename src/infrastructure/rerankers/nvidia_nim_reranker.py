"""NVIDIA NIM reranker provider (own request shape, not OpenAI-compatible)."""

from __future__ import annotations

import logging
from typing import Any, override

import httpx

from src.core.exceptions import ConfigurationError, RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0


class NvidiaNimRerankerProvider(RerankerRepository):
    """RerankerRepository backed by NVIDIA NIM's "/v1/ranking" endpoint.

    Not OpenAI-compatible — NIM's ranking endpoint has its own request shape
    (query.text + passages[].text) and response shape ({"rankings": [{"index",
    "logit"}, ...]}, matching NVIDIA's documented NeMo Retriever Reranking NIM
    API — not yet exercised against a live response, verify in #79), so this
    uses a plain httpx client rather than the OpenAI SDK the LLM/embedding NIM
    providers use.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ConfigurationError(
                "Reranker provider 'nvidia_nim' requires reranker.nvidia_nim.api_key "
                "(set RERANKER__NVIDIA_NIM__API_KEY)"
            )
        self.api_key: str = normalized_key
        self.model: str = model
        self.base_url: str = base_url.rstrip("/")
        self.timeout_seconds: float = timeout_seconds
        self._client: httpx.Client | None = client
        self._owns_client: bool = client is None

    # ── RerankerRepository interface ───────────────────────────────────────────

    @override
    def score(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        """Return cross-encoder relevance scores for each chunk (input order)."""
        if not chunks:
            return []
        try:
            rankings = self._rank(query, [c.text for c in chunks])
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(
                f"NVIDIA NIM ranking failed with HTTP {exc.response.status_code} "
                f"for {len(chunks)} chunks",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise RetrievalError(
                f"NVIDIA NIM ranking failed for {len(chunks)} chunks", cause=exc
            ) from exc
        scores = [0.0] * len(chunks)
        for item in rankings:
            scores[int(item["index"])] = float(item["logit"])
        return list(zip(chunks, scores, strict=True))

    @override
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Return up to *top_k* chunks sorted by NIM's relevance score."""
        ranked = sorted(self.score(query, chunks), key=lambda x: x[1], reverse=True)
        logger.debug("NVIDIA NIM reranker: %d chunks → keeping top %d", len(chunks), top_k)
        return [c for c, _ in ranked[:top_k]]

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> NvidiaNimRerankerProvider:
        from src.core.settings import settings

        cfg = settings.reranker.nvidia_nim
        return cls(api_key=cfg.api_key.get_secret_value(), model=cfg.model, base_url=cfg.base_url)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _rank(self, query: str, texts: list[str]) -> list[dict[str, Any]]:
        response = self._http().post(
            f"{self.base_url}/ranking",
            headers=self._headers(),
            json={
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": t} for t in texts],
            },
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return list(data["rankings"])

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_seconds)
            self._owns_client = True
        return self._client

    def close(self) -> None:
        """Close the owned httpx client, if any."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
