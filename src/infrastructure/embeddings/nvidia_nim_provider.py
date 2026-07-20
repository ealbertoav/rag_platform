"""NVIDIA NIM embedding provider (dense only, OpenAI-compatible API)."""

from __future__ import annotations

import logging
from typing import Any, Literal, override

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

logger = logging.getLogger(__name__)

_MAX_BATCH = 96  # NIM has no documented per-request limit; kept conservative, like other providers
NimInputType = Literal["query", "passage"]


def _is_rate_limit(exc: BaseException) -> bool:
    try:
        from openai import RateLimitError

        return isinstance(exc, RateLimitError)
    except ImportError:
        msg = str(exc).lower()
        return any(kw in msg for kw in ("429", "rate_limit", "rate limit", "too many requests"))


class NvidiaNimEmbeddingProvider(EmbeddingRepository):
    """NVIDIA NIM embed API provider (OpenAI-compatible "/v1/embeddings").

    Dense only — sparse vectors fall back to BM25.
    Uses input_type="passage" for ingestion and "query" for query embedding,
    passed via the OpenAI client's extra_body (NIM's input_type isn't part of
    the standard OpenAI embeddings signature).
    Retries on HTTP 429 with exponential backoff via tenacity.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ) -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.base_url: str = base_url
        self._client: Any | None = None

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    @override
    def embed(self, texts: list[str]) -> list[DenseVector]:
        """Embed texts for document storage (input_type=passage)."""
        return self._embed_with_type(texts, "passage")

    @override
    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{} for _ in texts]

    @override
    def embed_query(self, texts: list[str]) -> list[DenseVector]:
        """Embed query texts (input_type=query). Use during retrieval."""
        return self._embed_with_type(texts, "query")

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> NvidiaNimEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings.nvidia_nim
        return cls(api_key=cfg.api_key.get_secret_value(), model=cfg.model, base_url=cfg.base_url)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _embed_with_type(self, texts: list[str], input_type: NimInputType) -> list[DenseVector]:
        if not texts:
            return []
        results: list[DenseVector] = []
        for i in range(0, len(texts), _MAX_BATCH):
            results.extend(self._embed_batch(texts[i : i + _MAX_BATCH], input_type))
        return results

    def _embed_batch(self, texts: list[str], input_type: NimInputType) -> list[DenseVector]:
        try:
            return self._call_with_retry(texts, input_type)
        except Exception as exc:
            raise EmbeddingError(
                f"NVIDIA NIM embed failed for {len(texts)} texts", cause=exc
            ) from exc

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_with_retry(self, texts: list[str], input_type: NimInputType) -> list[DenseVector]:
        return self._call_api(texts, input_type)

    def _call_api(self, texts: list[str], input_type: NimInputType) -> list[DenseVector]:
        client = self._get_client()
        response = client.embeddings.create(
            input=texts,
            model=self.model,
            extra_body={"input_type": input_type},
        )
        return [list(item.embedding) for item in sorted(response.data, key=lambda x: x.index)]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingError(
                    "openai package is not installed. Run: uv sync --extra api-embeddings"
                ) from exc
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
