from __future__ import annotations

from typing import Any

from src.infrastructure.embeddings.sentence_transformer_base import (
    SentenceTransformerEmbeddingProvider,
)

_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class QwenEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    """Qwen3-Embedding via sentence-transformers.

    Produces dense vectors that pair naturally with the Qwen3 LLM used for
    generation.  Available sizes:

        Qwen/Qwen3-Embedding-0.6B  → 1024-dim (default, matches BGE-M3 dim)
        Qwen/Qwen3-Embedding       → 4096-dim (update embeddings.dense_dim)

    When switching from BGE-M3, verify "embeddings.dense_dim" matches the
    model's output dimension and run
    "python scripts/rebuild_embeddings.py --recreate-collection".
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        device: str = "mps",
        batch_size: int = 32,
        normalize: bool = True,
        max_length: int = 8192,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_length = max_length
        self._model = None

    def _encode_kwargs(self) -> dict[str, Any]:
        return {"max_length": self.max_length}

    @classmethod
    def from_settings(cls) -> QwenEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings
        return cls(
            model_path=cfg.model_path,
            device=cfg.device,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
        )
