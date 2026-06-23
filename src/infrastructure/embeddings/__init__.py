from __future__ import annotations

from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
from src.infrastructure.embeddings.nomic import NomicEmbeddingProvider
from src.infrastructure.embeddings.qwen_embedding import QwenEmbeddingProvider


def get_embedding_provider() -> EmbeddingRepository:
    """Return the configured embedding provider based on "settings.embeddings.provider".

    Supported values (set in configs/embeddings.yaml or via env var
    "EMBEDDINGS__PROVIDER"):

        bge_m3         — BGE-M3 1024-dim dense and sparse (default)
        nomic          — Nomic-Embed-Text v1.5, 768-dim dense only
        qwen_embedding — Qwen3-Embedding-0.6B, 1024-dim dense only
                         (or Qwen/Qwen3-Embedding, 4096-dim)

    Note: when switching providers, update "embeddings.dense_dim" and
    "embeddings.model_path" to match, then run
    "python scripts/rebuild_embeddings.py --recreate-collection".
    """
    from src.core.settings import settings

    provider = settings.embeddings.provider
    match provider:
        case "bge_m3":
            return BGEM3EmbeddingProvider.from_settings()
        case "nomic":
            return NomicEmbeddingProvider.from_settings()
        case "qwen_embedding":
            return QwenEmbeddingProvider.from_settings()
        case _:
            raise ValueError(f"Unknown embedding provider: {provider!r}")


__all__ = [
    "BGEM3EmbeddingProvider",
    "NomicEmbeddingProvider",
    "QwenEmbeddingProvider",
    "get_embedding_provider",
]
