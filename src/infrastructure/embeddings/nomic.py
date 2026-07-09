from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from src.infrastructure.embeddings.sentence_transformer_base import (
    SentenceTransformerEmbeddingProvider,
)


class NomicEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    """Nomic-Embed-Text v1.5 via sentence-transformers.

    Produces normalized dense vectors (768-dim by default, Matryoshka
    truncation supported via "matryoshka_dim").

    When switching from BGE-M3, set "embeddings.dense_dim = 768" in
    configs/embeddings.yaml and run
    "python scripts/rebuild_embeddings.py --recreate-collection".
    """

    def __init__(
        self,
        model_path: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = "mps",
        batch_size: int = 32,
        normalize: bool = True,
        matryoshka_dim: int | None = None,
    ) -> None:
        self.model_path: str = model_path
        self.device: str = device
        self.batch_size: int = batch_size
        self.normalize: bool = normalize
        self.matryoshka_dim: int | None = matryoshka_dim
        self._model: SentenceTransformer | None = None

    @override
    def _encode_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.matryoshka_dim is not None:
            kwargs["truncate_dim"] = self.matryoshka_dim
        return kwargs

    @classmethod
    def from_settings(cls) -> NomicEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings
        return cls(
            model_path=cfg.model_path,
            device=cfg.device,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
        )
