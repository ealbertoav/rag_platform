"""CLIP multimodal embedding provider (text + image, shared embedding space)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import DenseVector
from src.infrastructure.embeddings.sentence_transformer_base import (
    SentenceTransformerEmbeddingProvider,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sentence_transformers import SentenceTransformer


class ClipEmbeddingProvider(SentenceTransformerEmbeddingProvider):
    """OpenAI CLIP via sentence-transformers.

    Encodes text and images into the same 512-dim space (clip-ViT-B-32
    default), so embed() and embed_image() outputs are directly comparable —
    unlike text-only providers, which raise on embed_image() (T-250).
    """

    def __init__(
        self,
        model_path: str = "sentence-transformers/clip-ViT-B-32",
        device: str = "mps",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_path: str = model_path
        self.device: str = device
        self.batch_size: int = batch_size
        self.normalize: bool = normalize
        self._model: SentenceTransformer | None = None

    @override
    def _encode_kwargs(self) -> dict[str, Any]:
        return {}

    @override
    def embed_image(self, paths: list[Path]) -> list[DenseVector]:
        """Return one dense vector per image, in the same space as embed()."""
        if not paths:
            return []
        from PIL import Image

        try:
            images = [Image.open(path).convert("RGB") for path in paths]
        except (OSError, ValueError) as exc:
            raise EmbeddingError(f"Cannot load image for {type(self).__name__}", cause=exc) from exc

        model = self._get_model()
        try:
            vecs = model.encode(
                images,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
            )
            return [v.tolist() for v in vecs]
        except Exception as exc:
            raise EmbeddingError(
                f"{type(self).__name__} image encode failed for {len(paths)} images", cause=exc
            ) from exc

    @classmethod
    def from_settings(cls) -> ClipEmbeddingProvider:
        from src.core.settings import settings

        cfg = settings.embeddings
        return cls(
            model_path=cfg.model_path,
            device=cfg.device,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
        )
