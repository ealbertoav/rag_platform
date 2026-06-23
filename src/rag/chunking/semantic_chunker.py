from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_SOURCE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.recursive_chunker import RecursiveChunker

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class SemanticChunker:
    """Splits text at topic boundaries detected by drops in sentence-embedding similarity.

    The *encoded* parameter accepts any callable "(list[str]) -> array-like"
    so the default SentenceTransformer can be swapped out in tests without
    downloading a model.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.4,
        max_tokens: int = 500,
        encode: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self._encode = encode  # injected encoder — lazy-loads real model when None

    # ── Public ─────────────────────────────────────────────────────────────────

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = _SENTENCE_RE.split(document.content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []

        embeddings = np.array(self._get_encode()(sentences))

        # Find sentence indices where similarity drops below threshold.
        groups: list[list[str]] = []
        current: list[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            dist = 1.0 - _cosine(embeddings[i - 1], embeddings[i])
            if dist > self.similarity_threshold:
                groups.append(current)
                current = []
            current.append(sentences[i])
        groups.append(current)

        # Build chunks, further splitting groups that exceed max_tokens.
        splitter = RecursiveChunker(chunk_size=self.max_tokens, overlap=0)
        chunks: list[Chunk] = []
        for group in groups:
            text = " ".join(group)
            temp_doc = Document(source=document.source, content=text)
            sub_texts = [c.text for c in splitter.chunk(temp_doc)]
            for text_part in sub_texts:
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=text_part,
                        metadata={
                            **document.metadata,
                            CHUNK_SOURCE_KEY: document.source,
                            CHUNK_INDEX_KEY: len(chunks),
                        },
                    )
                )

        return chunks

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_encode(self) -> Callable[[list[str]], Any]:
        if self._encode is not None:
            return self._encode
        from sentence_transformers import SentenceTransformer  # lazy import
        model = SentenceTransformer(self.model_name)
        def encode(texts: list[str]) -> Any:
            return model.encode(texts)

        self._encode = encode
        return encode
