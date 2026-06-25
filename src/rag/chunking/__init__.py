from __future__ import annotations

from typing import Protocol

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.contextual_headers import ContextualHeadersChunker
from src.rag.chunking.parent_child_chunker import ParentChildChunker
from src.rag.chunking.recursive_chunker import RecursiveChunker
from src.rag.chunking.semantic_chunker import SemanticChunker


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


def get_chunker(
    strategy: str = "recursive",
    *,
    use_contextual_headers: bool = False,
    **kwargs: object,
) -> Chunker:
    """Return a chunker for *strategy*, forwarding *kwargs* to its constructor."""
    match strategy:
        case "recursive":
            chunker: Chunker = RecursiveChunker(**kwargs)  # type: ignore[arg-type]
        case "semantic":
            chunker = SemanticChunker(**kwargs)  # type: ignore[arg-type]
        case "parent_child":
            chunker = ParentChildChunker(**kwargs)  # type: ignore[arg-type]
        case _:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    if use_contextual_headers:
        chunker = ContextualHeadersChunker(chunker)
    return chunker


__all__ = [
    "Chunker",
    "ContextualHeadersChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "get_chunker",
]
