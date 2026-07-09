from __future__ import annotations

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_SOURCE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.metadata import chunk_metadata


# 1 token ≈ 4 characters for English. Fast approximation used to keep chunkers
# free of tokenizer dependencies; replace with tiktoken if exact counts matter.
def _tok(text: str) -> int:
    return max(1, len(text) // 4)


_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """Splits text by trying separators from coarse to fine until pieces fit.

    Mirrors the logic of LangChain's RecursiveCharacterTextSplitter but is
    self-contained and dependency-free.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap

    # ── Public ─────────────────────────────────────────────────────────────────

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self._split(document.content, _SEPARATORS)
        return [
            Chunk(
                document_id=document.id,
                text=text,
                metadata={
                    **chunk_metadata(document.metadata),
                    CHUNK_SOURCE_KEY: document.source,
                    CHUNK_INDEX_KEY: i,
                },
            )
            for i, text in enumerate(texts)
        ]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _split(self, text: str, separators: list[str]) -> list[str]:
        sep, *rest = separators
        raw = text.split(sep) if sep else list(text)

        # Recursively break any piece that is still too large.
        pieces: list[str] = []
        for p in raw:
            if p.strip():
                if _tok(p) > self.chunk_size and rest:
                    pieces.extend(self._split(p, rest))
                else:
                    pieces.append(p)

        return self._merge(pieces, sep)

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        buf: list[str] = []

        def _join(parts: list[str]) -> str:
            return sep.join(parts) if sep else "".join(parts)

        for piece in pieces:
            if buf and _tok(_join([*buf, piece])) > self.chunk_size:
                chunks.append(_join(buf))
                # Trim buffer to at most `overlap` tokens, keeping the tail.
                while len(buf) > 1 and _tok(_join(buf)) > self.overlap:
                    _ = buf.pop(0)
                # If the single remaining item is still > overlap, discard it.
                if buf and _tok(_join(buf)) > self.overlap:
                    buf.clear()

            buf.append(piece)

        if buf:
            chunks.append(_join(buf))

        return [c.strip() for c in chunks if c.strip()]
