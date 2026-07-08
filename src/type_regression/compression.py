"""Typed smoke checks for compression APIs — validated by mypy at lint time (T-171)."""

from __future__ import annotations

from src.domain.entities.chunk import Chunk
from src.rag.compression.contextual_compression import ContextualCompressor
from src.rag.compression.token_reducer import count_tokens, total_tokens, truncate_to_tokens


def check_token_reducer_types(chunks: list[Chunk]) -> tuple[int, str, int]:
    """Exercise token_reducer return types against real chunk inputs."""
    total: int = total_tokens(chunks)
    truncated: str = truncate_to_tokens("hello world", 5)
    count: int = count_tokens("hello")
    return total, truncated, count


def check_compressor_returns_chunks(
    comp: ContextualCompressor,
    query: str,
    chunks: list[Chunk],
) -> list[Chunk]:
    """Exercise ContextualCompressor.compress return type."""
    result: list[Chunk] = comp.compress(query, chunks)
    return result
