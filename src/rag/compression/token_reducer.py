from __future__ import annotations

from src.domain.entities.chunk import Chunk

# 1 token ≈ 4 characters for English.  Fast approximation — replace with
# tiktoken if exact per-model counts are needed.
_CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    """Approximate token count for *text*."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def total_tokens(chunks: list[Chunk]) -> int:
    """Sum of approximate token counts across all chunks."""
    return sum(count_tokens(c.text) for c in chunks)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens*, preferring sentence boundaries."""
    if max_tokens <= 0:
        return ""
    target_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= target_chars:
        return text
    truncated = text[:target_chars]
    # Try to end on a sentence boundary so the text reads cleanly.
    last_period = truncated.rfind(".")
    if last_period > 0:
        return truncated[: last_period + 1].strip()
    return truncated.strip()
