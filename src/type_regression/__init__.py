"""Typed smoke modules for API regression detection (T-171).

Mypy analyzes these modules via ``uv run mypy src``. Unit tests call the
check functions to assert runtime behavior still matches the typed contracts.
"""

from src.type_regression.compression import (
    check_compressor_returns_chunks,
    check_token_reducer_types,
)
from src.type_regression.contextual_headers import (
    check_contextual_headers_api_types,
    check_contextual_headers_chunker_returns_chunks,
)

__all__ = [
    "check_compressor_returns_chunks",
    "check_contextual_headers_api_types",
    "check_contextual_headers_chunker_returns_chunks",
    "check_token_reducer_types",
]
