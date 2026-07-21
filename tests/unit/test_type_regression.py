"""Unit tests for src/type_regression typed smoke modules (T-171)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.compression.contextual_compression import ContextualCompressor
from src.type_regression.compression import (
    check_compressor_returns_chunks,
    check_token_reducer_types,
)
from src.type_regression.contextual_headers import (
    check_contextual_headers_api_types,
    check_contextual_headers_chunker_returns_chunks,
)


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(id=f"c{i}", document_id="doc", text=f"chunk {i} text") for i in range(n)]


def _compressor() -> ContextualCompressor:
    llm = MagicMock()
    llm.generate.return_value = "Extracted relevant sentence."
    return ContextualCompressor(llm=llm, max_tokens=500, enabled=True)


def _doc(content: str = "Revenue grew 12% year over year.") -> Document:
    return Document(
        source="/data/raw/annual_report_2023.pdf",
        content=content,
        metadata={"filename": "annual_report_2023.pdf", "section": "Revenue", "page": 42},
    )


def _chunk(text: str = "Revenue grew 12% year over year.") -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=text,
        metadata={"filename": "annual_report_2023.pdf", "section": "Revenue", "page": 42},
    )


class TestCompressionTypeRegression:
    def test_check_token_reducer_types(self):
        total, truncated, count = check_token_reducer_types(_chunks(2))
        assert total > 0
        assert truncated
        assert count > 0

    async def test_check_compressor_returns_chunks(self):
        result = await check_compressor_returns_chunks(_compressor(), "query", _chunks(1))
        assert len(result) == 1
        assert isinstance(result[0].text, str)


class TestContextualHeadersTypeRegression:
    def test_check_contextual_headers_api_types(self):
        header, prefixed, context, key, groups, joined = check_contextual_headers_api_types(
            _doc(), _chunk()
        )
        assert header
        assert prefixed
        assert context
        assert key
        assert groups
        assert joined

    def test_check_contextual_headers_chunker_returns_chunks(self):
        chunks = check_contextual_headers_chunker_returns_chunks(_doc("Body text here."))
        assert len(chunks) >= 1
        assert isinstance(chunks[0].text, str)
