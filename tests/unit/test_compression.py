"""T-024 — ContextualCompressor and token_reducer tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.constants import CHUNK_PARENT_ID_KEY, CHUNK_RAW_TEXT_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.rag.chunking.contextual_headers import chunk_context_text
from src.rag.compression.contextual_compression import ContextualCompressor
from src.rag.compression.token_reducer import (
    count_tokens,
    total_tokens,
    truncate_to_tokens,
)
from src.type_regression.compression import (
    check_compressor_returns_chunks,
    check_token_reducer_types,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int, text: str = "") -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=text or f"chunk {i} text")


def _chunks(n: int, text: str = "word " * 20) -> list[Chunk]:
    return [_chunk(i, text) for i in range(n)]


def _compressor(
    response: str = "Extracted relevant sentence.",
    max_tokens: int = 500,
    enabled: bool = True,
) -> ContextualCompressor:
    llm = MagicMock()
    llm.generate.return_value = response
    return ContextualCompressor(llm=llm, max_tokens=max_tokens, enabled=enabled)


# ── token_reducer ──────────────────────────────────────────────────────────────


class TestCountTokens:
    def test_positive_for_nonempty(self):
        assert count_tokens("hello") >= 1

    def test_longer_text_more_tokens(self):
        assert count_tokens("a" * 400) > count_tokens("a" * 40)

    def test_empty_string_returns_one(self):
        assert count_tokens("") == 1


class TestTotalTokens:
    def test_sums_chunk_tokens(self):
        chunks = [_chunk(0, "a" * 40), _chunk(1, "b" * 40)]
        assert total_tokens(chunks) == count_tokens("a" * 40) + count_tokens("b" * 40)

    def test_empty_list_returns_zero(self):
        assert total_tokens([]) == 0


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        text = "Short sentence."
        assert truncate_to_tokens(text, 100) == text

    def test_long_text_truncated(self):
        long_text = "word " * 200
        result = truncate_to_tokens(long_text, 50)
        assert count_tokens(result) <= 50

    def test_prefers_sentence_boundary(self):
        # With a period near the start, the result should end at the period.
        text = "Good sentence. " + "x" * 400
        result = truncate_to_tokens(text, 10)
        assert result.endswith(".")

    def test_zero_max_returns_empty(self):
        assert truncate_to_tokens("some text", 0) == ""

    def test_result_within_budget(self):
        long_text = "x" * 800
        assert count_tokens(truncate_to_tokens(long_text, 100)) <= 100


# ── ContextualCompressor ───────────────────────────────────────────────────────


class TestCompressDisabled:
    async def test_returns_chunks_unchanged(self):
        chunks = _chunks(3)
        comp = _compressor(enabled=False)
        result = await comp.compress("q", chunks)
        assert result is chunks

    async def test_no_llm_call(self):
        comp = _compressor(enabled=False)
        await comp.compress("q", _chunks(2))
        comp._llm.generate.assert_not_called()  # type: ignore[attr-defined]

    async def test_empty_chunks_returns_empty(self):
        assert await _compressor().compress("q", []) == []


class TestCompressEnabled:
    async def test_returns_list_of_chunks(self):
        result = await _compressor().compress("q", _chunks(2))
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    async def test_text_replaced_with_extraction(self):
        result = await _compressor("Relevant extract.").compress("q", [_chunk(0)])
        assert result[0].text == "Relevant extract."

    async def test_chunk_id_preserved(self):
        chunks = [_chunk(7, "some text")]
        result = await _compressor("extracted").compress("q", chunks)
        assert result[0].id == "c7"

    async def test_original_chunk_immutable(self):
        original = _chunk(0, "original text")
        await _compressor("new text").compress("q", [original])
        assert original.text == "original text"

    async def test_output_respects_max_tokens(self):
        long_extract = "word " * 200  # ~50 tokens
        comp = _compressor(response=long_extract, max_tokens=30)
        result = await comp.compress("q", _chunks(5, text="word " * 40))
        assert total_tokens(result) <= 30

    async def test_chunks_beyond_budget_dropped(self):
        # Each chunk extraction is ~50 tokens; budget = 60 → only 1 fits
        comp = _compressor(response="word " * 200, max_tokens=60)
        result = await comp.compress("q", _chunks(5))
        assert len(result) < 5

    async def test_llm_failure_falls_back_to_original(self):
        comp = _compressor()
        comp._llm.generate.side_effect = RuntimeError("LLM down")  # type: ignore[attr-defined]
        original_text = "original chunk text"
        result = await comp.compress("q", [_chunk(0, original_text)])
        assert result[0].text == original_text

    async def test_empty_llm_response_falls_back_to_original(self):
        result = await _compressor(response="").compress("q", [_chunk(0, "original")])
        assert result[0].text == "original"

    async def test_calls_llm_once_per_chunk(self):
        comp = _compressor()
        await comp.compress("q", _chunks(3))
        assert comp._llm.generate.call_count == 3  # type: ignore[attr-defined]

    async def test_extractions_run_concurrently(self):
        """#87 — per-chunk extraction latency must not stack sequentially."""
        import time

        def _slow_extract(query: str, chunk: Chunk) -> str:
            time.sleep(0.05)
            return "extracted"

        comp = _compressor()
        with patch.object(comp, "_extract", side_effect=_slow_extract):
            t0 = time.monotonic()
            await comp.compress("q", _chunks(5))
            elapsed = time.monotonic() - t0
        # Sequential would take ~0.25s; concurrent should stay well under that.
        assert elapsed < 0.2

    async def test_syncs_raw_text_after_compression(self):
        """CCH stores the pre-compression body in raw_text; compression must update it."""
        chunk = Chunk(
            id="c0",
            document_id="doc",
            text="[Document: report.pdf | Section: Revenue | Page: 1]\nLong original body.",
            metadata={CHUNK_RAW_TEXT_KEY: "Long original body."},
        )
        result = await _compressor("Short extract.").compress("q", [chunk])
        assert result[0].text == "Short extract."
        assert result[0].metadata[CHUNK_RAW_TEXT_KEY] == "Short extract."
        assert chunk_context_text(result[0]) == "Short extract."

    async def test_syncs_parent_context_text_after_compression(self):
        """Parent-expanded chunks must keep metadata in sync so LLM context is compressed."""
        chunk = Chunk(
            id="child-0",
            document_id="doc",
            text="child slice.",
            metadata={
                CHUNK_PARENT_ID_KEY: "parent-0",
                PARENT_CONTEXT_TEXT_KEY: "Full parent passage with lots of detail.",
            },
        )
        result = await _compressor("Compressed parent excerpt.").compress("q", [chunk])
        assert result[0].text == "Compressed parent excerpt."
        assert result[0].metadata[PARENT_CONTEXT_TEXT_KEY] == "Compressed parent excerpt."
        assert chunk_context_text(result[0]) == "Compressed parent excerpt."

    async def test_sibling_parent_context_compresses_once(self):
        parent_text = "Full parent passage with lots of detail."
        siblings = [
            Chunk(
                id="child-0",
                document_id="doc",
                text="slice a.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
            Chunk(
                id="child-1",
                document_id="doc",
                text="slice b.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
        ]
        comp = _compressor("Compressed parent excerpt.")
        result = await comp.compress("q", siblings)
        assert len(result) == 2
        assert [c.id for c in result] == ["child-0", "child-1"]
        parent_ctx_a = result[0].metadata[PARENT_CONTEXT_TEXT_KEY]
        parent_ctx_b = result[1].metadata[PARENT_CONTEXT_TEXT_KEY]
        assert parent_ctx_a == parent_ctx_b
        assert comp._llm.generate.call_count == 1  # type: ignore[attr-defined]

    async def test_sibling_parent_context_kept_when_budget_exhausted(self):
        parent_text = "Full parent passage with lots of detail."
        siblings = [
            Chunk(
                id="child-0",
                document_id="doc",
                text="slice a.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
            Chunk(
                id="child-1",
                document_id="doc",
                text="slice b.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
        ]
        long_extract = "word " * 200
        comp = _compressor(response=long_extract, max_tokens=60)
        result = await comp.compress("q", siblings)
        assert len(result) == 2
        assert [c.id for c in result] == ["child-0", "child-1"]
        assert comp._llm.generate.call_count == 1  # type: ignore[attr-defined]

    async def test_parent_hit_and_enriched_child_compress_once(self):
        parent_text = "Full parent passage with lots of detail."
        chunks = [
            Chunk(
                id="parent-0",
                document_id="doc",
                text=parent_text,
            ),
            Chunk(
                id="child-0",
                document_id="doc",
                text="child slice.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
        ]
        comp = _compressor("Compressed parent excerpt.")
        result = await comp.compress("q", chunks)
        assert len(result) == 2
        assert [c.id for c in result] == ["parent-0", "child-0"]
        assert result[0].text == result[1].text == "Compressed parent excerpt."
        assert comp._llm.generate.call_count == 1  # type: ignore[attr-defined]


class TestFromSettings:
    def test_returns_compressor(self):
        llm = MagicMock()
        comp = ContextualCompressor.from_settings(llm)
        assert isinstance(comp, ContextualCompressor)

    def test_uses_settings_max_tokens(self):
        from src.core.settings import settings

        llm = MagicMock()
        comp = ContextualCompressor.from_settings(llm)
        assert comp._max_tokens == settings.compression.max_tokens


class TestTypeRegressionFixtures:
    """Runtime checks for src/type_regression — mypy validates those modules at lint time."""

    def test_token_reducer_api_types(self) -> None:
        total, truncated, count = check_token_reducer_types(_chunks(2))
        assert total > 0 and truncated and count > 0

    async def test_compressor_returns_typed_chunks(self) -> None:
        result = await check_compressor_returns_chunks(_compressor(), "query", _chunks(1))
        assert len(result) == 1
        assert isinstance(result[0].text, str)
