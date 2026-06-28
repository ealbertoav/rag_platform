"""T-124 — parent context resolver tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.retrieval_service import RetrievalService
from src.rag.chunking.contextual_headers import chunk_context_text, join_chunk_context
from src.rag.enrichment.parent_context_resolver import (
    ChunkLookup,
    drop_redundant_parent_hits,
    enrich_with_parent_context,
)
from src.rag.enrichment.relevant_segment_extraction import chunk_source_ids


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    text: str = "chunk text",
    parent_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    chunk_metadata: dict[str, object] = dict(metadata or {})
    if parent_id is not None:
        chunk_metadata[CHUNK_PARENT_ID_KEY] = parent_id
    return Chunk(id=chunk_id, document_id=document_id, text=text, metadata=chunk_metadata)


class _StubLookup:
    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks
        self.get_by_id_calls: list[str] = []

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        self.get_by_id_calls.append(chunk_id)
        return self._chunks.get(chunk_id)


def _dense_mock() -> MagicMock:
    mock = MagicMock()
    mock.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
    return mock


def _hybrid_mock(chunks: list[Chunk]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve = AsyncMock(return_value=[(chunk, 0.9) for chunk in chunks])
    return mock


def _parent_context_service(
    chunks: list[Chunk],
    lookup: ChunkLookup,
    *,
    parent_context_enabled: bool = False,
    parent_child_strategy: bool = True,
    **kwargs: object,
) -> RetrievalService:
    return RetrievalService(
        dense_retriever=_dense_mock(),
        hybrid_retriever=_hybrid_mock(chunks),
        parent_context_enabled=parent_context_enabled,
        parent_child_strategy=parent_child_strategy,
        chunk_lookup=lookup,
        **kwargs,
    )


async def _retrieve(svc: RetrievalService, query_text: str = "test"):
    return await svc.retrieve(Query(text=query_text))


def _parent_with_sibling_children(
    parent_text: str,
    *,
    child_a_text: str = "slice a.",
    child_b_text: str = "slice b.",
) -> tuple[Chunk, Chunk, Chunk, _StubLookup]:
    parent = _chunk("parent-0", text=parent_text)
    child_a = _chunk("child-0", text=child_a_text, parent_id="parent-0")
    child_b = _chunk("child-1", text=child_b_text, parent_id="parent-0")
    lookup = _StubLookup({"parent-0": parent})
    return parent, child_a, child_b, lookup


def _enrich_sibling_children(
    parent_text: str,
    *,
    child_a_text: str = "slice a.",
    child_b_text: str = "slice b.",
) -> tuple[list[Chunk], int, _StubLookup]:
    _, child_a, child_b, lookup = _parent_with_sibling_children(
        parent_text,
        child_a_text=child_a_text,
        child_b_text=child_b_text,
    )
    enriched, resolved = enrich_with_parent_context([child_a, child_b], lookup)
    return enriched, resolved, lookup


class TestEnrichWithParentContext:
    def test_empty_list(self):
        lookup = _StubLookup({})
        enriched, resolved = enrich_with_parent_context([], lookup)
        assert enriched == []
        assert resolved == 0

    def test_chunk_without_parent_id_unchanged(self):
        child = _chunk("child-0", text="child only")
        lookup = _StubLookup({})
        enriched, resolved = enrich_with_parent_context([child], lookup)
        assert enriched == [child]
        assert resolved == 0

    def test_child_enriched_with_parent_text(self):
        parent = _chunk("parent-0", text="full parent passage with broader context.")
        child = _chunk("child-0", text="narrow child slice.", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})

        enriched, resolved = enrich_with_parent_context([child], lookup)
        assert resolved == 1
        assert enriched[0].id == "child-0"
        assert enriched[0].text == "narrow child slice."
        assert enriched[0].metadata[PARENT_CONTEXT_TEXT_KEY] == parent.text
        assert chunk_context_text(enriched[0]) == parent.text

    def test_falls_back_to_child_when_parent_missing(self):
        child = _chunk("child-0", text="child fallback text.", parent_id="missing-parent")
        lookup = _StubLookup({})

        enriched, resolved = enrich_with_parent_context([child], lookup)
        assert resolved == 0
        assert enriched[0] == child
        assert chunk_context_text(enriched[0]) == child.text

    def test_preserves_child_id_for_citations(self):
        parent = _chunk("parent-0", text="parent body.")
        child = _chunk("child-42", text="child body.", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})

        enriched, _ = enrich_with_parent_context([child], lookup)
        assert chunk_source_ids(enriched[0]) == ["child-42"]

    def test_uses_parent_raw_text_when_cch_applied(self):
        parent = _chunk(
            "parent-0",
            text="[Document: Report | Section: — | Page: —]\nparent raw",
            metadata={"raw_text": "parent raw"},
        )
        child = _chunk("child-0", text="child slice.", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})

        enriched, resolved = enrich_with_parent_context([child], lookup)
        assert resolved == 1
        assert enriched[0].metadata[PARENT_CONTEXT_TEXT_KEY] == "parent raw"

    def test_sibling_children_deduplicated_in_context(self):
        enriched, resolved, _ = _enrich_sibling_children("shared parent body.")
        assert resolved == 2
        assert join_chunk_context(enriched) == "shared parent body."
        assert [c.id for c in enriched] == ["child-0", "child-1"]

    def test_parent_lookup_cached_for_siblings(self):
        enriched, resolved, lookup = _enrich_sibling_children("shared parent body.")
        assert resolved == 2
        assert lookup.get_by_id_calls == ["parent-0"]
        ctx_a = enriched[0].metadata[PARENT_CONTEXT_TEXT_KEY]
        ctx_b = enriched[1].metadata[PARENT_CONTEXT_TEXT_KEY]
        assert ctx_a == ctx_b

    def test_empty_parent_text_falls_back_to_child(self):
        parent = _chunk("parent-0", text="")
        child = _chunk("child-0", text="child fallback text.", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})

        enriched, resolved = enrich_with_parent_context([child], lookup)
        assert resolved == 0
        assert enriched[0] == child
        assert PARENT_CONTEXT_TEXT_KEY not in enriched[0].metadata
        assert chunk_context_text(enriched[0]) == "child fallback text."

    def test_empty_parent_text_siblings_keep_distinct_child_context(self):
        enriched, resolved, _ = _enrich_sibling_children("")
        assert resolved == 0
        assert join_chunk_context(enriched) == "slice a.\n\nslice b."


class TestDropRedundantParentHits:
    def test_noop_without_enriched_children(self):
        parent = _chunk("parent-0", text="parent body.")
        child = _chunk("child-0", text="child slice.", parent_id="parent-0")
        assert drop_redundant_parent_hits([parent, child]) == [parent, child]

    def test_drops_parent_when_enriched_child_present(self):
        parent = _chunk("parent-0", text="parent body.")
        child = _chunk("child-0", text="child slice.", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})
        enriched, _ = enrich_with_parent_context([parent, child], lookup)
        deduped = drop_redundant_parent_hits(enriched)
        assert [c.id for c in deduped] == ["child-0"]
        assert join_chunk_context(deduped) == "parent body."


class TestRetrievalServiceParentContext:
    @pytest.mark.asyncio
    async def test_disabled_leaves_child_text(self):
        child = _chunk("child-0", text="child text.", parent_id="parent-0")
        parent = _chunk("parent-0", text="parent text.")
        lookup = _StubLookup({"parent-0": parent})

        result = await _retrieve(
            _parent_context_service([child], lookup, parent_context_enabled=False)
        )
        assert chunk_context_text(result.chunks[0]) == "child text."
        assert lookup.get_by_id_calls == []

    @pytest.mark.asyncio
    async def test_enabled_without_parent_child_strategy_is_noop(self):
        child = _chunk("child-0", text="child text.", parent_id="parent-0")
        lookup = _StubLookup({})

        result = await _retrieve(
            _parent_context_service(
                [child],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=False,
            )
        )
        assert chunk_context_text(result.chunks[0]) == "child text."
        assert lookup.get_by_id_calls == []

    @pytest.mark.asyncio
    async def test_enabled_expands_child_to_parent_in_context(self):
        child = _chunk("child-0", text="child slice.", parent_id="parent-0")
        parent = _chunk("parent-0", text="full parent context passage.")
        lookup = _StubLookup({"parent-0": parent})

        result = await _retrieve(
            _parent_context_service(
                [child],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
            )
        )
        assert result.context == "full parent context passage."
        assert result.chunks[0].id == "child-0"
        assert lookup.get_by_id_calls == ["parent-0"]

    @pytest.mark.asyncio
    async def test_sources_remain_child_ids(self):
        child = _chunk("child-99", text="child slice.", parent_id="parent-0")
        parent = _chunk("parent-0", text="parent passage.")
        lookup = _StubLookup({"parent-0": parent})

        result = await _retrieve(
            _parent_context_service(
                [child],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
            )
        )
        assert chunk_source_ids(result.chunks[0]) == ["child-99"]

    @pytest.mark.asyncio
    async def test_runs_after_rse_before_compression(self):
        child_a = _chunk(
            "child-0", text="part one", parent_id="parent-0", metadata={"chunk_index": 0}
        )
        child_b = _chunk(
            "child-1", text="part two", parent_id="parent-0", metadata={"chunk_index": 1}
        )
        parent = _chunk("parent-0", text="parent body.")
        lookup = _StubLookup({"parent-0": parent})
        compressor = MagicMock()
        compressor.compress.side_effect = lambda _q, cs: cs

        result = await _retrieve(
            _parent_context_service(
                [child_a, child_b],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
                compressor=compressor,
                rse_enabled=True,
                rse_max_segment_tokens=500,
            )
        )
        # RSE merges siblings; parent context then resolves merged chunk to parent text
        assert len(result.chunks) == 1
        assert result.context == "parent body."
        compressor.compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_compression_applied_to_parent_context(self):
        child = _chunk("child-0", text="child slice.", parent_id="parent-0")
        parent = _chunk("parent-0", text="full parent context passage with extra detail.")
        lookup = _StubLookup({"parent-0": parent})
        compressor = MagicMock()
        compressor.compress.side_effect = lambda _q, cs: [
            c.model_copy(
                update={
                    "text": "compressed parent excerpt.",
                    "metadata": {
                        **c.metadata,
                        PARENT_CONTEXT_TEXT_KEY: "compressed parent excerpt.",
                    },
                }
            )
            for c in cs
        ]

        result = await _retrieve(
            _parent_context_service(
                [child],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
                compressor=compressor,
            )
        )
        assert result.context == "compressed parent excerpt."
        assert chunk_context_text(result.chunks[0]) == "compressed parent excerpt."

    @pytest.mark.asyncio
    async def test_sibling_children_emit_parent_context_once(self):
        _, child_a, child_b, lookup = _parent_with_sibling_children("shared parent passage.")

        result = await _retrieve(
            _parent_context_service(
                [child_a, child_b],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
            )
        )
        assert result.context == "shared parent passage."
        assert len(result.chunks) == 2
        assert [c.id for c in result.chunks] == ["child-0", "child-1"]

    @pytest.mark.asyncio
    async def test_parent_hit_deduped_when_enriched_child_present(self):
        parent = _chunk("parent-0", text="shared parent passage.")
        child = _chunk("child-0", text="child slice.", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})

        result = await _retrieve(
            _parent_context_service(
                [parent, child],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
            )
        )
        assert result.context == "shared parent passage."
        assert [c.id for c in result.chunks] == ["child-0"]

    @pytest.mark.asyncio
    async def test_sibling_children_survive_tight_compression_budget(self):
        from src.rag.compression.contextual_compression import ContextualCompressor

        _, child_a, child_b, lookup = _parent_with_sibling_children("shared parent passage " * 100)
        llm = MagicMock()
        llm.generate.return_value = "compressed excerpt."
        compressor = ContextualCompressor(llm=llm, max_tokens=10, enabled=True)

        result = await _retrieve(
            _parent_context_service(
                [child_a, child_b],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
                compressor=compressor,
            )
        )
        assert len(result.chunks) == 2
        assert [c.id for c in result.chunks] == ["child-0", "child-1"]
        assert result.context == "compressed excerpt."
        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_parent_falls_back_to_child_context(self):
        _, child_a, child_b, lookup = _parent_with_sibling_children("")

        result = await _retrieve(
            _parent_context_service(
                [child_a, child_b],
                lookup,
                parent_context_enabled=True,
                parent_child_strategy=True,
            )
        )
        assert result.context == "slice a.\n\nslice b."
        assert len(result.chunks) == 2
