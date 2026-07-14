"""T-202 — Structured table chunks at ingesting."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import (
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_TABLE,
    TABLE_ID_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.ingestion.table_chunker import (
    TableChunker,
    build_table_chunks,
    built_table_ids,
    collect_table_ids,
    existing_table_chunk_ids,
    extract_markdown_tables,
    is_table_chunk,
    known_table_chunk_ids,
    merged_table_chunk_ids,
    metadata_table_ids,
    retained_table_chunk_ids_on_embed_failure,
    stale_table_ids_safe_to_purge,
    table_build_succeeded,
    table_chunk_id,
    table_chunks_needing_upsert,
    table_embedding_succeeded,
    table_sync_succeeded,
)
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline, IngestionResult
from tests.unit.ingestion_helpers import (
    assert_purges_only_stale_text_chunk,
    assert_skip_purged_only_structured_chunks,
    assert_skip_without_reindex,
    embedded_chunk,
    full_reindex_on_skip_preserves_stable_structured_chunk_ids,
    ingest_without_structured_chunker_indexes_text_only,
    ingestion_pipeline_from_settings,
    mock_chunker_with_empty_index,
    mock_ingestion_pipeline,
    mock_reingest_metadata,
    reingest_preserves_stable_structured_chunks_in_real_bm25,
    run_reingest_with_empty_structured_chunker,
    run_skip_purge_only_structured_chunker_ingest,
    run_skip_structured_chunker_ingest,
    run_skip_with_indexed_structured_chunks,
    run_structured_chunker_ingest,
    skip_backfills_missing_structured_chunks_on_unchanged_hash,
    skip_purges_stale_structured_chunks_when_layout_changes,
    skip_unchanged_without_structured_chunker,
    unchanged_hash_metadata,
)

_TABLE_CHUNKER = "src.rag.ingestion.table_chunker"
_INGESTION_PIPELINE = "src.rag.pipelines.ingestion_pipeline"


def _internal(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


resolve_table_text = cast(
    Callable[[dict[str, Any], str | None], str | None],
    _internal(_TABLE_CHUNKER, "_resolve_table_text"),
)

content_table_for_id = cast(
    Callable[[str, list[str]], str | None],
    _internal(_TABLE_CHUNKER, "_content_table_for_id"),
)

build_table_chunker = cast(
    Callable[..., object | None],
    _internal(_INGESTION_PIPELINE, "_build_table_chunker"),
)

_SAMPLE_TABLE = "| A | B |\n|---|---|\n| 1 | 2 |"
_SAMPLE_TABLE_2 = "| X | Y |\n|---|---|\n| 3 | 4 |"


def _doc(
    *,
    content: str = "Body text.",
    tables: list[dict[str, Any]] | None = None,
    source: str = "/tmp/report.pdf",
) -> Document:
    metadata: dict[str, Any] = {"loader": "docling", "filename": "report.pdf"}
    if tables is not None:
        metadata["tables"] = tables
    return Document(source=source, content=content, metadata=metadata)


def _table_chunk(text: str = _SAMPLE_TABLE, table_id: str = "table-1") -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=text,
        metadata={
            CHUNK_TYPE_KEY: CHUNK_TYPE_TABLE,
            TABLE_ID_KEY: table_id,
            CHUNK_SOURCE_KEY: "/tmp/report.pdf",
        },
    )


def _report_pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def _embedded_table_chunk(document: Document) -> Chunk:
    table = build_table_chunks(document)[0]
    return table.model_copy(update={"embedding": [0.1] * 4, "sparse_vector": {1: 0.9}})


def _run_table_chunker_ingest(
    path: Path,
    document: Document,
    *,
    table_chunker: MagicMock,
    pipeline: IngestionPipeline,
) -> IngestionResult:
    return run_structured_chunker_ingest(
        path,
        document,
        chunker=table_chunker,
        chunker_attr="_table_chunker",
        pipeline=pipeline,
    )


def _run_skip_table_chunker_ingest(
    path: Path,
    document: Document,
    *,
    table_chunker: MagicMock,
    chunk_ids: list[str],
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock]:
    return run_skip_structured_chunker_ingest(
        path,
        document,
        chunker=table_chunker,
        chunker_attr="_table_chunker",
        chunk_ids=chunk_ids,
    )


def _doc_with_sample_table(source: str) -> Document:
    return _doc(
        source=source,
        tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
    )


def _run_skip_with_indexed_table_chunks(
    path: Path,
    document: Document,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock]:
    return run_skip_with_indexed_structured_chunks(
        path,
        document,
        indexed_chunk=_embedded_table_chunk(document),
        chunker_attr="_table_chunker",
    )


def _mock_table_chunker_with_empty_index() -> MagicMock:
    return mock_chunker_with_empty_index()


def _run_skip_purge_only_table_chunker_ingest(
    path: Path,
    document: Document,
    *,
    stale_table_id: str,
    caplog: pytest.LogCaptureFixture,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    return run_skip_purge_only_structured_chunker_ingest(
        path,
        document,
        chunker_attr="_table_chunker",
        stale_chunk_id=stale_table_id,
        logger_name=_INGESTION_PIPELINE,
        caplog=caplog,
    )


def _assert_skip_purged_only_table_chunks(
    result: IngestionResult,
    service: MagicMock,
    table_chunker: MagicMock,
    vector_store: MagicMock,
    bm25: MagicMock,
    metadata: MagicMock,
    *,
    stale_table_id: str,
    merged_ids: list[str],
    index_called: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert_skip_purged_only_structured_chunks(
        result,
        service,
        table_chunker,
        vector_store,
        bm25,
        metadata,
        stale_chunk_id=stale_table_id,
        merged_ids=merged_ids,
        index_called=index_called,
        kind="table",
        caplog=caplog,
    )


class TestTableChunkIdHelpers:
    def test_known_table_chunk_ids_maps_layout_table_ids(self) -> None:
        source = "/tmp/report.pdf"
        assert known_table_chunk_ids(source, ["table-1", "table-2"]) == {
            table_chunk_id(source, "table-1"),
            table_chunk_id(source, "table-2"),
        }

    def test_collect_table_ids_merges_metadata_and_indexed_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2}],
        )
        existing = [table_chunk_id(source, "table-1"), "text-chunk-1"]
        assert collect_table_ids(document, existing) == {"table-1", "table-2"}

    def test_existing_table_chunk_ids_uses_bm25_payloads(self) -> None:
        source = "/tmp/report.pdf"
        custom_id = table_chunk_id(source, "custom-layout-id")
        document = _doc(source=source, tables=[])
        bm25 = MagicMock()
        bm25.get_by_id.side_effect = lambda chunk_id: (
            _table_chunk(table_id="custom-layout-id").model_copy(update={"id": custom_id})
            if chunk_id == custom_id
            else None
        )
        assert existing_table_chunk_ids(
            source,
            [custom_id, "text-chunk-1"],
            document=document,
            bm25=bm25,
        ) == {custom_id}

    def test_existing_table_chunk_ids_without_bm25_uses_layout_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        assert existing_table_chunk_ids(
            source,
            ["text-chunk-1"],
            document=document,
            bm25=None,
        ) == {table_chunk_id(source, "table-1")}

    def test_existing_table_chunk_ids_without_get_by_id_uses_layout_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        assert existing_table_chunk_ids(
            source,
            ["text-chunk-1"],
            document=document,
            bm25=object(),
        ) == {table_chunk_id(source, "table-1")}

    def test_collect_table_ids_returns_empty_for_no_tables_or_indexed_ids(self) -> None:
        document = _doc(source="/tmp/report.pdf", tables=[])
        assert collect_table_ids(document, []) == set()


class TestMetadataTableIds:
    def test_returns_ids_from_valid_entries(self) -> None:
        document = _doc(
            tables=[
                {TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE},
                {TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2},
            ]
        )
        assert metadata_table_ids(document) == {"table-1", "table-2"}

    def test_skips_invalid_entries(self) -> None:
        base = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        metadata = dict(base.metadata)
        metadata["tables"] = ["bad", metadata["tables"][0], {}]
        document = base.model_copy(update={"metadata": metadata})
        assert metadata_table_ids(document) == {"table-1"}

    def test_returns_empty_without_tables(self) -> None:
        assert metadata_table_ids(_doc()) == set()
        assert metadata_table_ids(_doc(tables=[])) == set()
        document = _doc()
        document.metadata["tables"] = "bad"
        assert metadata_table_ids(document) == set()


class TestBuiltTableIds:
    def test_extracts_table_ids_from_chunks(self) -> None:
        document = _doc(
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        built = build_table_chunks(document)
        assert built_table_ids(built) == {"table-1"}

    def test_returns_empty_for_no_chunks(self) -> None:
        assert built_table_ids([]) == set()


class TestTableEmbeddingSafetyHelpers:
    def test_table_embedding_succeeded_when_all_desired_embedded(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        built = build_table_chunks(document)
        embedded = [_embedded_table_chunk(document)]
        assert table_embedding_succeeded(built, embedded) is True

    def test_table_embedding_succeeded_when_no_tables_expected(self) -> None:
        assert table_embedding_succeeded([], []) is True

    def test_table_embedding_succeeded_false_on_partial_or_failed_embed(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[
                {TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE},
                {TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2},
            ],
        )
        built = build_table_chunks(document)
        assert table_embedding_succeeded(built, []) is False
        assert table_embedding_succeeded(built, built[:1]) is False

    def test_table_build_succeeded_when_all_metadata_tables_built(self) -> None:
        document = _doc(
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        built = build_table_chunks(document)
        assert table_build_succeeded(document, built) is True

    def test_table_build_succeeded_when_no_metadata_tables(self) -> None:
        document = _doc(tables=[])
        assert table_build_succeeded(document, []) is True

    def test_table_build_succeeded_false_when_metadata_tables_unbuildable(self) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1"}])
        assert table_build_succeeded(document, []) is False

    def test_table_sync_succeeded_requires_build_and_embed(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        built = build_table_chunks(document)
        embedded = [_embedded_table_chunk(document)]
        assert table_sync_succeeded(document, built, embedded) is True
        assert table_sync_succeeded(document, built, []) is False
        unbuildable = _doc(source=source, tables=[{TABLE_ID_KEY: "table-1"}])
        assert table_sync_succeeded(unbuildable, [], []) is False

    def test_stale_table_ids_safe_to_purge_only_after_successful_sync(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2}],
        )
        built = build_table_chunks(document)
        stale = [table_chunk_id(source, "table-1")]
        assert stale_table_ids_safe_to_purge(document, built, [], stale) == []
        embedded = [_embedded_table_chunk(document)]
        assert stale_table_ids_safe_to_purge(document, built, embedded, stale) == stale
        removed = _doc(source=source, tables=[])
        assert stale_table_ids_safe_to_purge(removed, [], [], stale) == stale

    def test_stale_table_ids_safe_to_purge_blocks_unbuildable_metadata_tables(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, tables=[{TABLE_ID_KEY: "table-1"}])
        stale = [table_chunk_id(source, "table-1")]
        assert stale_table_ids_safe_to_purge(document, [], [], stale) == []

    def test_retained_table_chunk_ids_on_embed_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        built = build_table_chunks(document)
        stable_id = built[0].id
        assert retained_table_chunk_ids_on_embed_failure(
            source,
            document,
            ["text-chunk-1", stable_id],
            built,
            [],
        ) == {stable_id}
        embedded = [_embedded_table_chunk(document)]
        assert (
            retained_table_chunk_ids_on_embed_failure(
                source,
                document,
                ["text-chunk-1", stable_id],
                built,
                embedded,
            )
            == set()
        )

    def test_retained_table_chunk_ids_on_build_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, tables=[{TABLE_ID_KEY: "table-1"}])
        stable_id = table_chunk_id(source, "table-1")
        assert retained_table_chunk_ids_on_embed_failure(
            source,
            document,
            ["text-chunk-1", stable_id],
            [],
            [],
        ) == {stable_id}

    def test_merged_table_chunk_ids_after_successful_embed(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2}],
        )
        built = build_table_chunks(document)
        embedded = [_embedded_table_chunk(document)]
        known = {
            table_chunk_id(source, "table-1"),
            embedded[0].id,
        }
        assert merged_table_chunk_ids(
            ["text-chunk-1", table_chunk_id(source, "table-1")],
            known,
            document,
            built,
            embedded,
        ) == [embedded[0].id]

    def test_merged_table_chunk_ids_keeps_existing_on_embed_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2}],
        )
        built = build_table_chunks(document)
        stale_id = table_chunk_id(source, "table-1")
        new_id = built[0].id
        known = {stale_id, new_id}
        assert merged_table_chunk_ids(
            ["text-chunk-1", stale_id],
            known,
            document,
            built,
            [],
        ) == [stale_id]

    def test_merged_table_chunk_ids_keeps_existing_on_build_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, tables=[{TABLE_ID_KEY: "table-1"}])
        stable_id = table_chunk_id(source, "table-1")
        known = {stable_id}
        assert merged_table_chunk_ids(
            ["text-chunk-1", stable_id],
            known,
            document,
            [],
            [],
        ) == [stable_id]

    def test_merged_table_chunk_ids_after_table_removal(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, tables=[])
        stale_id = table_chunk_id(source, "table-1")
        known = {stale_id}
        assert merged_table_chunk_ids(["text-chunk-1", stale_id], known, document, [], []) == []

    def test_retained_table_chunk_ids_uses_bm25_payloads(self) -> None:
        source = "/tmp/report.pdf"
        custom_id = table_chunk_id(source, "custom-layout-id")
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "custom-layout-id", "text": _SAMPLE_TABLE}],
        )
        built = build_table_chunks(document)
        bm25 = MagicMock()
        bm25.get_by_id.side_effect = lambda chunk_id: (
            _table_chunk(table_id="custom-layout-id").model_copy(update={"id": custom_id})
            if chunk_id == custom_id
            else None
        )
        assert retained_table_chunk_ids_on_embed_failure(
            source,
            document,
            ["text-chunk-1", custom_id],
            built,
            [],
            bm25=bm25,
        ) == {custom_id}


class TestTableChunksNeedingUpsert:
    def test_returns_new_chunks(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        embedded = _embedded_table_chunk(document)
        assert table_chunks_needing_upsert([embedded], ["text-chunk-1"]) == [embedded]

    def test_returns_chunks_with_updated_text(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE_2}],
        )
        embedded = _embedded_table_chunk(document)
        bm25 = MagicMock()
        bm25.get_by_id.return_value = _table_chunk(text=_SAMPLE_TABLE).model_copy(
            update={"id": embedded.id}
        )
        assert table_chunks_needing_upsert(
            [embedded],
            ["text-chunk-1", embedded.id],
            bm25=bm25,
        ) == [embedded]

    def test_skips_unchanged_indexed_chunks(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        embedded = _embedded_table_chunk(document)
        bm25 = MagicMock()
        bm25.get_by_id.return_value = embedded.model_copy(update={"embedding": None})
        assert (
            table_chunks_needing_upsert(
                [embedded],
                ["text-chunk-1", embedded.id],
                bm25=bm25,
            )
            == []
        )

    def test_returns_existing_chunk_when_bm25_payload_missing(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        embedded = _embedded_table_chunk(document)
        bm25 = MagicMock()
        bm25.get_by_id.return_value = None
        assert table_chunks_needing_upsert(
            [embedded],
            ["text-chunk-1", embedded.id],
            bm25=bm25,
        ) == [embedded]

    def test_without_get_by_id_skips_existing_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        embedded = _embedded_table_chunk(document)
        assert (
            table_chunks_needing_upsert(
                [embedded],
                ["text-chunk-1", embedded.id],
                bm25=object(),
            )
            == []
        )

    def test_without_bm25_skips_existing_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        embedded = _embedded_table_chunk(document)
        assert (
            table_chunks_needing_upsert(
                [embedded],
                ["text-chunk-1", embedded.id],
                bm25=None,
            )
            == []
        )


class TestIsTableChunk:
    def test_true_for_table_type(self) -> None:
        assert is_table_chunk(_table_chunk()) is True

    def test_false_for_text_chunk(self) -> None:
        assert is_table_chunk(embedded_chunk()) is False


class TestExtractMarkdownTables:
    def test_extracts_single_table(self) -> None:
        content = f"Intro\n\n{_SAMPLE_TABLE}\n\nOutro"
        assert extract_markdown_tables(content) == [_SAMPLE_TABLE]

    def test_extracts_multiple_tables_in_order(self) -> None:
        content = f"{_SAMPLE_TABLE}\n\n{_SAMPLE_TABLE_2}"
        assert extract_markdown_tables(content) == [_SAMPLE_TABLE, _SAMPLE_TABLE_2]

    def test_returns_empty_when_no_tables(self) -> None:
        assert extract_markdown_tables("plain paragraph") == []

    def test_ignores_blank_matches(self) -> None:
        assert extract_markdown_tables("") == []


class TestContentTableForId:
    def test_maps_table_id_to_nth_content_table(self) -> None:
        tables = [_SAMPLE_TABLE, _SAMPLE_TABLE_2]
        assert content_table_for_id("table-1", tables) == _SAMPLE_TABLE
        assert content_table_for_id("table-2", tables) == _SAMPLE_TABLE_2

    def test_returns_none_for_unknown_or_non_numeric_ids(self) -> None:
        assert content_table_for_id("table-99", [_SAMPLE_TABLE]) is None
        assert content_table_for_id("custom-id", [_SAMPLE_TABLE]) is None


class TestResolveTableText:
    def test_prefers_text_key(self) -> None:
        entry = {"text": " from metadata "}
        assert resolve_table_text(entry, "| fallback |") == "from metadata"

    def test_uses_markdown_key_when_text_missing(self) -> None:
        entry = {"markdown": " md table "}
        assert resolve_table_text(entry, None) == "md table"

    def test_falls_back_to_content_table(self) -> None:
        assert resolve_table_text({}, _SAMPLE_TABLE) == _SAMPLE_TABLE

    def test_returns_none_when_all_missing(self) -> None:
        assert resolve_table_text({}, None) is None
        assert resolve_table_text({"text": "   "}, "  ") is None


class TestBuildTableChunks:
    def test_table_chunk_ids_are_deterministic(self) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        first = build_table_chunks(document)
        second = build_table_chunks(document)
        assert len(first) == 1
        assert first[0].id == second[0].id
        assert first[0].id == table_chunk_id(document.source, "table-1")

    def test_table_chunk_ids_stable_across_document_reload(self) -> None:
        source = "/tmp/report.pdf"
        tables = [{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}]
        first_load = _doc(source=source, tables=tables)
        reloaded = _doc(source=source, tables=tables)
        assert first_load.id != reloaded.id
        assert build_table_chunks(first_load)[0].id == build_table_chunks(reloaded)[0].id

    def test_returns_empty_without_tables_metadata(self) -> None:
        assert build_table_chunks(_doc()) == []

    def test_returns_empty_for_empty_tables_list(self) -> None:
        assert build_table_chunks(_doc(tables=[])) == []

    def test_builds_chunk_from_metadata_text(self) -> None:
        document = _doc(
            tables=[
                {
                    TABLE_ID_KEY: "table-1",
                    "text": _SAMPLE_TABLE,
                    CHUNK_PAGE_KEY: 2,
                    BBOX_KEY: [1.0, 2.0, 3.0, 4.0],
                }
            ]
        )
        chunks = build_table_chunks(document)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == _SAMPLE_TABLE
        assert chunk.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_TABLE
        assert chunk.metadata[TABLE_ID_KEY] == "table-1"
        assert chunk.metadata[CHUNK_PAGE_KEY] == 2
        assert chunk.metadata[BBOX_KEY] == [1.0, 2.0, 3.0, 4.0]
        assert chunk.metadata[CHUNK_SOURCE_KEY] == document.source
        assert chunk.metadata["loader"] == "docling"
        assert "tables" not in chunk.metadata

    def test_falls_back_to_markdown_in_document_content(self) -> None:
        document = _doc(
            content=f"Intro\n\n{_SAMPLE_TABLE}",
            tables=[{TABLE_ID_KEY: "table-1"}],
        )
        chunks = build_table_chunks(document)
        assert len(chunks) == 1
        assert chunks[0].text == _SAMPLE_TABLE

    def test_fallback_uses_table_id_not_metadata_index(self) -> None:
        base = _doc(
            content=f"{_SAMPLE_TABLE}\n\n{_SAMPLE_TABLE_2}",
            tables=[{TABLE_ID_KEY: "table-2"}],
        )
        metadata = dict(base.metadata)
        metadata["tables"] = ["bad", metadata["tables"][0]]
        document = base.model_copy(update={"metadata": metadata})
        chunks = build_table_chunks(document)
        assert len(chunks) == 1
        assert chunks[0].metadata[TABLE_ID_KEY] == "table-2"
        assert chunks[0].text == _SAMPLE_TABLE_2

    def test_skips_non_dict_entries(self, caplog: pytest.LogCaptureFixture) -> None:
        base = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        metadata = dict(base.metadata)
        metadata["tables"] = ["bad", metadata["tables"][0]]
        document = base.model_copy(update={"metadata": metadata})
        with caplog.at_level(logging.DEBUG, logger=_TABLE_CHUNKER):
            chunks = build_table_chunks(document)
        assert len(chunks) == 1
        assert "Skipping non-dict" in caplog.text

    def test_skips_entries_without_table_id(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(tables=[{"text": _SAMPLE_TABLE}])
        with caplog.at_level(logging.DEBUG, logger=_TABLE_CHUNKER):
            chunks = build_table_chunks(document)
        assert chunks == []
        assert TABLE_ID_KEY in caplog.text

    def test_skips_entries_without_text(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-9"}])
        with caplog.at_level(logging.WARNING, logger=_TABLE_CHUNKER):
            chunks = build_table_chunks(document)
        assert chunks == []
        assert "No text for table table-9" in caplog.text


class TestTableChunker:
    @staticmethod
    def _chunker(chunks: list[Chunk] | None = None) -> TableChunker:
        embedder = MagicMock()
        embedder.embed_both.return_value = (
            [[0.1] * 4 for _ in (chunks or [_table_chunk()])],
            [{1: 0.9} for _ in (chunks or [_table_chunk()])],
        )
        return TableChunker(embedder=embedder)  # type: ignore[arg-type]

    def test_index_returns_embedded_chunks(self) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        chunker = self._chunker()
        result = chunker.index(document)
        assert len(result) == 1
        assert result[0].embedding is not None
        assert result[0].sparse_vector is not None
        chunker._embedder.embed_both.assert_called_once()  # type: ignore[attr-defined]

    def test_index_returns_empty_when_no_tables(self) -> None:
        chunker = self._chunker()
        assert chunker.index(_doc()) == []
        chunker._embedder.embed_both.assert_not_called()  # type: ignore[attr-defined]

    def test_index_returns_empty_on_embedding_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _doc(tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}])
        embedder = MagicMock()
        embedder.embed_both.side_effect = RuntimeError("embed failed")
        chunker = TableChunker(embedder=embedder)  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger=_TABLE_CHUNKER):
            assert chunker.index(document) == []
        assert "Embedding table chunks failed" in caplog.text


class TestBuildTableChunker:
    def test_returns_none_when_disabled(self) -> None:
        assert build_table_chunker(MagicMock(), type("Cfg", (), {"enabled": False})()) is None

    def test_returns_chunker_when_enabled(self) -> None:
        embedder = MagicMock(spec=EmbeddingRepository)
        chunker = build_table_chunker(embedder, type("Cfg", (), {"enabled": True})())
        assert isinstance(chunker, TableChunker)


class TestIngestionPipelineTableChunks:
    def test_table_chunks_indexed_in_qdrant_and_bm25(self, tmp_path: Path) -> None:
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF-1.4")
        base = embedded_chunk(0)
        table = _table_chunk()
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline([base])
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=_doc(source=str(path.resolve())),
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 1
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 2
        assert any(is_table_chunk(c) for c in upserted)
        bm25_added = bm25.add.call_args.args[0]
        assert any(is_table_chunk(c) for c in bm25_added)

    def test_table_only_document_indexes_without_text_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "tables-only.pdf"
        path.write_bytes(b"%PDF-1.4")
        table = _table_chunk()
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(prepared_chunks=[])
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=_doc(source=str(path.resolve())),
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 0
        service.prepare.assert_called_once()
        table_chunker.index.assert_called_once()
        vector_store.upsert.assert_called_once()
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert is_table_chunk(upserted[0])
        bm25.add.assert_called_once()

    def test_table_only_reingest_purges_old_chunks_and_indexes_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "tables-only.pdf"
        path.write_bytes(b"%PDF-1.4")
        table = _table_chunk()
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        metadata = mock_reingest_metadata()
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[],
            metadata=metadata,
        )
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=_doc(source=str(path.resolve())),
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 0
        vector_store.upsert.assert_called_once()
        vector_store.delete.assert_called_once_with(["old-chunk-1"])
        bm25.remove_by_ids.assert_called_once_with(["old-chunk-1"])
        metadata.upsert_document.assert_called_once()

    def test_reingest_preserves_stable_table_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_table(str(path.resolve()))
        table = _embedded_table_chunk(document)
        stable_id = table.id
        metadata = mock_reingest_metadata(chunk_ids=["old-text-chunk-1", stable_id])
        base = embedded_chunk(0).model_copy(update={"id": "new-text-chunk-1"})
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[base],
            metadata=metadata,
        )
        result = _run_table_chunker_ingest(
            path,
            document,
            table_chunker=table_chunker,
            pipeline=pipeline,
        )

        assert result.skipped is False
        vector_store.upsert.assert_called_once()
        assert_purges_only_stale_text_chunk(vector_store, bm25)
        upserted_ids = [chunk.id for chunk in vector_store.upsert.call_args.args[0]]
        assert stable_id in upserted_ids

    def test_reingest_purges_removed_table_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), tables=[])
        removed_table_id = table_chunk_id(document.source, "table-1")
        metadata = mock_reingest_metadata(chunk_ids=["text-chunk-1", removed_table_id])
        table_chunker = MagicMock()
        table_chunker.index.return_value = []
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[embedded_chunk(0)],
            metadata=metadata,
        )
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=document,
        ):
            pipeline.ingest_file(path)

        vector_store.delete.assert_called_once_with(["text-chunk-1", removed_table_id])
        bm25.remove_by_ids.assert_called_once_with(["text-chunk-1", removed_table_id])

    def test_reingest_preserves_stable_table_chunks_in_real_bm25(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_table(str(path.resolve()))
        reingest_preserves_stable_structured_chunks_in_real_bm25(
            path,
            document,
            structured_chunk=_embedded_table_chunk(document),
            chunker_attr="_table_chunker",
            stale_structured_text="stale table content",
        )

    def test_full_reindex_on_skip_preserves_stable_table_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_table(str(path.resolve()))
        full_reindex_on_skip_preserves_stable_structured_chunk_ids(
            path,
            document,
            indexed_chunk=_embedded_table_chunk(document),
            chunker_attr="_table_chunker",
        )

    def test_table_chunker_none_skips_indexing(self, tmp_path: Path) -> None:
        ingest_without_structured_chunker_indexes_text_only(tmp_path)

    def test_from_settings_wires_table_chunker_when_enabled(self) -> None:
        pipeline = ingestion_pipeline_from_settings(
            parsing=MagicMock(table_chunks=MagicMock(enabled=True)),
        )
        assert isinstance(pipeline._table_chunker, TableChunker)  # noqa: SLF001

    def test_skip_backfills_missing_table_chunks_on_unchanged_hash(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(
            source=str(path.resolve()),
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE}],
        )
        skip_backfills_missing_structured_chunks_on_unchanged_hash(
            path,
            document,
            indexed_chunk=_embedded_table_chunk(document),
            chunker_attr="_table_chunker",
            is_structured_chunk=is_table_chunk,
        )

    def test_skip_unchanged_without_table_chunker(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        skip_unchanged_without_structured_chunker(
            path,
            _doc(source=str(path.resolve())),
            chunker_attr="_table_chunker",
        )

    def test_skip_skips_when_table_chunks_already_indexed(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        result, service, vector_store, bm25 = _run_skip_with_indexed_table_chunks(
            path,
            _doc_with_sample_table(str(path.resolve())),
        )
        assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_skips_when_table_chunks_already_indexed_after_reload(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        first_load = _doc_with_sample_table(source)
        reloaded = _doc_with_sample_table(source)
        assert first_load.id != reloaded.id
        result, service, vector_store, bm25 = _run_skip_with_indexed_table_chunks(
            path,
            reloaded,
        )
        assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_backfill_noop_when_table_chunker_returns_empty(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()))
        result, service, vector_store, bm25, _ = _run_skip_table_chunker_ingest(
            path,
            document,
            table_chunker=_mock_table_chunker_with_empty_index(),
            chunk_ids=["text-chunk-1"],
        )

        assert result.skipped is True
        assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_purges_removed_table_chunks_on_unchanged_hash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), tables=[])
        removed_table_id = table_chunk_id(document.source, "table-1")
        result, service, vector_store, bm25, metadata, table_chunker = (
            _run_skip_purge_only_table_chunker_ingest(
                path,
                document,
                stale_table_id=removed_table_id,
                caplog=caplog,
            )
        )
        _assert_skip_purged_only_table_chunks(
            result,
            service,
            table_chunker,
            vector_store,
            bm25,
            metadata,
            stale_table_id=removed_table_id,
            merged_ids=["text-chunk-1"],
            index_called=False,
            caplog=caplog,
        )

    def test_skip_purges_stale_table_chunks_when_layout_changes(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2}],
        )
        skip_purges_stale_structured_chunks_when_layout_changes(
            path,
            document,
            indexed_chunk=_embedded_table_chunk(document),
            stale_chunk_id=table_chunk_id(source, "table-1"),
            chunker_attr="_table_chunker",
        )

    def test_skip_skips_when_table_embed_fails_on_layout_change(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-2", "text": _SAMPLE_TABLE_2}],
        )
        stale_table_id = table_chunk_id(source, "table-1")
        result, service, vector_store, bm25, metadata, table_chunker = (
            _run_skip_purge_only_table_chunker_ingest(
                path,
                document,
                stale_table_id=stale_table_id,
                caplog=caplog,
            )
        )

        assert result.skipped is True
        service.prepare.assert_not_called()
        table_chunker.index.assert_called_once()
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        assert "Purged" not in caplog.text

    def test_skip_retains_table_chunks_when_metadata_tables_have_no_text(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(source=source, tables=[{TABLE_ID_KEY: "table-1"}])
        stable_id = table_chunk_id(source, "table-1")
        table_chunker = _mock_table_chunker_with_empty_index()
        metadata = unchanged_hash_metadata(
            path,
            document,
            chunk_ids=["text-chunk-1", stable_id],
        )
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        with caplog.at_level(logging.INFO, logger=_INGESTION_PIPELINE):
            result = _run_table_chunker_ingest(
                path,
                document,
                table_chunker=table_chunker,
                pipeline=pipeline,
            )

        assert result.skipped is True
        service.prepare.assert_not_called()
        table_chunker.index.assert_not_called()
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        assert "Purged" not in caplog.text

    def test_reingest_retains_table_chunks_when_metadata_tables_have_no_text(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), tables=[{TABLE_ID_KEY: "table-1"}])
        stable_id = table_chunk_id(document.source, "table-1")
        _, metadata, vector_store, bm25 = run_reingest_with_empty_structured_chunker(
            path,
            document,
            chunker_attr="_table_chunker",
            old_chunk_ids=["old-text-chunk-1", stable_id],
        )

        assert_purges_only_stale_text_chunk(vector_store, bm25)
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert stable_id in merged_ids

    def test_reingest_retains_stable_table_chunks_when_embedding_fails(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_table(str(path.resolve()))
        stable_id = table_chunk_id(document.source, "table-1")
        _, metadata, vector_store, bm25 = run_reingest_with_empty_structured_chunker(
            path,
            document,
            chunker_attr="_table_chunker",
            old_chunk_ids=["old-text-chunk-1", stable_id],
        )

        assert_purges_only_stale_text_chunk(vector_store, bm25)
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert stable_id in merged_ids

    def test_table_only_reingest_retains_old_table_chunks_when_embedding_fails(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_table(str(path.resolve()))
        stable_id = table_chunk_id(document.source, "table-1")
        metadata = mock_reingest_metadata(chunk_ids=[stable_id])
        table_chunker = MagicMock()
        table_chunker.index.return_value = []
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[],
            metadata=metadata,
        )
        pipeline._table_chunker = table_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=document,
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 0
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert merged_ids == [stable_id]

    def test_skip_refreshes_changed_table_text_on_unchanged_hash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(
            source=source,
            tables=[{TABLE_ID_KEY: "table-1", "text": _SAMPLE_TABLE_2}],
        )
        table = _embedded_table_chunk(document)
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        metadata = unchanged_hash_metadata(
            path,
            document,
            chunk_ids=["text-chunk-1", table.id],
        )
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        bm25.get_by_id.side_effect = lambda chunk_id: (
            _table_chunk(text=_SAMPLE_TABLE).model_copy(update={"id": table.id})
            if chunk_id == table.id
            else None
        )
        with caplog.at_level(logging.INFO, logger=_INGESTION_PIPELINE):
            result = _run_table_chunker_ingest(
                path,
                document,
                table_chunker=table_chunker,
                pipeline=pipeline,
            )

        assert result.skipped is False
        service.prepare.assert_not_called()
        table_chunker.index.assert_called_once()
        vector_store.upsert.assert_called_once()
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert upserted[0].text == _SAMPLE_TABLE_2
        bm25.add.assert_called_once()
        vector_store.delete.assert_not_called()
        metadata.upsert_document.assert_called_once()
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert merged_ids == ["text-chunk-1", table.id]
        assert "Backfilled 1 table chunk(s)" in caplog.text

    def test_skip_purges_custom_table_chunk_ids_using_bm25_payloads(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(source=source, tables=[])
        custom_table_id = table_chunk_id(source, "custom-layout-id")
        table_chunker = MagicMock()
        table_chunker.index.return_value = []
        metadata = unchanged_hash_metadata(
            path,
            document,
            chunk_ids=["text-chunk-1", custom_table_id],
        )
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        bm25.get_by_id.side_effect = lambda chunk_id: (
            _table_chunk(table_id="custom-layout-id").model_copy(update={"id": custom_table_id})
            if chunk_id == custom_table_id
            else None
        )
        result = _run_table_chunker_ingest(
            path,
            document,
            table_chunker=table_chunker,
            pipeline=pipeline,
        )

        assert result.skipped is False
        service.prepare.assert_not_called()
        vector_store.delete.assert_called_once_with([custom_table_id])
        bm25.remove_by_ids.assert_called_once_with([custom_table_id])
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_table(str(path.resolve()))
        table_chunker = MagicMock()
        table_chunker.index.return_value = []
        result, service, vector_store, bm25, metadata = _run_skip_table_chunker_ingest(
            path,
            document,
            table_chunker=table_chunker,
            chunk_ids=["text-chunk-1"],
        )

        assert result.skipped is True
        service.prepare.assert_not_called()
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.add.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        metadata.upsert_document.assert_called_once()
        _, kwargs = metadata.upsert_document.call_args
        assert kwargs["skipped"] is True
