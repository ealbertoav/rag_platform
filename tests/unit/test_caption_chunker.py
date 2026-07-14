"""T-232 — Structured caption chunks at ingesting."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import (
    ASSET_PATH_KEY,
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_KEY,
    FIGURE_CAPTION_KEY,
    FIGURE_ID_KEY,
    MODALITY_CAPTION,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.ingestion.caption_chunker import (
    CaptionChunker,
    build_caption_chunks,
    built_caption_figure_ids,
    caption_build_succeeded,
    caption_chunk_id,
    caption_chunks_needing_upsert,
    caption_embedding_succeeded,
    caption_sync_succeeded,
    collect_caption_figure_ids,
    existing_caption_chunk_ids,
    is_caption_chunk,
    known_caption_chunk_ids,
    merged_caption_chunk_ids,
    metadata_caption_figure_ids,
    retained_caption_chunk_ids_on_embed_failure,
    stale_caption_ids_safe_to_purge,
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

_CAPTION_CHUNKER = "src.rag.ingestion.caption_chunker"
_INGESTION_PIPELINE = "src.rag.pipelines.ingestion_pipeline"

_SAMPLE_CAPTION = "A bar chart showing quarterly revenue."
_SAMPLE_CAPTION_2 = "Architecture diagram of the RAG pipeline."


def _internal(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


resolve_caption_text = cast(
    Callable[[dict[str, Any]], str | None],
    _internal(_CAPTION_CHUNKER, "_resolve_caption_text"),
)

build_caption_chunker = cast(
    Callable[..., object | None],
    _internal(_INGESTION_PIPELINE, "_build_caption_chunker"),
)


def _doc(
    *,
    content: str = "Body text.",
    figures: list[Any] | None = None,
    source: str = "/tmp/report.pdf",
) -> Document:
    metadata: dict[str, Any] = {"loader": "docling", "filename": "report.pdf"}
    if figures is not None:
        metadata["figures"] = figures
    return Document(source=source, content=content, metadata=metadata)


def _caption_chunk(text: str = _SAMPLE_CAPTION, figure_id: str = "figure-1") -> Chunk:
    return Chunk(
        document_id="doc-1",
        text=text,
        metadata={
            CHUNK_TYPE_KEY: CHUNK_TYPE_CAPTION,
            FIGURE_ID_KEY: figure_id,
            CHUNK_SOURCE_KEY: "/tmp/report.pdf",
        },
        modality=MODALITY_CAPTION,
    )


def _report_pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def _embedded_caption_chunk(document: Document) -> Chunk:
    caption = build_caption_chunks(document)[0]
    return caption.model_copy(update={"embedding": [0.1] * 4, "sparse_vector": {1: 0.9}})


def _run_caption_chunker_ingest(
    path: Path,
    document: Document,
    *,
    caption_chunker: MagicMock,
    pipeline: IngestionPipeline,
) -> IngestionResult:
    return run_structured_chunker_ingest(
        path,
        document,
        chunker=caption_chunker,
        chunker_attr="_caption_chunker",
        pipeline=pipeline,
    )


def _run_skip_caption_chunker_ingest(
    path: Path,
    document: Document,
    *,
    caption_chunker: MagicMock,
    chunk_ids: list[str],
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock]:
    return run_skip_structured_chunker_ingest(
        path,
        document,
        chunker=caption_chunker,
        chunker_attr="_caption_chunker",
        chunk_ids=chunk_ids,
    )


def _doc_with_sample_caption(source: str) -> Document:
    return _doc(
        source=source,
        figures=[
            {
                FIGURE_ID_KEY: "figure-1",
                FIGURE_CAPTION_KEY: _SAMPLE_CAPTION,
                ASSET_PATH_KEY: "/tmp/assets/figure-1.png",
            }
        ],
    )


def _run_skip_with_indexed_caption_chunks(
    path: Path,
    document: Document,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock]:
    return run_skip_with_indexed_structured_chunks(
        path,
        document,
        indexed_chunk=_embedded_caption_chunk(document),
        chunker_attr="_caption_chunker",
    )


def _mock_caption_chunker_with_empty_index() -> MagicMock:
    return mock_chunker_with_empty_index()


def _run_skip_purge_only_caption_chunker_ingest(
    path: Path,
    document: Document,
    *,
    stale_caption_id: str,
    caplog: pytest.LogCaptureFixture,
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    return run_skip_purge_only_structured_chunker_ingest(
        path,
        document,
        chunker_attr="_caption_chunker",
        stale_chunk_id=stale_caption_id,
        logger_name=_INGESTION_PIPELINE,
        caplog=caplog,
    )


def _assert_skip_purged_only_caption_chunks(
    result: IngestionResult,
    service: MagicMock,
    caption_chunker: MagicMock,
    vector_store: MagicMock,
    bm25: MagicMock,
    metadata: MagicMock,
    *,
    stale_caption_id: str,
    merged_ids: list[str],
    index_called: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert_skip_purged_only_structured_chunks(
        result,
        service,
        caption_chunker,
        vector_store,
        bm25,
        metadata,
        stale_chunk_id=stale_caption_id,
        merged_ids=merged_ids,
        index_called=index_called,
        kind="caption",
        caplog=caplog,
    )


class TestCaptionChunkIdHelpers:
    def test_known_caption_chunk_ids_maps_layout_figure_ids(self) -> None:
        source = "/tmp/report.pdf"
        assert known_caption_chunk_ids(source, ["figure-1", "figure-2"]) == {
            caption_chunk_id(source, "figure-1"),
            caption_chunk_id(source, "figure-2"),
        }

    def test_collect_caption_figure_ids_merges_metadata_and_indexed_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-2", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION_2}],
        )
        existing = [caption_chunk_id(source, "figure-1"), "text-chunk-1"]
        assert collect_caption_figure_ids(document, existing) == {"figure-1", "figure-2"}

    def test_existing_caption_chunk_ids_uses_bm25_payloads(self) -> None:
        source = "/tmp/report.pdf"
        custom_id = caption_chunk_id(source, "custom-layout-id")
        document = _doc(source=source, figures=[])
        bm25 = MagicMock()
        bm25.get_by_id.side_effect = lambda chunk_id: (
            _caption_chunk(figure_id="custom-layout-id").model_copy(update={"id": custom_id})
            if chunk_id == custom_id
            else None
        )
        assert existing_caption_chunk_ids(
            source,
            [custom_id, "text-chunk-1"],
            document=document,
            bm25=bm25,
        ) == {custom_id}

    def test_existing_caption_chunk_ids_without_bm25_uses_layout_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        assert existing_caption_chunk_ids(
            source,
            ["text-chunk-1"],
            document=document,
            bm25=None,
        ) == {caption_chunk_id(source, "figure-1")}

    def test_existing_caption_chunk_ids_without_get_by_id_uses_layout_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        bm25 = object()
        assert existing_caption_chunk_ids(
            source,
            ["text-chunk-1"],
            document=document,
            bm25=bm25,
        ) == {caption_chunk_id(source, "figure-1")}

    def test_collect_caption_figure_ids_returns_empty_for_no_figures_or_indexed_ids(
        self,
    ) -> None:
        assert collect_caption_figure_ids(_doc(figures=[]), []) == set()


class TestMetadataCaptionFigureIds:
    def test_returns_ids_from_captioned_entries(self) -> None:
        document = _doc(
            figures=[
                {FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION},
                {FIGURE_ID_KEY: "figure-2", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION_2},
            ]
        )
        assert metadata_caption_figure_ids(document) == {"figure-1", "figure-2"}

    def test_skips_invalid_entries_and_uncaptioned(self) -> None:
        document = _doc(
            figures=[
                "not-a-dict",
                {FIGURE_ID_KEY: "figure-1"},
                {FIGURE_CAPTION_KEY: _SAMPLE_CAPTION},
                {FIGURE_ID_KEY: "figure-2", FIGURE_CAPTION_KEY: "  "},
                {FIGURE_ID_KEY: "figure-3", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION_2},
            ]
        )
        assert metadata_caption_figure_ids(document) == {"figure-3"}

    def test_returns_empty_without_figures(self) -> None:
        assert metadata_caption_figure_ids(_doc()) == set()
        assert metadata_caption_figure_ids(_doc(figures=[])) == set()
        assert (
            metadata_caption_figure_ids(
                Document(source="/tmp/x.pdf", content="", metadata={"figures": "bad"})
            )
            == set()
        )


class TestBuiltCaptionFigureIds:
    def test_extracts_figure_ids_from_chunks(self) -> None:
        chunks = [
            _caption_chunk(figure_id="figure-1"),
            _caption_chunk(figure_id="figure-2"),
        ]
        assert built_caption_figure_ids(chunks) == {"figure-1", "figure-2"}

    def test_returns_empty_for_no_chunks(self) -> None:
        assert built_caption_figure_ids([]) == set()


class TestCaptionEmbeddingSafetyHelpers:
    def test_caption_embedding_succeeded_when_all_desired_embedded(self) -> None:
        built = [_caption_chunk().model_copy(update={"id": "c1"})]
        embedded = [built[0].model_copy(update={"embedding": [0.1]})]
        assert caption_embedding_succeeded(built, embedded) is True

    def test_caption_embedding_succeeded_when_no_captions_expected(self) -> None:
        assert caption_embedding_succeeded([], []) is True

    def test_caption_embedding_succeeded_false_on_partial_or_failed_embed(self) -> None:
        built = [
            _caption_chunk().model_copy(update={"id": "c1"}),
            _caption_chunk(figure_id="figure-2").model_copy(update={"id": "c2"}),
        ]
        assert caption_embedding_succeeded(built, [built[0]]) is False
        assert caption_embedding_succeeded(built, []) is False

    def test_caption_build_succeeded_when_all_captioned_figures_built(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        assert caption_build_succeeded(document, build_caption_chunks(document)) is True

    def test_caption_build_succeeded_when_no_captioned_figures(self) -> None:
        assert caption_build_succeeded(_doc(figures=[]), []) is True

    def test_caption_build_succeeded_false_when_captioned_figures_unbuildable(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        assert caption_build_succeeded(document, []) is False

    def test_caption_sync_succeeded_requires_build_and_embed(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        built = build_caption_chunks(document)
        embedded = [built[0].model_copy(update={"embedding": [0.1]})]
        assert caption_sync_succeeded(document, built, embedded) is True
        assert caption_sync_succeeded(document, built, []) is False

    def test_stale_caption_ids_safe_to_purge_only_after_successful_sync(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        built = build_caption_chunks(document)
        embedded = [built[0].model_copy(update={"embedding": [0.1]})]
        stale = ["old-caption"]
        assert stale_caption_ids_safe_to_purge(document, built, embedded, stale) == stale
        assert stale_caption_ids_safe_to_purge(document, built, [], stale) == []

    def test_stale_caption_ids_safe_to_purge_blocks_unbuildable(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        assert stale_caption_ids_safe_to_purge(document, [], [], ["old"]) == []

    def test_retained_caption_chunk_ids_on_embed_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[
                {FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION},
                {FIGURE_ID_KEY: "figure-2", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION_2},
            ],
        )
        built = build_caption_chunks(document)
        old_ids = ["text-1", built[0].id, built[1].id]
        retained = retained_caption_chunk_ids_on_embed_failure(
            source,
            document,
            old_ids,
            built,
            [built[0]],
        )
        assert retained == {built[0].id, built[1].id}

    def test_retained_caption_chunk_ids_on_build_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        stable_id = caption_chunk_id(source, "figure-1")
        retained = retained_caption_chunk_ids_on_embed_failure(
            source,
            document,
            ["text-1", stable_id],
            [],
            [],
        )
        assert retained == {stable_id}

    def test_retained_returns_empty_on_successful_sync(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        built = build_caption_chunks(document)
        embedded = [built[0].model_copy(update={"embedding": [0.1]})]
        assert (
            retained_caption_chunk_ids_on_embed_failure(
                source,
                document,
                ["text-1", built[0].id],
                built,
                embedded,
            )
            == set()
        )

    def test_merged_caption_chunk_ids_after_successful_embed(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        built = build_caption_chunks(document)
        embedded = [built[0].model_copy(update={"embedding": [0.1]})]
        known = {built[0].id}
        assert merged_caption_chunk_ids(
            ["text-1", built[0].id],
            known,
            document,
            built,
            embedded,
        ) == [built[0].id]

    def test_merged_caption_chunk_ids_keeps_existing_on_embed_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        built = build_caption_chunks(document)
        known = {built[0].id}
        assert merged_caption_chunk_ids(
            ["text-1", built[0].id],
            known,
            document,
            built,
            [],
        ) == [built[0].id]

    def test_merged_caption_chunk_ids_keeps_existing_on_build_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        stable_id = caption_chunk_id(source, "figure-1")
        known = {stable_id}
        assert merged_caption_chunk_ids(
            ["text-1", stable_id],
            known,
            document,
            [],
            [],
        ) == [stable_id]

    def test_merged_caption_chunk_ids_after_caption_removal(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[])
        stale = caption_chunk_id(source, "figure-1")
        assert merged_caption_chunk_ids(["text-1", stale], {stale}, document, [], []) == []

    def test_retained_caption_chunk_ids_uses_bm25_payloads(self) -> None:
        source = "/tmp/report.pdf"
        custom_id = caption_chunk_id(source, "custom-id")
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "custom-id", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}],
        )
        built = [
            _caption_chunk(figure_id="custom-id").model_copy(update={"id": custom_id}),
        ]
        bm25 = MagicMock()
        bm25.get_by_id.side_effect = lambda chunk_id: built[0] if chunk_id == custom_id else None
        retained = retained_caption_chunk_ids_on_embed_failure(
            source,
            document,
            ["text-1", custom_id],
            built,
            [],
            bm25=bm25,
        )
        assert retained == {custom_id}


class TestCaptionChunksNeedingUpsert:
    def test_returns_new_chunks(self) -> None:
        chunk = _caption_chunk().model_copy(update={"id": "new-id"})
        assert caption_chunks_needing_upsert([chunk], ["text-1"]) == [chunk]

    def test_returns_chunks_with_updated_text(self) -> None:
        chunk = _caption_chunk().model_copy(update={"id": "c1", "text": "updated"})
        bm25 = MagicMock()
        bm25.get_by_id.return_value = _caption_chunk().model_copy(
            update={"id": "c1", "text": "old"}
        )
        assert caption_chunks_needing_upsert([chunk], ["c1"], bm25=bm25) == [chunk]

    def test_skips_unchanged_indexed_chunks(self) -> None:
        chunk = _caption_chunk().model_copy(update={"id": "c1"})
        bm25 = MagicMock()
        bm25.get_by_id.return_value = chunk.model_copy()
        assert caption_chunks_needing_upsert([chunk], ["c1"], bm25=bm25) == []

    def test_returns_existing_chunk_when_bm25_payload_missing(self) -> None:
        chunk = _caption_chunk().model_copy(update={"id": "c1"})
        bm25 = MagicMock()
        bm25.get_by_id.return_value = None
        assert caption_chunks_needing_upsert([chunk], ["c1"], bm25=bm25) == [chunk]

    def test_without_get_by_id_skips_existing_ids(self) -> None:
        chunk = _caption_chunk().model_copy(update={"id": "c1"})
        assert caption_chunks_needing_upsert([chunk], ["c1"], bm25=object()) == []

    def test_without_bm25_skips_existing_ids(self) -> None:
        chunk = _caption_chunk().model_copy(update={"id": "c1"})
        assert caption_chunks_needing_upsert([chunk], ["c1"], bm25=None) == []


class TestIsCaptionChunk:
    def test_true_for_caption_type(self) -> None:
        assert is_caption_chunk(_caption_chunk()) is True

    def test_false_for_text_chunk(self) -> None:
        assert is_caption_chunk(embedded_chunk()) is False


class TestResolveCaptionText:
    def test_prefers_caption_key(self) -> None:
        assert resolve_caption_text({FIGURE_CAPTION_KEY: "  hello  "}) == "hello"

    def test_returns_none_when_missing_or_blank(self) -> None:
        assert resolve_caption_text({}) is None
        assert resolve_caption_text({FIGURE_CAPTION_KEY: "   "}) is None
        assert resolve_caption_text({FIGURE_CAPTION_KEY: 123}) is None


class TestBuildCaptionChunks:
    def test_caption_chunk_ids_are_deterministic(self) -> None:
        document = _doc_with_sample_caption("/tmp/report.pdf")
        a = build_caption_chunks(document)
        b = build_caption_chunks(document)
        assert a[0].id == b[0].id == caption_chunk_id(document.source, "figure-1")

    def test_caption_chunk_ids_stable_across_document_reload(self) -> None:
        source = "/tmp/report.pdf"
        first = build_caption_chunks(_doc_with_sample_caption(source))
        second = build_caption_chunks(_doc_with_sample_caption(source))
        assert first[0].id == second[0].id
        assert first[0].document_id != second[0].document_id

    def test_returns_empty_without_figures_metadata(self) -> None:
        assert build_caption_chunks(_doc()) == []

    def test_returns_empty_for_empty_figures_list(self) -> None:
        assert build_caption_chunks(_doc(figures=[])) == []

    def test_builds_chunk_from_caption_text(self) -> None:
        document = _doc(
            figures=[
                {
                    FIGURE_ID_KEY: "figure-1",
                    FIGURE_CAPTION_KEY: _SAMPLE_CAPTION,
                    CHUNK_PAGE_KEY: 2,
                    BBOX_KEY: [0.1, 0.2, 0.3, 0.4],
                    ASSET_PATH_KEY: "/tmp/assets/figure-1.png",
                }
            ]
        )
        chunks = build_caption_chunks(document)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.text == _SAMPLE_CAPTION
        assert chunk.modality == MODALITY_CAPTION
        assert chunk.asset_path == "/tmp/assets/figure-1.png"
        assert chunk.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_CAPTION
        assert chunk.metadata[FIGURE_ID_KEY] == "figure-1"
        assert chunk.metadata[CHUNK_SOURCE_KEY] == document.source
        assert chunk.metadata[CHUNK_PAGE_KEY] == 2
        assert chunk.metadata[BBOX_KEY] == [0.1, 0.2, 0.3, 0.4]
        assert chunk.metadata[ASSET_PATH_KEY] == "/tmp/assets/figure-1.png"
        assert "figures" not in chunk.metadata

    def test_builds_without_asset_path(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        chunk = build_caption_chunks(document)[0]
        assert chunk.asset_path is None
        assert ASSET_PATH_KEY not in chunk.metadata

    def test_uses_asset_path_alias_key(self) -> None:
        document = _doc(
            figures=[
                {
                    FIGURE_ID_KEY: "figure-1",
                    FIGURE_CAPTION_KEY: _SAMPLE_CAPTION,
                    "asset_path": "/tmp/a.png",
                }
            ]
        )
        chunk = build_caption_chunks(document)[0]
        assert chunk.asset_path == "/tmp/a.png"

    def test_skips_non_dict_entries(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(
            figures=[
                "bad",
                {FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION},
            ]
        )
        with caplog.at_level(logging.DEBUG, logger=_CAPTION_CHUNKER):
            chunks = build_caption_chunks(document)
        assert len(chunks) == 1
        assert "non-dict" in caplog.text

    def test_skips_entries_without_figure_id(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(figures=[{FIGURE_CAPTION_KEY: _SAMPLE_CAPTION}])
        with caplog.at_level(logging.DEBUG, logger=_CAPTION_CHUNKER):
            assert build_caption_chunks(document) == []
        assert FIGURE_ID_KEY in caplog.text

    def test_skips_entries_without_caption(self, caplog: pytest.LogCaptureFixture) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1"}])
        with caplog.at_level(logging.DEBUG, logger=_CAPTION_CHUNKER):
            assert build_caption_chunks(document) == []
        assert "No caption" in caplog.text


class TestCaptionChunker:
    def test_index_returns_embedded_chunks(self) -> None:
        document = _doc_with_sample_caption("/tmp/report.pdf")
        embedder = MagicMock(spec=EmbeddingRepository)
        embedder.embed_both.return_value = ([[0.1, 0.2]], [{1: 0.9}])
        chunks = CaptionChunker(embedder).index(document)
        assert len(chunks) == 1
        assert chunks[0].embedding == [0.1, 0.2]
        assert chunks[0].sparse_vector == {1: 0.9}
        embedder.embed_both.assert_called_once_with([_SAMPLE_CAPTION])

    def test_index_returns_empty_when_no_captions(self) -> None:
        embedder = MagicMock(spec=EmbeddingRepository)
        assert CaptionChunker(embedder).index(_doc(figures=[])) == []
        embedder.embed_both.assert_not_called()

    def test_index_returns_empty_on_embedding_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _doc_with_sample_caption("/tmp/report.pdf")
        embedder = MagicMock(spec=EmbeddingRepository)
        embedder.embed_both.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.WARNING, logger=_CAPTION_CHUNKER):
            assert CaptionChunker(embedder).index(document) == []
        assert "Embedding caption chunks failed" in caplog.text


class TestBuildCaptionChunkerFactory:
    def test_returns_none_when_disabled(self) -> None:
        assert build_caption_chunker(MagicMock(), MagicMock(enabled=False)) is None

    def test_returns_chunker_when_enabled(self) -> None:
        chunker = build_caption_chunker(MagicMock(), MagicMock(enabled=True))
        assert isinstance(chunker, CaptionChunker)


class TestIngestionPipelineCaptionChunks:
    def test_caption_chunks_indexed_in_qdrant_and_bm25(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        caption = _embedded_caption_chunk(document)
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = [caption]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline()
        result = _run_caption_chunker_ingest(
            path,
            document,
            caption_chunker=caption_chunker,
            pipeline=pipeline,
        )

        assert result.chunk_count == 1
        upserted = vector_store.upsert.call_args.args[0]
        assert any(is_caption_chunk(c) for c in upserted)
        bm25.add.assert_called_once()
        assert any(is_caption_chunk(c) for c in bm25.add.call_args.args[0])

    def test_caption_only_document_indexes_without_text_chunks(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        caption = _embedded_caption_chunk(document)
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = [caption]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(prepared_chunks=[])
        result = _run_caption_chunker_ingest(
            path,
            document,
            caption_chunker=caption_chunker,
            pipeline=pipeline,
        )

        assert result.chunk_count == 0
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert is_caption_chunk(upserted[0])
        bm25.add.assert_called_once()

    def test_caption_only_reingest_purges_old_chunks_and_indexes_captions(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        caption = _embedded_caption_chunk(document)
        metadata = mock_reingest_metadata(chunk_ids=["old-text-chunk-1"])
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = [caption]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[],
            metadata=metadata,
        )
        _run_caption_chunker_ingest(
            path,
            document,
            caption_chunker=caption_chunker,
            pipeline=pipeline,
        )

        assert_purges_only_stale_text_chunk(vector_store, bm25)
        upserted = vector_store.upsert.call_args.args[0]
        assert [c.id for c in upserted] == [caption.id]

    def test_reingest_preserves_stable_caption_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        caption = _embedded_caption_chunk(document)
        stable_id = caption.id
        metadata = mock_reingest_metadata(chunk_ids=["old-text-chunk-1", stable_id])
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = [caption]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[embedded_chunk(0).model_copy(update={"id": "new-text-chunk-1"})],
            metadata=metadata,
        )
        _run_caption_chunker_ingest(
            path,
            document,
            caption_chunker=caption_chunker,
            pipeline=pipeline,
        )

        assert_purges_only_stale_text_chunk(vector_store, bm25)
        upserted_ids = {c.id for c in vector_store.upsert.call_args.args[0]}
        assert stable_id in upserted_ids

    def test_reingest_purges_removed_caption_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), figures=[])
        stale_id = caption_chunk_id(document.source, "figure-1")
        _, _, vector_store, _ = run_reingest_with_empty_structured_chunker(
            path,
            document,
            chunker_attr="_caption_chunker",
            old_chunk_ids=["old-text-chunk-1", stale_id],
        )

        deleted = vector_store.delete.call_args.args[0]
        assert stale_id in deleted
        assert "old-text-chunk-1" in deleted

    def test_reingest_preserves_stable_caption_chunks_in_real_bm25(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        reingest_preserves_stable_structured_chunks_in_real_bm25(
            path,
            document,
            structured_chunk=_embedded_caption_chunk(document),
            chunker_attr="_caption_chunker",
            stale_structured_text="stale caption content",
        )

    def test_full_reindex_on_skip_preserves_stable_caption_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        full_reindex_on_skip_preserves_stable_structured_chunk_ids(
            path,
            document,
            indexed_chunk=_embedded_caption_chunk(document),
            chunker_attr="_caption_chunker",
        )

    def test_caption_chunker_none_skips_indexing(self, tmp_path: Path) -> None:
        ingest_without_structured_chunker_indexes_text_only(tmp_path)

    def test_from_settings_wires_caption_chunker_when_enabled(self) -> None:
        pipeline = ingestion_pipeline_from_settings(
            parsing=MagicMock(
                table_chunks=MagicMock(enabled=False),
                caption_chunks=MagicMock(enabled=True),
            ),
        )
        assert isinstance(pipeline._caption_chunker, CaptionChunker)  # noqa: SLF001

    def test_skip_backfills_missing_caption_chunks_on_unchanged_hash(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        skip_backfills_missing_structured_chunks_on_unchanged_hash(
            path,
            document,
            indexed_chunk=_embedded_caption_chunk(document),
            chunker_attr="_caption_chunker",
            is_structured_chunk=is_caption_chunk,
        )

    def test_skip_unchanged_without_caption_chunker(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        skip_unchanged_without_structured_chunker(
            path,
            _doc(source=str(path.resolve())),
            chunker_attr="_caption_chunker",
        )

    def test_skip_skips_when_caption_chunks_already_indexed(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        result, service, vector_store, bm25 = _run_skip_with_indexed_caption_chunks(
            path,
            _doc_with_sample_caption(str(path.resolve())),
        )
        assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_skips_when_caption_chunks_already_indexed_after_reload(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        first_load = _doc_with_sample_caption(source)
        reloaded = _doc_with_sample_caption(source)
        assert first_load.id != reloaded.id
        result, service, vector_store, bm25 = _run_skip_with_indexed_caption_chunks(
            path,
            reloaded,
        )
        assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_backfill_noop_when_caption_chunker_returns_empty(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()))
        result, service, vector_store, bm25, _ = _run_skip_caption_chunker_ingest(
            path,
            document,
            caption_chunker=_mock_caption_chunker_with_empty_index(),
            chunk_ids=["text-chunk-1"],
        )

        assert result.skipped is True
        assert_skip_without_reindex(result, service, vector_store, bm25)

    def test_skip_purges_removed_caption_chunks_on_unchanged_hash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), figures=[])
        removed_caption_id = caption_chunk_id(document.source, "figure-1")
        result, service, vector_store, bm25, metadata, caption_chunker = (
            _run_skip_purge_only_caption_chunker_ingest(
                path,
                document,
                stale_caption_id=removed_caption_id,
                caplog=caplog,
            )
        )
        _assert_skip_purged_only_caption_chunks(
            result,
            service,
            caption_chunker,
            vector_store,
            bm25,
            metadata,
            stale_caption_id=removed_caption_id,
            merged_ids=["text-chunk-1"],
            index_called=False,
            caplog=caplog,
        )

    def test_skip_purges_stale_caption_chunks_when_layout_changes(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-2", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION_2}],
        )
        skip_purges_stale_structured_chunks_when_layout_changes(
            path,
            document,
            indexed_chunk=_embedded_caption_chunk(document),
            stale_chunk_id=caption_chunk_id(source, "figure-1"),
            chunker_attr="_caption_chunker",
        )

    def test_skip_skips_when_caption_embed_fails_on_layout_change(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(
            source=source,
            figures=[{FIGURE_ID_KEY: "figure-2", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION_2}],
        )
        stale_caption_id = caption_chunk_id(source, "figure-1")
        result, service, vector_store, bm25, metadata, caption_chunker = (
            _run_skip_purge_only_caption_chunker_ingest(
                path,
                document,
                stale_caption_id=stale_caption_id,
                caplog=caplog,
            )
        )

        assert result.skipped is True
        service.prepare.assert_not_called()
        caption_chunker.index.assert_called_once()
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        assert "Purged" not in caplog.text

    def test_skip_purges_caption_chunks_when_captions_removed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(source=source, figures=[{FIGURE_ID_KEY: "figure-1"}])
        stable_id = caption_chunk_id(source, "figure-1")
        result, service, vector_store, bm25, metadata, caption_chunker = (
            _run_skip_purge_only_caption_chunker_ingest(
                path,
                document,
                stale_caption_id=stable_id,
                caplog=caplog,
            )
        )
        _assert_skip_purged_only_caption_chunks(
            result,
            service,
            caption_chunker,
            vector_store,
            bm25,
            metadata,
            stale_caption_id=stable_id,
            merged_ids=["text-chunk-1"],
            index_called=False,
            caplog=caplog,
        )

    def test_reingest_purges_caption_chunks_when_captions_removed(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), figures=[{FIGURE_ID_KEY: "figure-1"}])
        stable_id = caption_chunk_id(document.source, "figure-1")
        _, metadata, vector_store, bm25 = run_reingest_with_empty_structured_chunker(
            path,
            document,
            chunker_attr="_caption_chunker",
            old_chunk_ids=["old-text-chunk-1", stable_id],
        )

        deleted = set(vector_store.delete.call_args.args[0])
        assert "old-text-chunk-1" in deleted
        assert stable_id in deleted
        bm25.remove_by_ids.assert_called_once()
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert stable_id not in merged_ids

    def test_reingest_retains_stable_caption_chunks_when_embedding_fails(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        stable_id = caption_chunk_id(document.source, "figure-1")
        _, metadata, vector_store, bm25 = run_reingest_with_empty_structured_chunker(
            path,
            document,
            chunker_attr="_caption_chunker",
            old_chunk_ids=["old-text-chunk-1", stable_id],
        )

        assert_purges_only_stale_text_chunk(vector_store, bm25)
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert stable_id in merged_ids

    def test_empty_index_with_failed_caption_embed_retains_prior_captions(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        stable_id = caption_chunk_id(document.source, "figure-1")
        metadata = mock_reingest_metadata(chunk_ids=[stable_id])
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = []
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[],
            metadata=metadata,
        )
        result = _run_caption_chunker_ingest(
            path,
            document,
            caption_chunker=caption_chunker,
            pipeline=pipeline,
        )

        assert result.chunk_count == 0
        vector_store.upsert.assert_not_called()
        vector_store.delete.assert_not_called()
        bm25.remove_by_ids.assert_not_called()
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert merged_ids == [stable_id]

    def test_skip_backfill_logs_upsert_message(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_caption(str(path.resolve()))
        caption = _embedded_caption_chunk(document)
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = [caption]
        with caplog.at_level(logging.INFO, logger=_INGESTION_PIPELINE):
            _run_skip_caption_chunker_ingest(
                path,
                document,
                caption_chunker=caption_chunker,
                chunk_ids=["text-chunk-1"],
            )
        assert "Backfilled 1 caption chunk(s)" in caplog.text

    def test_skip_returns_table_backfill_when_caption_noop(self, tmp_path: Path) -> None:
        from src.core.constants import TABLE_ID_KEY
        from src.rag.ingestion.table_chunker import build_table_chunks, is_table_chunk

        path = _report_pdf_path(tmp_path)
        sample_table = "| A | B |\n|---|---|\n| 1 | 2 |"
        document = _doc(
            source=str(path.resolve()),
            figures=[],
            content="Body text.",
        )
        document = document.model_copy(
            update={
                "metadata": {
                    **document.metadata,
                    "tables": [{TABLE_ID_KEY: "table-1", "text": sample_table}],
                    "figures": [],
                }
            }
        )
        table = build_table_chunks(document)[0].model_copy(
            update={"embedding": [0.1] * 4, "sparse_vector": {1: 0.9}}
        )
        table_chunker = MagicMock()
        table_chunker.index.return_value = [table]
        caption_chunker = _mock_caption_chunker_with_empty_index()
        metadata = unchanged_hash_metadata(path, document, chunk_ids=["text-chunk-1"])
        pipeline, service, vector_store, bm25 = mock_ingestion_pipeline(metadata=metadata)
        pipeline._table_chunker = table_chunker  # noqa: SLF001
        pipeline._caption_chunker = caption_chunker  # noqa: SLF001
        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=document,
        ):
            result = pipeline.ingest_file(path)

        assert result.skipped is False
        service.prepare.assert_not_called()
        caption_chunker.index.assert_not_called()
        table_chunker.index.assert_called_once()
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert is_table_chunk(upserted[0])
        bm25.add.assert_called_once()

    def test_skip_backfills_captions_on_empty_ocr_candidate_with_layout_captions(
        self, tmp_path: Path
    ) -> None:
        from src.rag.pipelines.ingestion_pipeline import source_file_hash

        path = _report_pdf_path(tmp_path)
        source = str(path.resolve())
        document = _doc(
            source=source,
            content="",
            figures=[
                {FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: _SAMPLE_CAPTION},
            ],
        )
        caption = _embedded_caption_chunk(document)
        caption_chunker = MagicMock()
        caption_chunker.index.return_value = [caption]
        file_hash = source_file_hash(source, path)
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=file_hash,
            chunk_count=1,
        )
        metadata.get_chunk_ids.return_value = ["text-chunk-1"]
        pipeline, service, vector_store, _ = mock_ingestion_pipeline(metadata=metadata)
        pipeline._caption_chunker = caption_chunker  # noqa: SLF001
        with (
            patch(
                "src.rag.pipelines.ingestion_pipeline.load_document",
                return_value=document,
            ),
            patch(
                "src.rag.pipelines.ingestion_pipeline.should_attempt_ocr",
                return_value=True,
            ),
        ):
            result = pipeline.ingest_file(path)

        assert result.skipped is False
        service.prepare.assert_not_called()
        caption_chunker.index.assert_called_once()
        vector_store.upsert.assert_called_once()
