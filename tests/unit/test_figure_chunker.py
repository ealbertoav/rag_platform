"""T-253 — Structured figure chunks with image vectors at ingesting."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import ASSET_PATH_KEY, FIGURE_ID_KEY
from src.core.exceptions import EmbeddingError
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.ingestion.figure_chunker import (
    FigureChunker,
    collect_figure_ids,
    existing_figure_chunk_ids,
    figure_chunks_needing_upsert,
    known_figure_chunk_ids,
    merged_figure_chunk_ids,
    metadata_figure_ids,
    retained_figure_chunk_ids_on_embed_failure,
    stale_figure_ids_safe_to_purge,
)
from src.rag.ingestion.figure_extractor import build_figure_chunks, figure_chunk_id, is_figure_chunk
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline, IngestionResult
from tests.unit.ingestion_helpers import (
    assert_purges_only_stale_text_chunk,
    assert_skip_without_reindex,
    embedded_chunk,
    ingest_without_structured_chunker_indexes_text_only,
    ingestion_pipeline_from_settings,
    mock_chunker_with_empty_index,
    mock_ingestion_pipeline,
    mock_reingest_metadata,
    run_reingest_with_empty_structured_chunker,
    run_skip_structured_chunker_ingest,
    run_structured_chunker_ingest,
    skip_backfills_missing_structured_chunks_on_unchanged_hash,
    skip_unchanged_without_structured_chunker,
)

_FIGURE_CHUNKER = "src.rag.ingestion.figure_chunker"
_INGESTION_PIPELINE = "src.rag.pipelines.ingestion_pipeline"


def _internal(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


build_figure_chunker = cast(
    Callable[..., object | None],
    _internal(_INGESTION_PIPELINE, "_build_figure_chunker"),
)


def _doc(
    *,
    content: str = "Body text.",
    figures: list[dict[str, Any]] | None = None,
    source: str = "/tmp/report.pdf",
) -> Document:
    metadata: dict[str, Any] = {"loader": "docling", "filename": "report.pdf"}
    if figures is not None:
        metadata["figures"] = figures
    return Document(source=source, content=content, metadata=metadata)


def _figure_entry(
    *,
    figure_id: str = "figure-1",
    asset_path: str = "/a/figure-1.png",
    caption: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {FIGURE_ID_KEY: figure_id, ASSET_PATH_KEY: asset_path}
    if caption is not None:
        entry["caption"] = caption
    return entry


def _report_pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path


def _doc_with_sample_figure(source: str) -> Document:
    return _doc(source=source, figures=[_figure_entry()])


def _embedded_figure_chunk(document: Document) -> object:
    figure = build_figure_chunks(document)[0]
    return figure.model_copy(update={"embedding": [0.1] * 4, "sparse_vector": {1: 0.9}})


def _run_figure_chunker_ingest(
    path: Path,
    document: Document,
    *,
    figure_chunker: MagicMock,
    pipeline: IngestionPipeline,
) -> IngestionResult:
    return run_structured_chunker_ingest(
        path,
        document,
        chunker=figure_chunker,
        chunker_attr="_figure_chunker",
        pipeline=pipeline,
    )


def _run_skip_figure_chunker_ingest(
    path: Path,
    document: Document,
    *,
    figure_chunker: MagicMock,
    chunk_ids: list[str],
) -> tuple[IngestionResult, MagicMock, MagicMock, MagicMock, MagicMock]:
    return run_skip_structured_chunker_ingest(
        path,
        document,
        chunker=figure_chunker,
        chunker_attr="_figure_chunker",
        chunk_ids=chunk_ids,
    )


class TestFigureIdHelpers:
    def test_known_figure_chunk_ids_maps_layout_figure_ids(self) -> None:
        source = "/tmp/report.pdf"
        assert known_figure_chunk_ids(source, ["figure-1", "figure-2"]) == {
            figure_chunk_id(source, "figure-1"),
            figure_chunk_id(source, "figure-2"),
        }

    def test_collect_figure_ids_merges_metadata_and_indexed_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[_figure_entry(figure_id="figure-2")])
        existing = [figure_chunk_id(source, "figure-1"), "text-chunk-1"]
        assert collect_figure_ids(document, existing) == {"figure-1", "figure-2"}

    def test_existing_figure_chunk_ids_uses_bm25_payloads(self) -> None:
        source = "/tmp/report.pdf"
        custom_id = figure_chunk_id(source, "custom-layout-id")
        document = _doc(source=source, figures=[])
        bm25 = MagicMock()
        chunk = build_figure_chunks(
            _doc(source=source, figures=[_figure_entry(figure_id="custom-layout-id")])
        )[0].model_copy(update={"id": custom_id})
        bm25.get_by_id.side_effect = lambda chunk_id: chunk if chunk_id == custom_id else None
        assert existing_figure_chunk_ids(
            source,
            [custom_id, "text-chunk-1"],
            document=document,
            bm25=bm25,
        ) == {custom_id}

    def test_existing_figure_chunk_ids_without_bm25_uses_layout_ids(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[_figure_entry()])
        assert existing_figure_chunk_ids(
            source,
            ["text-chunk-1"],
            document=document,
            bm25=None,
        ) == {figure_chunk_id(source, "figure-1")}


class TestMetadataFigureIds:
    def test_returns_ids_with_asset_path(self) -> None:
        document = _doc(
            figures=[_figure_entry(figure_id="figure-1"), _figure_entry(figure_id="figure-2")]
        )
        assert metadata_figure_ids(document) == {"figure-1", "figure-2"}

    def test_skips_entries_without_asset_path(self) -> None:
        document = _doc(figures=[{FIGURE_ID_KEY: "figure-1"}, _figure_entry(figure_id="figure-2")])
        assert metadata_figure_ids(document) == {"figure-2"}

    def test_returns_empty_without_figures(self) -> None:
        assert metadata_figure_ids(_doc()) == set()
        assert metadata_figure_ids(_doc(figures=[])) == set()


class TestFigureSafetyHelpers:
    def test_stale_figure_ids_safe_to_purge_only_after_successful_sync(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[_figure_entry(figure_id="figure-2")])
        built = build_figure_chunks(document)
        stale = [figure_chunk_id(source, "figure-1")]
        assert stale_figure_ids_safe_to_purge(document, built, [], stale) == []
        embedded = [_embedded_figure_chunk(document)]
        assert stale_figure_ids_safe_to_purge(document, built, embedded, stale) == stale

    def test_retained_figure_chunk_ids_on_embed_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[_figure_entry()])
        built = build_figure_chunks(document)
        stable_id = built[0].id
        assert retained_figure_chunk_ids_on_embed_failure(
            source,
            document,
            ["text-chunk-1", stable_id],
            built,
            [],
        ) == {stable_id}
        embedded = [_embedded_figure_chunk(document)]
        assert (
            retained_figure_chunk_ids_on_embed_failure(
                source,
                document,
                ["text-chunk-1", stable_id],
                built,
                embedded,
            )
            == set()
        )

    def test_merged_figure_chunk_ids_after_successful_embed(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[_figure_entry(figure_id="figure-2")])
        built = build_figure_chunks(document)
        embedded = [_embedded_figure_chunk(document)]
        known = {figure_chunk_id(source, "figure-1"), embedded[0].id}
        assert merged_figure_chunk_ids(
            ["text-chunk-1", figure_chunk_id(source, "figure-1")],
            known,
            document,
            built,
            embedded,
        ) == [embedded[0].id]

    def test_merged_figure_chunk_ids_keeps_existing_on_embed_failure(self) -> None:
        source = "/tmp/report.pdf"
        document = _doc(source=source, figures=[_figure_entry(figure_id="figure-2")])
        built = build_figure_chunks(document)
        stale_id = figure_chunk_id(source, "figure-1")
        new_id = built[0].id
        known = {stale_id, new_id}
        assert merged_figure_chunk_ids(
            ["text-chunk-1", stale_id],
            known,
            document,
            built,
            [],
        ) == [stale_id]


class TestFigureChunksNeedingUpsert:
    def test_returns_new_chunks(self) -> None:
        document = _doc(figures=[_figure_entry()])
        embedded = _embedded_figure_chunk(document)
        assert figure_chunks_needing_upsert([embedded], ["text-chunk-1"]) == [embedded]

    def test_skips_unchanged_indexed_chunks(self) -> None:
        document = _doc(figures=[_figure_entry()])
        embedded = _embedded_figure_chunk(document)
        bm25 = MagicMock()
        bm25.get_by_id.return_value = embedded.model_copy(update={"embedding": None})
        assert (
            figure_chunks_needing_upsert(
                [embedded],
                ["text-chunk-1", embedded.id],
                bm25=bm25,
            )
            == []
        )

    def test_returns_chunks_with_updated_text(self) -> None:
        document = _doc(figures=[_figure_entry(caption="new caption")])
        embedded = _embedded_figure_chunk(document)
        bm25 = MagicMock()
        bm25.get_by_id.return_value = embedded.model_copy(update={"text": "[figure]"})
        assert figure_chunks_needing_upsert(
            [embedded],
            ["text-chunk-1", embedded.id],
            bm25=bm25,
        ) == [embedded]


class TestFigureChunker:
    @staticmethod
    def _embedder(
        *,
        image_vecs: list[list[float]] | None = None,
        image_error: Exception | None = None,
    ) -> MagicMock:
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1] * 4], [{1: 0.9}])
        if image_error is not None:
            embedder.embed_image.side_effect = image_error
        else:
            default_vecs = [[0.5, 0.5]]
            embedder.embed_image.return_value = default_vecs if image_vecs is None else image_vecs
        return embedder

    def test_index_returns_empty_when_no_figures(self) -> None:
        embedder = self._embedder()
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        assert chunker.index(_doc()) == []
        embedder.embed_both.assert_not_called()

    def test_index_returns_empty_on_text_embedding_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _doc(figures=[_figure_entry()])
        embedder = MagicMock()
        embedder.embed_both.side_effect = RuntimeError("embed failed")
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger=_FIGURE_CHUNKER):
            assert chunker.index(document) == []
        assert "Embedding figure chunks failed" in caplog.text

    def test_index_attaches_image_embedding_when_supported(self) -> None:
        document = _doc(figures=[_figure_entry(asset_path="/a/figure-1.png")])
        embedder = self._embedder(image_vecs=[[0.5, 0.5]])
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        result = chunker.index(document)
        assert len(result) == 1
        assert result[0].embedding == [0.1] * 4
        assert result[0].sparse_vector == {1: 0.9}
        assert result[0].image_embedding == [0.5, 0.5]
        embedder.embed_image.assert_called_once_with([Path("/a/figure-1.png")])

    def test_index_batches_image_embedding_across_figures(self) -> None:
        document = _doc(
            figures=[
                _figure_entry(figure_id="figure-1", asset_path="/a/1.png"),
                _figure_entry(figure_id="figure-2", asset_path="/a/2.png"),
            ]
        )
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1] * 4, [0.2] * 4], [{1: 0.9}, {2: 0.8}])
        embedder.embed_image.return_value = [[0.5, 0.5], [0.6, 0.6]]
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        result = chunker.index(document)
        assert [c.image_embedding for c in result] == [[0.5, 0.5], [0.6, 0.6]]
        embedder.embed_image.assert_called_once_with([Path("/a/1.png"), Path("/a/2.png")])

    def test_index_leaves_image_embedding_none_when_provider_unsupported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _doc(figures=[_figure_entry()])
        embedder = self._embedder(image_error=EmbeddingError("no image support"))
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        with caplog.at_level(logging.DEBUG, logger=_FIGURE_CHUNKER):
            result = chunker.index(document)
        assert len(result) == 1
        assert result[0].embedding is not None
        assert result[0].image_embedding is None

    def test_index_leaves_image_embedding_none_on_unexpected_image_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = _doc(figures=[_figure_entry()])
        embedder = self._embedder(image_error=RuntimeError("boom"))
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING, logger=_FIGURE_CHUNKER):
            result = chunker.index(document)
        assert result[0].image_embedding is None
        assert "Embedding figure images failed" in caplog.text

    def test_attach_image_embeddings_noop_without_asset_path(self) -> None:
        embedder = self._embedder()
        chunker = FigureChunker(embedder=embedder)  # type: ignore[arg-type]
        chunk = embedded_chunk(0)
        assert chunk.asset_path is None
        result = chunker._attach_image_embeddings([chunk], "/tmp/report.pdf")  # noqa: SLF001
        assert result == [chunk]
        embedder.embed_image.assert_not_called()


class TestBuildFigureChunker:
    def test_returns_none_when_disabled(self) -> None:
        assert build_figure_chunker(MagicMock(), type("Cfg", (), {"enabled": False})()) is None

    def test_returns_chunker_when_enabled(self) -> None:
        embedder = MagicMock(spec=EmbeddingRepository)
        chunker = build_figure_chunker(embedder, type("Cfg", (), {"enabled": True})())
        assert isinstance(chunker, FigureChunker)


class TestIngestionPipelineFigureChunks:
    def test_figure_chunks_indexed_in_qdrant_and_bm25(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        base = embedded_chunk(0)
        document = _doc_with_sample_figure(str(path.resolve()))
        figure = _embedded_figure_chunk(document)
        figure_chunker = MagicMock()
        figure_chunker.index.return_value = [figure]
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline([base])
        pipeline._figure_chunker = figure_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=document,
        ):
            result = pipeline.ingest_file(path)

        assert result.chunk_count == 1
        upserted = vector_store.upsert.call_args.args[0]
        assert len(upserted) == 2
        assert any(is_figure_chunk(c) for c in upserted)
        bm25_added = bm25.add.call_args.args[0]
        assert any(is_figure_chunk(c) for c in bm25_added)

    def test_figure_chunker_none_skips_indexing(self, tmp_path: Path) -> None:
        ingest_without_structured_chunker_indexes_text_only(tmp_path)

    def test_from_settings_wires_figure_chunker_when_enabled(self) -> None:
        pipeline = ingestion_pipeline_from_settings(
            parsing=MagicMock(figure_chunks=MagicMock(enabled=True)),
        )
        assert isinstance(pipeline._figure_chunker, FigureChunker)  # noqa: SLF001

    def test_from_settings_skips_figure_chunker_when_disabled(self) -> None:
        pipeline = ingestion_pipeline_from_settings(
            parsing=MagicMock(figure_chunks=MagicMock(enabled=False)),
        )
        assert pipeline._figure_chunker is None  # noqa: SLF001

    def test_skip_backfills_missing_figure_chunks_on_unchanged_hash(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_figure(str(path.resolve()))
        skip_backfills_missing_structured_chunks_on_unchanged_hash(
            path,
            document,
            indexed_chunk=_embedded_figure_chunk(document),
            chunker_attr="_figure_chunker",
            is_structured_chunk=is_figure_chunk,
        )

    def test_skip_unchanged_without_figure_chunker(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        skip_unchanged_without_structured_chunker(
            path,
            _doc(source=str(path.resolve())),
            chunker_attr="_figure_chunker",
        )

    def test_reingest_retains_stable_figure_chunks_when_embedding_fails(
        self, tmp_path: Path
    ) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc_with_sample_figure(str(path.resolve()))
        stable_id = figure_chunk_id(document.source, "figure-1")
        _, metadata, vector_store, bm25 = run_reingest_with_empty_structured_chunker(
            path,
            document,
            chunker_attr="_figure_chunker",
            old_chunk_ids=["old-text-chunk-1", stable_id],
        )

        assert_purges_only_stale_text_chunk(vector_store, bm25)
        _, _, merged_ids = metadata.upsert_document.call_args.args
        assert stable_id in merged_ids

    def test_reingest_purges_removed_figure_chunk_ids(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()), figures=[])
        removed_figure_id = figure_chunk_id(document.source, "figure-1")
        metadata = mock_reingest_metadata(chunk_ids=["text-chunk-1", removed_figure_id])
        figure_chunker = MagicMock()
        figure_chunker.index.return_value = []
        pipeline, _, vector_store, bm25 = mock_ingestion_pipeline(
            prepared_chunks=[embedded_chunk(0)],
            metadata=metadata,
        )
        pipeline._figure_chunker = figure_chunker  # noqa: SLF001

        with patch(
            "src.rag.pipelines.ingestion_pipeline.load_document",
            return_value=document,
        ):
            pipeline.ingest_file(path)

        vector_store.delete.assert_called_once_with(["text-chunk-1", removed_figure_id])
        bm25.remove_by_ids.assert_called_once_with(["text-chunk-1", removed_figure_id])

    def test_skip_backfill_noop_when_figure_chunker_returns_empty(self, tmp_path: Path) -> None:
        path = _report_pdf_path(tmp_path)
        document = _doc(source=str(path.resolve()))
        result, service, vector_store, bm25, _ = _run_skip_figure_chunker_ingest(
            path,
            document,
            figure_chunker=mock_chunker_with_empty_index(),
            chunk_ids=["text-chunk-1"],
        )

        assert result.skipped is True
        assert_skip_without_reindex(result, service, vector_store, bm25)
