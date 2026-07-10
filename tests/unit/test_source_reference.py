"""T-210 — multimodal domain model extensions.

Covers: modality constants, Chunk multimodal fields, SourceReference,
resolve_modality / from_chunk helpers, Answer.source_references, and
domain-layer import hygiene.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from src.core.constants import (
    ASSET_PATH_KEY,
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SECTION_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_PAGE,
    CHUNK_TYPE_TABLE,
    CHUNK_TYPE_TO_MODALITY,
    FIGURE_ID_KEY,
    KNOWN_MODALITIES,
    MODALITY_CAPTION,
    MODALITY_FIGURE,
    MODALITY_IMAGE,
    MODALITY_PAGE,
    MODALITY_TABLE,
    MODALITY_TEXT,
    TABLE_ID_KEY,
)
from src.domain.entities import (
    Answer,
    Chunk,
    SourceReference,
    resolve_modality,
    source_references_for_chunks,
)


def _assert_module_has_no_infra_imports(module: ModuleType) -> None:
    module_file = module.__file__
    if module_file is None:
        pytest.fail(f"module has no source file: {module.__name__}")
    text = Path(module_file).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from src.infrastructure") or stripped.startswith(
            "import src.infrastructure"
        ):
            pytest.fail(f"infrastructure import in domain layer: {stripped}")


# ── Modality constants ─────────────────────────────────────────────────────────


class TestModalityConstants:
    def test_known_modalities_complete(self) -> None:
        assert {
            MODALITY_TEXT,
            MODALITY_TABLE,
            MODALITY_FIGURE,
            MODALITY_CAPTION,
            MODALITY_PAGE,
            MODALITY_IMAGE,
        } == KNOWN_MODALITIES

    def test_modality_values_are_unique(self) -> None:
        assert len(KNOWN_MODALITIES) == 6

    def test_chunk_type_to_modality_mapping(self) -> None:
        assert CHUNK_TYPE_TO_MODALITY[CHUNK_TYPE_TABLE] == MODALITY_TABLE
        assert CHUNK_TYPE_TO_MODALITY[CHUNK_TYPE_FIGURE] == MODALITY_FIGURE
        assert CHUNK_TYPE_TO_MODALITY[CHUNK_TYPE_CAPTION] == MODALITY_CAPTION
        assert CHUNK_TYPE_TO_MODALITY[CHUNK_TYPE_PAGE] == MODALITY_PAGE

    def test_asset_path_key(self) -> None:
        assert ASSET_PATH_KEY == "asset_path"


# ── resolve_modality ───────────────────────────────────────────────────────────


class TestResolveModality:
    def test_defaults_to_text(self) -> None:
        assert resolve_modality() == MODALITY_TEXT

    def test_explicit_non_text_wins(self) -> None:
        assert (
            resolve_modality(modality=MODALITY_FIGURE, chunk_type=CHUNK_TYPE_TABLE)
            == MODALITY_FIGURE
        )

    def test_maps_chunk_type_when_modality_is_text(self) -> None:
        assert resolve_modality(chunk_type=CHUNK_TYPE_TABLE) == MODALITY_TABLE
        assert resolve_modality(chunk_type=CHUNK_TYPE_FIGURE) == MODALITY_FIGURE
        assert resolve_modality(chunk_type=CHUNK_TYPE_CAPTION) == MODALITY_CAPTION
        assert resolve_modality(chunk_type=CHUNK_TYPE_PAGE) == MODALITY_PAGE

    def test_unknown_chunk_type_keeps_modality(self) -> None:
        assert resolve_modality(chunk_type="proposition") == MODALITY_TEXT
        assert resolve_modality(modality=MODALITY_IMAGE, chunk_type="unknown") == MODALITY_IMAGE

    def test_none_chunk_type_keeps_modality(self) -> None:
        assert resolve_modality(modality=MODALITY_TEXT, chunk_type=None) == MODALITY_TEXT


# ── Chunk multimodal fields ────────────────────────────────────────────────────


class TestChunkMultimodalFields:
    def test_modality_defaults_to_text(self) -> None:
        chunk = Chunk(document_id="d", text="hello")
        assert chunk.modality == MODALITY_TEXT
        assert chunk.image_embedding is None
        assert chunk.asset_path is None

    def test_round_trip_with_multimodal_fields(self) -> None:
        chunk = Chunk(
            document_id="d",
            text="caption",
            modality=MODALITY_FIGURE,
            image_embedding=[0.1, 0.2],
            asset_path="data/assets/fig-1.png",
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_FIGURE, FIGURE_ID_KEY: "fig-1"},
        )
        restored = Chunk.model_validate(chunk.model_dump())
        assert restored == chunk

    def test_add_image_embedding_via_model_copy(self) -> None:
        chunk = Chunk(document_id="d", text="t", modality=MODALITY_IMAGE)
        updated = chunk.model_copy(update={"image_embedding": [1.0, 2.0]})
        assert updated.image_embedding == [1.0, 2.0]
        assert chunk.image_embedding is None

    def test_frozen_modality(self) -> None:
        chunk = Chunk(document_id="d", text="t")
        with pytest.raises(ValidationError):
            chunk.modality = MODALITY_TABLE  # type: ignore[misc]


# ── SourceReference ────────────────────────────────────────────────────────────


class TestSourceReference:
    def test_round_trip(self) -> None:
        ref = SourceReference(chunk_id="c1", modality=MODALITY_TABLE, table_id="table-1")
        assert SourceReference.model_validate(ref.model_dump()) == ref

    def test_defaults(self) -> None:
        ref = SourceReference(chunk_id="c1")
        assert ref.modality == MODALITY_TEXT
        assert ref.document_id is None
        assert ref.source is None
        assert ref.page is None
        assert ref.section is None
        assert ref.table_id is None
        assert ref.figure_id is None
        assert ref.bbox is None
        assert ref.snippet is None
        assert ref.score is None
        assert ref.asset_path is None

    def test_frozen(self) -> None:
        ref = SourceReference(chunk_id="c1")
        with pytest.raises(ValidationError):
            ref.chunk_id = "mutated"  # type: ignore[misc]

    def test_from_chunk_text_defaults(self) -> None:
        chunk = Chunk(id="c1", document_id="doc-1", text="hello")
        ref = SourceReference.from_chunk(chunk)
        assert ref.chunk_id == "c1"
        assert ref.document_id == "doc-1"
        assert ref.modality == MODALITY_TEXT
        assert ref.snippet is None
        assert ref.score is None

    def test_from_chunk_infers_table_modality_from_metadata(self) -> None:
        chunk = Chunk(
            id="c-table",
            document_id="doc-1",
            text="| a | b |",
            metadata={
                CHUNK_TYPE_KEY: CHUNK_TYPE_TABLE,
                TABLE_ID_KEY: "table-1",
                CHUNK_PAGE_KEY: 3,
                CHUNK_SECTION_KEY: "Results",
                CHUNK_SOURCE_KEY: "data/raw/report.pdf",
                BBOX_KEY: [0.0, 1.0, 2.0, 3.0],
            },
        )
        ref = SourceReference.from_chunk(chunk, score=0.91, snippet="| a | b |")
        assert ref.modality == MODALITY_TABLE
        assert ref.table_id == "table-1"
        assert ref.page == 3
        assert ref.section == "Results"
        assert ref.source == "data/raw/report.pdf"
        assert ref.bbox == [0.0, 1.0, 2.0, 3.0]
        assert ref.score == pytest.approx(0.91)
        assert ref.snippet == "| a | b |"

    def test_from_chunk_explicit_modality_wins_over_type(self) -> None:
        chunk = Chunk(
            document_id="d",
            text="x",
            modality=MODALITY_IMAGE,
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_FIGURE},
        )
        assert SourceReference.from_chunk(chunk).modality == MODALITY_IMAGE

    def test_from_chunk_asset_path_field_preferred(self) -> None:
        chunk = Chunk(
            document_id="d",
            text="fig",
            modality=MODALITY_FIGURE,
            asset_path="data/assets/from-field.png",
            metadata={ASSET_PATH_KEY: "data/assets/from-meta.png", FIGURE_ID_KEY: "fig-9"},
        )
        ref = SourceReference.from_chunk(chunk)
        assert ref.asset_path == "data/assets/from-field.png"
        assert ref.figure_id == "fig-9"

    def test_from_chunk_asset_path_from_metadata(self) -> None:
        chunk = Chunk(
            document_id="d",
            text="fig",
            metadata={ASSET_PATH_KEY: "data/assets/from-meta.png"},
        )
        assert SourceReference.from_chunk(chunk).asset_path == "data/assets/from-meta.png"

    def test_from_chunk_coerces_non_string_ids(self) -> None:
        chunk = Chunk(
            document_id="d",
            text="t",
            metadata={TABLE_ID_KEY: 42, FIGURE_ID_KEY: 7, CHUNK_SOURCE_KEY: 99},
        )
        ref = SourceReference.from_chunk(chunk)
        assert ref.table_id == "42"
        assert ref.figure_id == "7"
        assert ref.source == "99"

    def test_from_chunk_ignores_non_int_page(self) -> None:
        chunk = Chunk(document_id="d", text="t", metadata={CHUNK_PAGE_KEY: "3"})
        assert SourceReference.from_chunk(chunk).page is None

    def test_from_chunk_ignores_bool_page(self) -> None:
        chunk = Chunk(document_id="d", text="t", metadata={CHUNK_PAGE_KEY: True})
        assert SourceReference.from_chunk(chunk).page is None

    def test_from_chunk_ignores_invalid_bbox(self) -> None:
        bad_chunks = [
            Chunk(document_id="d", text="t", metadata={BBOX_KEY: "nope"}),
            Chunk(document_id="d", text="t", metadata={BBOX_KEY: [1, "x"]}),
            Chunk(document_id="d", text="t", metadata={BBOX_KEY: [1, True]}),
        ]
        for chunk in bad_chunks:
            assert SourceReference.from_chunk(chunk).bbox is None

    def test_from_chunk_bbox_from_tuple(self) -> None:
        chunk = Chunk(document_id="d", text="t", metadata={BBOX_KEY: (1, 2, 3, 4)})
        assert SourceReference.from_chunk(chunk).bbox == [1.0, 2.0, 3.0, 4.0]

    def test_from_chunk_non_string_chunk_type_ignored(self) -> None:
        chunk = Chunk(document_id="d", text="t", metadata={CHUNK_TYPE_KEY: 123})
        assert SourceReference.from_chunk(chunk).modality == MODALITY_TEXT

    def test_from_chunk_coerces_non_string_section(self) -> None:
        chunk = Chunk(document_id="d", text="t", metadata={CHUNK_SECTION_KEY: 5})
        assert SourceReference.from_chunk(chunk).section == "5"

    def test_importable_from_entities_package(self) -> None:
        import src.domain.entities as entities

        assert entities.SourceReference is SourceReference
        assert entities.resolve_modality is resolve_modality
        assert entities.source_references_for_chunks is source_references_for_chunks


# ── source_references_for_chunks ───────────────────────────────────────────────


class TestSourceReferencesForChunks:
    def test_empty(self) -> None:
        assert source_references_for_chunks([]) == []

    def test_maps_chunks_with_scores(self) -> None:
        c0 = Chunk(id="c0", document_id="d", text="a")
        c1 = Chunk(
            id="c1",
            document_id="d",
            text="b",
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_CAPTION},
        )
        refs = source_references_for_chunks([c0, c1], scores={"c0": 0.5, "c1": 0.8})
        assert len(refs) == 2
        assert refs[0].chunk_id == "c0"
        assert refs[0].score == pytest.approx(0.5)
        assert refs[0].modality == MODALITY_TEXT
        assert refs[1].modality == MODALITY_CAPTION
        assert refs[1].score == pytest.approx(0.8)

    def test_scores_none_defaults(self) -> None:
        chunk = Chunk(id="c0", document_id="d", text="a")
        refs = source_references_for_chunks([chunk])
        assert refs[0].score is None


# ── Answer.source_references ───────────────────────────────────────────────────


class TestAnswerSourceReferences:
    def test_defaults_to_empty(self) -> None:
        answer = Answer(query_id="q", text="a")
        assert answer.source_references == []
        assert answer.sources == []

    def test_round_trip_with_references(self) -> None:
        ref = SourceReference(chunk_id="c1", modality=MODALITY_TABLE, table_id="t1")
        answer = Answer(
            query_id="q",
            text="answer",
            sources=["c1"],
            source_references=[ref],
        )
        restored = Answer.model_validate(answer.model_dump())
        assert restored == answer
        assert restored.source_references[0].table_id == "t1"

    def test_backward_compatible_without_source_references_key(self) -> None:
        answer = Answer.model_validate({"query_id": "q", "text": "a", "sources": ["c0"]})
        assert answer.source_references == []
        assert answer.sources == ["c0"]


# ── Domain import hygiene ──────────────────────────────────────────────────────


class TestNoDomainInfraLeak:
    @pytest.mark.parametrize(
        "module_name",
        [
            "src.domain.entities.source_reference",
            "src.domain.entities.chunk",
        ],
    )
    def test_domain_module_has_no_infra_imports(self, module_name: str) -> None:
        _assert_module_has_no_infra_imports(importlib.import_module(module_name))
