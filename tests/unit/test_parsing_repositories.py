"""T-190 — parsing repository ABCs, ParsedDocument entity, and multimodal constants.

Verifies that:
- LayoutParserRepository and OcrRepository cannot be instantiated directly.
- Complete subclasses implementing all abstract methods CAN be instantiated.
- Incomplete subclasses CANNOT be instantiated (TypeError).
- ParsedDocument serialises cleanly and is immutable.
- Multimodal chunk constants are defined and distinct.
- No infrastructure imports leak into the domain layer.
"""

from __future__ import annotations

import importlib
from abc import ABC
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from src.core.constants import (
    ASSET_PATH_KEY,
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SECTION_KEY,
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_PAGE,
    CHUNK_TYPE_TABLE,
    CHUNK_TYPE_TO_MODALITY,
    FIGURE_CAPTION_KEY,
    FIGURE_ID_KEY,
    KNOWN_MODALITIES,
    LAYOUT_DOCUMENT_METADATA_KEYS,
    MODALITY_CAPTION,
    MODALITY_FIGURE,
    MODALITY_IMAGE,
    MODALITY_PAGE,
    MODALITY_TABLE,
    MODALITY_TEXT,
    TABLE_ID_KEY,
)
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories import LayoutParserRepository, OcrRepository, VisionRepository


def _assert_abstract_instantiation_fails(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()  # pyright: ignore[reportAbstractUsage]


def _assert_module_has_no_infra_imports(module: ModuleType) -> None:
    module_file = module.__file__
    if module_file is None:
        pytest.fail(f"module has no source file: {module.__name__}")
    source_path = Path(module_file)
    text = source_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from src.infrastructure") or stripped.startswith(
            "import src.infrastructure"
        ):
            pytest.fail(f"infrastructure import in domain layer: {stripped}")


class _LayoutParser(LayoutParserRepository):
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(source=str(path), content="parsed text")


class _Ocr(OcrRepository):
    def ocr(self, path: Path) -> str:
        return f"ocr:{path.name}"


class _Vision(VisionRepository):
    def caption_image(self, path: Path, *, prompt: str | None = None) -> str:
        return f"caption:{path.name}:{prompt or ''}"


# ── ParsedDocument ─────────────────────────────────────────────────────────────


class TestParsedDocument:
    def test_round_trip(self) -> None:
        doc = ParsedDocument(source="data/raw/manual.pdf", content="hello world")
        restored = ParsedDocument.model_validate(doc.model_dump())
        assert restored == doc

    def test_metadata_defaults_to_empty_dict(self) -> None:
        doc = ParsedDocument(source="a.pdf", content="x")
        assert doc.metadata == {}

    def test_custom_metadata(self) -> None:
        doc = ParsedDocument(
            source="a.pdf",
            content="x",
            metadata={CHUNK_PAGE_KEY: 3, CHUNK_SECTION_KEY: "Intro"},
        )
        assert doc.metadata[CHUNK_PAGE_KEY] == 3
        assert doc.metadata[CHUNK_SECTION_KEY] == "Intro"

    def test_frozen(self) -> None:
        doc = ParsedDocument(source="a.pdf", content="x")
        with pytest.raises(ValidationError):
            doc.content = "mutated"  # type: ignore[misc]

    def test_importable_from_entities_package(self) -> None:
        import src.domain.entities as entities

        assert entities.ParsedDocument is ParsedDocument


# ── LayoutParserRepository ─────────────────────────────────────────────────────


class TestLayoutParserRepository:
    def test_abc_cannot_be_instantiated(self) -> None:
        _assert_abstract_instantiation_fails(LayoutParserRepository)

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        class _Incomplete(LayoutParserRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            pass

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(_LayoutParser(), LayoutParserRepository)

    def test_parse_returns_parsed_document(self) -> None:
        path = Path("data/raw/sample.pdf")
        result = _LayoutParser().parse(path)
        assert isinstance(result, ParsedDocument)
        assert result.source == str(path)
        assert result.content == "parsed text"


# ── OcrRepository ──────────────────────────────────────────────────────────────


class TestOcrRepository:
    def test_abc_cannot_be_instantiated(self) -> None:
        _assert_abstract_instantiation_fails(OcrRepository)

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        class _Incomplete(OcrRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            pass

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(_Ocr(), OcrRepository)

    def test_ocr_returns_str(self) -> None:
        path = Path("data/raw/scan.png")
        assert _Ocr().ocr(path) == "ocr:scan.png"


# ── VisionRepository ───────────────────────────────────────────────────────────


class TestVisionRepository:
    def test_abc_cannot_be_instantiated(self) -> None:
        _assert_abstract_instantiation_fails(VisionRepository)

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        class _Incomplete(VisionRepository, ABC):  # pyright: ignore[reportAbstractUsage]
            pass

        _assert_abstract_instantiation_fails(_Incomplete)

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(_Vision(), VisionRepository)

    def test_caption_image_returns_str(self) -> None:
        path = Path("data/assets/figure-1.png")
        assert _Vision().caption_image(path) == "caption:figure-1.png:"
        assert _Vision().caption_image(path, prompt="brief") == "caption:figure-1.png:brief"


# ── Multimodal chunk constants ─────────────────────────────────────────────────


class TestMultimodalConstants:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("CHUNK_TYPE_TABLE", CHUNK_TYPE_TABLE),
            ("CHUNK_TYPE_CAPTION", CHUNK_TYPE_CAPTION),
            ("CHUNK_TYPE_FIGURE", CHUNK_TYPE_FIGURE),
            ("CHUNK_TYPE_PAGE", CHUNK_TYPE_PAGE),
            ("TABLE_ID_KEY", TABLE_ID_KEY),
            ("FIGURE_ID_KEY", FIGURE_ID_KEY),
            ("FIGURE_CAPTION_KEY", FIGURE_CAPTION_KEY),
            ("BBOX_KEY", BBOX_KEY),
            ("ASSET_PATH_KEY", ASSET_PATH_KEY),
            ("CHUNK_PAGE_KEY", CHUNK_PAGE_KEY),
            ("CHUNK_SECTION_KEY", CHUNK_SECTION_KEY),
            ("MODALITY_TEXT", MODALITY_TEXT),
            ("MODALITY_TABLE", MODALITY_TABLE),
            ("MODALITY_FIGURE", MODALITY_FIGURE),
            ("MODALITY_CAPTION", MODALITY_CAPTION),
            ("MODALITY_PAGE", MODALITY_PAGE),
            ("MODALITY_IMAGE", MODALITY_IMAGE),
        ],
    )
    def test_constant_is_non_empty_str(self, name: str, value: str) -> None:
        assert isinstance(value, str)
        assert value, f"{name} must not be empty"

    def test_chunk_type_values_are_unique(self) -> None:
        values = [CHUNK_TYPE_TABLE, CHUNK_TYPE_CAPTION, CHUNK_TYPE_FIGURE, CHUNK_TYPE_PAGE]
        assert len(values) == len(set(values))

    def test_modality_constants_match_known_set(self) -> None:
        assert {
            MODALITY_TEXT,
            MODALITY_TABLE,
            MODALITY_FIGURE,
            MODALITY_CAPTION,
            MODALITY_PAGE,
            MODALITY_IMAGE,
        } == KNOWN_MODALITIES
        assert CHUNK_TYPE_TO_MODALITY[CHUNK_TYPE_TABLE] == MODALITY_TABLE

    def test_metadata_keys_are_unique(self) -> None:
        keys = [
            TABLE_ID_KEY,
            FIGURE_ID_KEY,
            FIGURE_CAPTION_KEY,
            BBOX_KEY,
            ASSET_PATH_KEY,
            CHUNK_PAGE_KEY,
            CHUNK_SECTION_KEY,
        ]
        assert len(keys) == len(set(keys))

    def test_layout_document_metadata_keys(self) -> None:
        assert frozenset({"tables", "figures", "sections", "headings", "slides"}) == (
            LAYOUT_DOCUMENT_METADATA_KEYS
        )

    def test_multimodal_page_key_matches_chunk_metadata(self) -> None:
        """Layout parsers must populate metadata.page — not page_number."""
        assert CHUNK_PAGE_KEY == "page"

    def test_multimodal_section_key_matches_chunk_metadata(self) -> None:
        """Layout parsers must populate metadata.section — not section_title."""
        assert CHUNK_SECTION_KEY == "section"


# ── No infrastructure imports ──────────────────────────────────────────────────


class TestNoDomainInfraLeak:
    def test_repositories_importable_from_package(self) -> None:
        from src.domain.repositories import (  # noqa: F401
            LayoutParserRepository,
            OcrRepository,
            VisionRepository,
        )

    @pytest.mark.parametrize(
        "module_name",
        [
            "src.domain.repositories.layout_parser_repository",
            "src.domain.repositories.ocr_repository",
            "src.domain.repositories.vision_repository",
            "src.domain.entities.parsed_document",
            "src.domain.entities.source_reference",
        ],
    )
    def test_domain_module_has_no_infra_imports(self, module_name: str) -> None:
        _assert_module_has_no_infra_imports(importlib.import_module(module_name))


# ── CI migration script (T-180) ──────────────────────────────────────────────


class TestMigrateCiChecksScript:
    def test_script_exists_and_is_executable(self) -> None:
        script = Path("scripts/migrate_ci_checks.sh")
        assert script.is_file()
        assert script.stat().st_mode & 0o111

    def test_script_documents_migration_targets(self) -> None:
        text = Path("scripts/migrate_ci_checks.sh").read_text(encoding="utf-8")
        for label in ("Quality", "Unit Tests", "Extended Tests"):
            assert label in text
        for deprecated in (
            "Dependency Scan",
            "Lint",
            "Integration Tests",
            "Retrieval Eval Regression",
        ):
            assert deprecated in text
