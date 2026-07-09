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

from abc import ABC
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.constants import (
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SECTION_KEY,
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_PAGE,
    CHUNK_TYPE_TABLE,
    FIGURE_ID_KEY,
    TABLE_ID_KEY,
)
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories import LayoutParserRepository, OcrRepository


def _assert_abstract_instantiation_fails(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()  # pyright: ignore[reportAbstractUsage]


class _LayoutParser(LayoutParserRepository):
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(source=str(path), content="parsed text")


class _Ocr(OcrRepository):
    def ocr(self, path: Path) -> str:
        return f"ocr:{path.name}"


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
        from src.domain.entities import ParsedDocument as PD  # noqa: F401

        assert PD is ParsedDocument


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
            ("BBOX_KEY", BBOX_KEY),
            ("CHUNK_PAGE_KEY", CHUNK_PAGE_KEY),
            ("CHUNK_SECTION_KEY", CHUNK_SECTION_KEY),
        ],
    )
    def test_constant_is_non_empty_str(self, name: str, value: str) -> None:
        assert isinstance(value, str)
        assert value, f"{name} must not be empty"

    def test_chunk_type_values_are_unique(self) -> None:
        values = [CHUNK_TYPE_TABLE, CHUNK_TYPE_CAPTION, CHUNK_TYPE_FIGURE, CHUNK_TYPE_PAGE]
        assert len(values) == len(set(values))

    def test_metadata_keys_are_unique(self) -> None:
        keys = [
            TABLE_ID_KEY,
            FIGURE_ID_KEY,
            BBOX_KEY,
            CHUNK_PAGE_KEY,
            CHUNK_SECTION_KEY,
        ]
        assert len(keys) == len(set(keys))

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
        )

    def test_layout_parser_module_has_no_infra_imports(self) -> None:
        import src.domain.repositories.layout_parser_repository as mod

        source_path = Path(mod.__file__)
        text = source_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from src.infrastructure") or stripped.startswith(
                "import src.infrastructure"
            ):
                pytest.fail(f"infrastructure import in domain layer: {stripped}")

    def test_ocr_module_has_no_infra_imports(self) -> None:
        import src.domain.repositories.ocr_repository as mod

        source_path = Path(mod.__file__)
        text = source_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from src.infrastructure") or stripped.startswith(
                "import src.infrastructure"
            ):
                pytest.fail(f"infrastructure import in domain layer: {stripped}")

    def test_parsed_document_module_has_no_infra_imports(self) -> None:
        import src.domain.entities.parsed_document as mod

        source_path = Path(mod.__file__)
        text = source_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from src.infrastructure") or stripped.startswith(
                "import src.infrastructure"
            ):
                pytest.fail(f"infrastructure import in domain layer: {stripped}")


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
