"""T-230 — figure asset extraction and Chunk builders."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pptx as python_pptx
import pytest
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches

from src.core.constants import (
    ASSET_PATH_KEY,
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_KEY,
    FIGURE_ID_KEY,
    MODALITY_FIGURE,
)
from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.ingestion.figure_extractor import (
    apply_figure_assets,
    build_figure_chunks,
    figure_chunk_id,
    is_figure_chunk,
)
from src.rag.ingestion.local_asset_store import LocalAssetStore

_EXTRACTOR = "src.rag.ingestion.figure_extractor"


def _settings(*, enabled: bool = True, store_dir: str | None = None) -> Settings:
    cfg: dict[str, Any] = {"enabled": enabled}
    if store_dir is not None:
        cfg["store_dir"] = store_dir
    return Settings(parsing={"figure_assets": cfg})


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (8, 8), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_png(fp: Any, *_args: Any, **_kwargs: Any) -> None:
    fp.write(_png_bytes())


def _save_empty(*_args: Any, **_kwargs: Any) -> None:
    return None


def _fake_pil_image(*, empty: bool = False) -> SimpleNamespace:
    return SimpleNamespace(save=_save_empty if empty else _save_png)


def _pptx_with_picture(tmp_path: Path) -> Path:
    png_path = tmp_path / "tiny.png"
    png_path.write_bytes(_png_bytes())
    pptx_path = tmp_path / "deck.pptx"
    presentation = python_pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.shapes.add_picture(str(png_path), Inches(0.5), Inches(0.5), width=Inches(1))
    presentation.save(str(pptx_path))
    return pptx_path


def _figure_pdf_document(
    tmp_path: Path,
    *,
    figures: list[Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Document]:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF")
    metadata: dict[str, Any] = {
        "figures": figures if figures is not None else [{FIGURE_ID_KEY: "figure-1"}],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return path, Document(source=str(path.resolve()), content="text", metadata=metadata)


def _apply_with_docling_pictures(
    document: Document,
    path: Path,
    store_dir: Path,
    pictures: list[Any],
    *,
    status_name: str | None = "SUCCESS",
    convert_error: BaseException | None = None,
) -> Document:
    converter = MagicMock()
    if convert_error is not None:
        converter.convert.side_effect = convert_error
    else:
        status = None if status_name is None else SimpleNamespace(name=status_name)
        converter.convert.return_value = SimpleNamespace(
            status=status,
            document=SimpleNamespace(pictures=list(pictures)),
        )
    with patch(f"{_EXTRACTOR}._create_picture_converter", return_value=converter):
        return apply_figure_assets(
            document,
            path,
            app_settings=_settings(enabled=True),
            store=LocalAssetStore(store_dir),
        )


class TestFigureChunkHelpers:
    def test_figure_chunk_id_stable(self) -> None:
        assert figure_chunk_id("/a.pdf", "figure-1") == figure_chunk_id("/a.pdf", "figure-1")
        assert figure_chunk_id("/a.pdf", "figure-1") != figure_chunk_id("/a.pdf", "figure-2")

    def test_is_figure_chunk(self) -> None:
        chunk = Chunk(
            document_id="d1",
            text="caption",
            metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_FIGURE},
            modality=MODALITY_FIGURE,
            asset_path="/assets/f.png",
        )
        assert is_figure_chunk(chunk) is True
        assert is_figure_chunk(Chunk(document_id="d1", text="x")) is False


class TestBuildFigureChunks:
    def test_empty_without_figures(self) -> None:
        doc = Document(source="/doc.pdf", content="text")
        assert build_figure_chunks(doc) == []

    def test_skips_invalid_entries(self) -> None:
        doc = Document(
            source="/doc.pdf",
            content="text",
            metadata={
                "figures": [
                    "bad",
                    {FIGURE_ID_KEY: "figure-1"},  # no asset_path
                    {
                        FIGURE_ID_KEY: "figure-2",
                        ASSET_PATH_KEY: "/a/f2.png",
                        "caption": "  Chart  ",
                        CHUNK_PAGE_KEY: 3,
                        BBOX_KEY: [1.0, 2.0, 3.0, 4.0],
                    },
                ]
            },
        )
        chunks = build_figure_chunks(doc)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.id == figure_chunk_id("/doc.pdf", "figure-2")
        assert chunk.text == "Chart"
        assert chunk.modality == MODALITY_FIGURE
        assert chunk.asset_path == "/a/f2.png"
        assert chunk.metadata[FIGURE_ID_KEY] == "figure-2"
        assert chunk.metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_FIGURE
        assert chunk.metadata[CHUNK_PAGE_KEY] == 3
        assert chunk.metadata[BBOX_KEY] == [1.0, 2.0, 3.0, 4.0]

    def test_default_text_without_caption(self) -> None:
        doc = Document(
            source="/doc.pdf",
            content="text",
            metadata={
                "figures": [{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: "/a/f1.png"}],
            },
        )
        chunks = build_figure_chunks(doc)
        assert chunks[0].text == "[figure]"


class TestApplyFigureAssetsDisabled:
    def test_noop_when_disabled(self, tmp_path: Path) -> None:
        path = tmp_path / "deck.pptx"
        path.write_bytes(b"not-used")
        doc = Document(source=str(path), content="slides")
        result = apply_figure_assets(doc, path, app_settings=_settings(enabled=False))
        assert result is doc

    def test_uses_default_settings_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PARSING__FIGURE_ASSETS__ENABLED", "false")
        path = tmp_path / "x.pdf"
        path.write_text("x")
        doc = Document(source=str(path), content="text")
        # Reload settings via explicit Settings() path is covered by app_settings=None
        # with the live singleton still disabled by YAML default.
        result = apply_figure_assets(doc, path, app_settings=None)
        assert result is doc


class TestApplyPptxFigures:
    def test_persists_picture_and_sets_asset_path(self, tmp_path: Path) -> None:
        pptx_path = _pptx_with_picture(tmp_path)
        store = LocalAssetStore(tmp_path / "assets")
        doc = Document(source=str(pptx_path.resolve()), content="slide text")
        result = apply_figure_assets(
            doc,
            pptx_path,
            app_settings=_settings(enabled=True, store_dir=str(tmp_path / "assets")),
            store=store,
        )
        figures = result.metadata["figures"]
        assert len(figures) == 1
        assert figures[0][FIGURE_ID_KEY] == "figure-1"
        assert figures[0][CHUNK_PAGE_KEY] == 1
        asset = Path(figures[0][ASSET_PATH_KEY])
        assert asset.is_file()
        assert asset.read_bytes()

        chunks = build_figure_chunks(result)
        assert len(chunks) == 1
        assert chunks[0].asset_path == str(asset)
        assert chunks[0].metadata[FIGURE_ID_KEY] == "figure-1"

    def test_merges_existing_figure_metadata(self, tmp_path: Path) -> None:
        pptx_path = _pptx_with_picture(tmp_path)
        store = LocalAssetStore(tmp_path / "assets")
        doc = Document(
            source=str(pptx_path.resolve()),
            content="slide text",
            metadata={
                "figures": [
                    {FIGURE_ID_KEY: "figure-1", "caption": "Existing caption"},
                ]
            },
        )
        result = apply_figure_assets(
            doc,
            pptx_path,
            app_settings=_settings(enabled=True),
            store=store,
        )
        assert result.metadata["figures"][0]["caption"] == "Existing caption"
        assert ASSET_PATH_KEY in result.metadata["figures"][0]

    def test_soft_fails_store_error(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        pptx_path = _pptx_with_picture(tmp_path)
        store = MagicMock(spec=LocalAssetStore)
        store.save.side_effect = OSError("disk full")
        doc = Document(source=str(pptx_path.resolve()), content="slide text")
        with caplog.at_level(logging.WARNING):
            result = apply_figure_assets(
                doc,
                pptx_path,
                app_settings=_settings(enabled=True),
                store=store,
            )
        assert ASSET_PATH_KEY not in result.metadata["figures"][0]
        assert "Failed to store PPTX figure" in caplog.text

    def test_skips_unreadable_picture(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pptx_path = _pptx_with_picture(tmp_path)
        doc = Document(source=str(pptx_path.resolve()), content="slide text")
        store = LocalAssetStore(tmp_path / "assets")

        class ExplodingImage:
            def __getattr__(self, name: str) -> object:
                raise RuntimeError("boom")

        exploding_shape = SimpleNamespace(
            shape_type=MSO_SHAPE_TYPE.PICTURE,
            image=ExplodingImage(),
        )
        good_shape = SimpleNamespace(
            shape_type=MSO_SHAPE_TYPE.PICTURE,
            image=SimpleNamespace(blob=_png_bytes((0, 255, 0)), ext="png"),
        )
        empty_shape = SimpleNamespace(
            shape_type=MSO_SHAPE_TYPE.PICTURE,
            image=SimpleNamespace(blob=b"", ext="png"),
        )
        text_shape = SimpleNamespace(shape_type=1)
        slide = SimpleNamespace(shapes=[exploding_shape, empty_shape, text_shape, good_shape])
        presentation = SimpleNamespace(slides=[slide])

        with (
            patch("pptx.Presentation", return_value=presentation),
            caplog.at_level(logging.WARNING),
        ):
            result = apply_figure_assets(
                doc,
                pptx_path,
                app_settings=_settings(enabled=True),
                store=store,
            )
        assert len(result.metadata["figures"]) == 1
        assert ASSET_PATH_KEY in result.metadata["figures"][0]
        assert "Skipping unreadable PPTX picture" in caplog.text


class TestApplyDoclingFigures:
    def test_noop_without_layout_figures(self, tmp_path: Path) -> None:
        path, doc = _figure_pdf_document(tmp_path, figures=[], extra_metadata={"loader": "docling"})
        result = apply_figure_assets(
            doc,
            path,
            app_settings=_settings(enabled=True),
            store=LocalAssetStore(tmp_path / "assets"),
        )
        assert result is doc

    def test_persists_docling_picture(self, tmp_path: Path) -> None:
        path, doc = _figure_pdf_document(
            tmp_path,
            figures=[
                {
                    FIGURE_ID_KEY: "figure-1",
                    "caption": "Plot",
                    CHUNK_PAGE_KEY: 2,
                    BBOX_KEY: [0.0, 0.0, 10.0, 10.0],
                },
                "skip-me",
                {FIGURE_ID_KEY: "figure-2"},
            ],
            extra_metadata={"loader": "docling"},
        )
        picture = MagicMock()
        picture.get_image.return_value = _fake_pil_image()
        result = _apply_with_docling_pictures(doc, path, tmp_path / "assets", [picture])

        figures = [f for f in result.metadata["figures"] if isinstance(f, dict)]
        assert figures[0][ASSET_PATH_KEY]
        assert Path(figures[0][ASSET_PATH_KEY]).is_file()
        # Second valid figure has no matching picture → no asset_path
        assert ASSET_PATH_KEY not in figures[1]
        chunks = build_figure_chunks(result)
        assert len(chunks) == 1
        assert chunks[0].text == "Plot"

    def test_handles_missing_image_bytes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        picture = MagicMock()
        picture.get_image.return_value = None
        with caplog.at_level(logging.WARNING):
            result = _apply_with_docling_pictures(
                doc, path, tmp_path / "assets", [picture], status_name=None
            )
        assert ASSET_PATH_KEY not in result.metadata["figures"][0]
        assert "No image bytes" in caplog.text

    def test_handles_export_exception(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        picture = MagicMock()
        picture.get_image.side_effect = RuntimeError("export failed")
        with caplog.at_level(logging.WARNING):
            result = _apply_with_docling_pictures(doc, path, tmp_path / "assets", [picture])
        assert ASSET_PATH_KEY not in result.metadata["figures"][0]
        assert "Failed to export Docling figure" in caplog.text

    def test_conversion_failure_status(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        with caplog.at_level(logging.WARNING):
            result = _apply_with_docling_pictures(
                doc, path, tmp_path / "assets", [], status_name="FAILURE"
            )
        assert result is doc or result.metadata.get("figures") == doc.metadata["figures"]
        assert "Figure asset extraction failed" in caplog.text

    def test_convert_raises_document_load_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        with caplog.at_level(logging.WARNING):
            result = _apply_with_docling_pictures(
                doc,
                path,
                tmp_path / "assets",
                [],
                convert_error=RuntimeError("boom"),
            )
        assert result is doc
        assert "Figure asset extraction failed" in caplog.text

    def test_configuration_error_soft_fails(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        with (
            patch(
                f"{_EXTRACTOR}._create_picture_converter",
                side_effect=ConfigurationError("missing docling"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = apply_figure_assets(
                doc,
                path,
                app_settings=_settings(enabled=True),
                store=LocalAssetStore(tmp_path / "assets"),
            )
        assert result is doc
        assert "misconfigured" in caplog.text

    def test_empty_pictures_list(self, tmp_path: Path) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        result = _apply_with_docling_pictures(doc, path, tmp_path / "assets", [])
        assert result is doc


class TestApplyFigureAssetsEdgeCases:
    def test_unsupported_extension(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "notes.md"
        path.write_text("# hi")
        doc = Document(source=str(path), content="hi")
        with caplog.at_level(logging.DEBUG):
            result = apply_figure_assets(
                doc,
                path,
                app_settings=_settings(enabled=True),
                store=LocalAssetStore(tmp_path / "assets"),
            )
        assert result is doc

    def test_unexpected_exception_soft_fails(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        with (
            patch(
                f"{_EXTRACTOR}._apply_docling_figures",
                side_effect=RuntimeError("surprise"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = apply_figure_assets(
                doc,
                path,
                app_settings=_settings(enabled=True),
                store=LocalAssetStore(tmp_path / "assets"),
            )
        assert result is doc
        assert "Unexpected figure asset error" in caplog.text

    def test_pptx_open_failure(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "broken.pptx"
        path.write_bytes(b"not-a-pptx")
        doc = Document(source=str(path), content="")
        with caplog.at_level(logging.WARNING):
            result = apply_figure_assets(
                doc,
                path,
                app_settings=_settings(enabled=True),
                store=LocalAssetStore(tmp_path / "assets"),
            )
        assert result is doc
        assert "Figure asset extraction failed" in caplog.text

    def test_pptx_import_error(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "deck.pptx"
        path.write_bytes(b"x")
        doc = Document(source=str(path), content="")
        with (
            patch(
                f"{_EXTRACTOR}._extract_pptx_picture_bytes",
                side_effect=ConfigurationError("PPTX figure extraction requires python-pptx"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = apply_figure_assets(
                doc,
                path,
                app_settings=_settings(enabled=True),
                store=LocalAssetStore(tmp_path / "assets"),
            )
        assert result is doc
        assert "misconfigured" in caplog.text

    def test_create_picture_converter_import_error(self) -> None:
        from src.rag.ingestion import figure_extractor as mod

        real_import = __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("docling"):
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ConfigurationError, match="docling"),
        ):
            mod._create_picture_converter()

    def test_create_picture_converter_success(self) -> None:
        from src.rag.ingestion import figure_extractor as mod

        fake_converter = object()
        fake_input = SimpleNamespace(PDF="pdf")

        class FakePdfPipelineOptions:
            def __init__(self) -> None:
                self.generate_picture_images = False

        captured: dict[str, Any] = {}

        def fake_converter_cls(*, format_options: Any) -> Any:
            captured["format_options"] = format_options
            return fake_converter

        def fake_format_option(*, pipeline_options: Any) -> Any:
            captured["pipeline_options"] = pipeline_options
            return {"pipeline_options": pipeline_options}

        import sys

        modules = {
            "docling": MagicMock(),
            "docling.datamodel": MagicMock(),
            "docling.datamodel.base_models": SimpleNamespace(InputFormat=fake_input),
            "docling.datamodel.pipeline_options": SimpleNamespace(
                PdfPipelineOptions=FakePdfPipelineOptions
            ),
            "docling.document_converter": SimpleNamespace(
                DocumentConverter=fake_converter_cls,
                PdfFormatOption=fake_format_option,
            ),
        }
        with patch.dict(sys.modules, modules):
            result = mod._create_picture_converter()
        assert result is fake_converter
        assert captured["pipeline_options"].generate_picture_images is True

    def test_extract_pptx_import_error_directly(self, tmp_path: Path) -> None:
        from src.rag.ingestion import figure_extractor as mod

        path = tmp_path / "deck.pptx"
        path.write_bytes(b"x")
        real_import = __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pptx" or name.startswith("pptx."):
                raise ImportError("missing pptx")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ConfigurationError, match="python-pptx"),
        ):
            mod._extract_pptx_picture_bytes(path)

    def test_picture_to_png_empty_buffer(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        picture = MagicMock()
        picture.get_image.return_value = _fake_pil_image(empty=True)
        with caplog.at_level(logging.WARNING):
            result = _apply_with_docling_pictures(doc, path, tmp_path / "assets", [picture])
        assert ASSET_PATH_KEY not in result.metadata["figures"][0]
        assert "No image bytes" in caplog.text

    def test_pptx_without_pictures(self, tmp_path: Path) -> None:
        pptx_path = tmp_path / "text-only.pptx"
        presentation = python_pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text = "Hello"
        presentation.save(str(pptx_path))
        doc = Document(source=str(pptx_path.resolve()), content="Hello")
        result = apply_figure_assets(
            doc,
            pptx_path,
            app_settings=_settings(enabled=True),
            store=LocalAssetStore(tmp_path / "assets"),
        )
        assert result is doc

    def test_docling_assigns_default_figure_id(self, tmp_path: Path) -> None:
        path, doc = _figure_pdf_document(tmp_path, figures=[{"caption": "No id"}])
        picture = MagicMock()
        picture.get_image.return_value = _fake_pil_image()
        result = _apply_with_docling_pictures(doc, path, tmp_path / "assets", [picture])
        assert result.metadata["figures"][0][FIGURE_ID_KEY] == "figure-1"
        assert ASSET_PATH_KEY in result.metadata["figures"][0]

    def test_ext_none_defaults_png(self, tmp_path: Path) -> None:
        from src.rag.ingestion import figure_extractor as mod

        shape = SimpleNamespace(
            shape_type=MSO_SHAPE_TYPE.PICTURE,
            image=SimpleNamespace(blob=_png_bytes(), ext=None),
        )
        presentation = SimpleNamespace(slides=[SimpleNamespace(shapes=[shape])])
        with patch("pptx.Presentation", return_value=presentation):
            pictures = mod._extract_pptx_picture_bytes(tmp_path / "x.pptx")
        assert len(pictures) == 1
        assert pictures[0][1] == "png"
        assert pictures[0][2] == 1

    def test_convert_raises_configuration_error(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path, doc = _figure_pdf_document(tmp_path)
        with caplog.at_level(logging.WARNING):
            result = _apply_with_docling_pictures(
                doc,
                path,
                tmp_path / "assets",
                [],
                convert_error=ConfigurationError("nested config"),
            )
        assert result is doc
        assert "misconfigured" in caplog.text
