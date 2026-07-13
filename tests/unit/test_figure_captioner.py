"""T-231 — VLM figure captioning at ingest."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import ASSET_PATH_KEY, FIGURE_CAPTION_KEY, FIGURE_ID_KEY
from src.core.exceptions import ConfigurationError, GenerationError
from src.core.settings import (
    FigureCaptionSettings,
    OpenAIVisionConfig,
    ParsingSettings,
    Settings,
)
from src.domain.entities.document import Document
from src.rag.ingestion.figure_captioner import (
    apply_figure_captions,
    caption_sidecar_path,
)
from src.rag.pipelines.ingestion_pipeline import content_hash
from tests.unit.ingestion_helpers import mock_ingestion_pipeline

_INGEST_MOD = "src.rag.pipelines.ingestion_pipeline"


def _settings(
    *,
    enabled: bool = True,
    provider: str = "openai",
    openai_api_key: str = "sk-test",
) -> Settings:
    captions = FigureCaptionSettings(
        enabled=enabled,
        provider=provider,  # type: ignore[arg-type]
        openai=OpenAIVisionConfig(api_key=openai_api_key),
    )
    return Settings(parsing=ParsingSettings(figure_captions=captions))


def _document(figures: list[dict[str, Any]] | list[Any] | None) -> Document:
    metadata: dict[str, Any] = {}
    if figures is not None:
        metadata["figures"] = figures
    return Document(id="doc-1", source="report.pdf", content="body", metadata=metadata)


def _asset(tmp_path: Path, name: str = "figure-1.png", data: bytes = b"png-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestCaptionSidecarPath:
    def test_uses_stem_caption_txt(self, tmp_path: Path) -> None:
        asset = tmp_path / "figure-1.png"
        assert caption_sidecar_path(asset) == tmp_path / "figure-1.caption.txt"


class TestApplyFigureCaptionsDisabled:
    def test_disabled_is_noop(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document(
            [{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset), FIGURE_CAPTION_KEY: "old"}]
        )
        result = apply_figure_captions(doc, app_settings=_settings(enabled=False))
        assert result is doc
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "old"

    def test_no_figures_is_noop(self) -> None:
        doc = _document(None)
        result = apply_figure_captions(doc, app_settings=_settings(enabled=True))
        assert result is doc

    def test_empty_figures_is_noop(self) -> None:
        doc = _document([])
        result = apply_figure_captions(doc, app_settings=_settings(enabled=True))
        assert result is doc


class TestApplyFigureCaptionsSuccess:
    def test_writes_caption_for_asset(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()
        provider.caption_image.return_value = "  A bar chart of revenue  "

        result = apply_figure_captions(
            doc,
            app_settings=_settings(enabled=True),
            vision_provider=provider,
        )

        assert result is not doc
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "A bar chart of revenue"
        provider.caption_image.assert_called_once_with(asset)
        assert caption_sidecar_path(asset).read_text(encoding="utf-8") == "A bar chart of revenue"

    def test_overwrites_existing_caption(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document(
            [
                {
                    FIGURE_ID_KEY: "figure-1",
                    ASSET_PATH_KEY: str(asset),
                    FIGURE_CAPTION_KEY: "docling caption",
                }
            ]
        )
        provider = MagicMock()
        provider.caption_image.return_value = "vlm caption"

        result = apply_figure_captions(doc, vision_provider=provider)
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "vlm caption"
        assert caption_sidecar_path(asset).read_text(encoding="utf-8") == "vlm caption"

    def test_accepts_asset_path_alias(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", "asset_path": str(asset)}])
        provider = MagicMock()
        provider.caption_image.return_value = "caption"

        result = apply_figure_captions(doc, vision_provider=provider)
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "caption"

    def test_skips_entries_without_asset_path(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document(
            [
                {FIGURE_ID_KEY: "figure-1"},
                {FIGURE_ID_KEY: "figure-2", ASSET_PATH_KEY: str(asset)},
            ]
        )
        provider = MagicMock()
        provider.caption_image.return_value = "only second"

        result = apply_figure_captions(doc, vision_provider=provider)
        assert FIGURE_CAPTION_KEY not in result.metadata["figures"][0]
        assert result.metadata["figures"][1][FIGURE_CAPTION_KEY] == "only second"
        provider.caption_image.assert_called_once()

    def test_preserves_non_dict_entries(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document(["bad", {FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()
        provider.caption_image.return_value = "ok"

        result = apply_figure_captions(doc, vision_provider=provider)
        assert result.metadata["figures"][0] == "bad"
        assert result.metadata["figures"][1][FIGURE_CAPTION_KEY] == "ok"


class TestApplyFigureCaptionsSidecar:
    def test_loads_sidecar_without_calling_vlm(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        caption_sidecar_path(asset).write_text("cached caption", encoding="utf-8")
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()

        result = apply_figure_captions(doc, vision_provider=provider)

        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "cached caption"
        provider.caption_image.assert_not_called()

    def test_matching_sidecar_is_noop(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        caption_sidecar_path(asset).write_text("same", encoding="utf-8")
        doc = _document(
            [
                {
                    FIGURE_ID_KEY: "figure-1",
                    ASSET_PATH_KEY: str(asset),
                    FIGURE_CAPTION_KEY: "same",
                }
            ]
        )
        provider = MagicMock()

        result = apply_figure_captions(doc, vision_provider=provider)

        assert result is doc
        provider.caption_image.assert_not_called()

    def test_empty_sidecar_falls_through_to_vlm(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        caption_sidecar_path(asset).write_text("   \n", encoding="utf-8")
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()
        provider.caption_image.return_value = "fresh"

        with caplog.at_level("WARNING"):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "fresh"
        assert "sidecar empty" in caplog.text
        assert caption_sidecar_path(asset).read_text(encoding="utf-8") == "fresh"

    def test_unreadable_sidecar_falls_through_to_vlm(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        sidecar = caption_sidecar_path(asset)
        sidecar.write_text("stale", encoding="utf-8")
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()
        provider.caption_image.return_value = "recovered"

        with (
            patch.object(Path, "read_text", side_effect=OSError("permission denied")),
            caplog.at_level("WARNING"),
        ):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "recovered"
        assert "sidecar unreadable" in caplog.text
        provider.caption_image.assert_called_once_with(asset)

    def test_sidecar_write_failure_still_updates_metadata(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()
        provider.caption_image.return_value = "in memory only"

        with (
            patch.object(Path, "write_text", side_effect=OSError("disk full")),
            caplog.at_level("WARNING"),
        ):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "in memory only"
        assert "sidecar write failed" in caplog.text


class TestApplyFigureCaptionsSoftFail:
    def test_missing_asset_keeps_entry(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing = tmp_path / "missing.png"
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(missing)}])
        provider = MagicMock()

        with caplog.at_level("WARNING"):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result is doc
        provider.caption_image.assert_not_called()
        assert "asset missing" in caplog.text

    def test_generation_error_keeps_existing_caption(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        doc = _document(
            [
                {
                    FIGURE_ID_KEY: "figure-1",
                    ASSET_PATH_KEY: str(asset),
                    FIGURE_CAPTION_KEY: "keep me",
                }
            ]
        )
        provider = MagicMock()
        provider.caption_image.side_effect = GenerationError("boom")

        with caplog.at_level("WARNING"):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result is doc
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "keep me"
        assert "Figure caption failed" in caplog.text

    def test_unexpected_error_keeps_entry(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        provider = MagicMock()
        provider.caption_image.side_effect = RuntimeError("boom")

        with caplog.at_level("WARNING"):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result is doc
        assert "Unexpected figure caption error" in caplog.text

    def test_empty_caption_keeps_existing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        doc = _document(
            [
                {
                    FIGURE_ID_KEY: "figure-1",
                    ASSET_PATH_KEY: str(asset),
                    FIGURE_CAPTION_KEY: "docling",
                }
            ]
        )
        provider = MagicMock()
        provider.caption_image.return_value = "   "

        with caplog.at_level("WARNING"):
            result = apply_figure_captions(doc, vision_provider=provider)

        assert result is doc
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "docling"
        assert "Figure caption empty" in caplog.text
        assert not caption_sidecar_path(asset).exists()

    def test_partial_success_updates_only_ok_figures(self, tmp_path: Path) -> None:
        ok = _asset(tmp_path, "ok.png")
        bad = _asset(tmp_path, "bad.png")
        doc = _document(
            [
                {FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(ok)},
                {FIGURE_ID_KEY: "figure-2", ASSET_PATH_KEY: str(bad)},
            ]
        )
        provider = MagicMock()
        provider.caption_image.side_effect = ["good caption", GenerationError("nope")]

        result = apply_figure_captions(doc, vision_provider=provider)
        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "good caption"
        assert FIGURE_CAPTION_KEY not in result.metadata["figures"][1]
        assert caption_sidecar_path(ok).read_text(encoding="utf-8") == "good caption"
        assert not caption_sidecar_path(bad).exists()

    def test_configuration_error_from_factory_is_soft(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])

        with (
            patch(
                "src.infrastructure.vision.get_vision_provider",
                side_effect=ConfigurationError("missing key"),
            ),
            caplog.at_level("WARNING"),
        ):
            result = apply_figure_captions(doc, app_settings=_settings(enabled=True))

        assert result is doc
        assert "misconfigured" in caplog.text

    def test_factory_none_returns_document(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])

        with patch("src.infrastructure.vision.get_vision_provider", return_value=None):
            result = apply_figure_captions(doc, app_settings=_settings(enabled=True))

        assert result is doc

    def test_misconfigured_provider_still_loads_sidecar(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        caption_sidecar_path(asset).write_text("from disk", encoding="utf-8")
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])

        with patch(
            "src.infrastructure.vision.get_vision_provider",
            side_effect=ConfigurationError("missing key"),
        ) as factory:
            result = apply_figure_captions(doc, app_settings=_settings(enabled=True))

        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "from disk"
        factory.assert_not_called()

    def test_factory_none_with_sidecar_hydrates_caption(self, tmp_path: Path) -> None:
        asset = _asset(tmp_path)
        caption_sidecar_path(asset).write_text("cached", encoding="utf-8")
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])

        with patch("src.infrastructure.vision.get_vision_provider", return_value=None) as factory:
            result = apply_figure_captions(doc, app_settings=_settings(enabled=True))

        assert result.metadata["figures"][0][FIGURE_CAPTION_KEY] == "cached"
        factory.assert_not_called()

    def test_uses_default_settings_when_app_settings_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asset = _asset(tmp_path)
        doc = _document([{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}])
        disabled = Settings(
            parsing=ParsingSettings(figure_captions=FigureCaptionSettings(enabled=False))
        )
        monkeypatch.setattr("src.core.settings.settings", disabled)

        result = apply_figure_captions(doc)
        assert result is doc

    def test_missing_figure_id_uses_index_in_logs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing = tmp_path / "gone.png"
        doc = _document([{ASSET_PATH_KEY: str(missing)}])
        provider = MagicMock()

        with caplog.at_level("WARNING"):
            apply_figure_captions(doc, vision_provider=provider)

        assert "index-0" in caplog.text


class TestIngestionPipelineCaptionWiring:
    def test_full_ingest_calls_captions_after_assets(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        path.write_text("hello figures")
        source = str(path.resolve())
        loaded = Document(id="doc-1", source=source, content="hello figures", metadata={})
        after_assets = loaded.model_copy(
            update={"metadata": {"figures": [{FIGURE_ID_KEY: "figure-1"}]}}
        )
        after_captions = after_assets.model_copy(
            update={
                "metadata": {
                    "figures": [
                        {FIGURE_ID_KEY: "figure-1", FIGURE_CAPTION_KEY: "captioned"},
                    ]
                }
            }
        )
        pipeline, service, *_ = mock_ingestion_pipeline()

        with (
            patch(f"{_INGEST_MOD}.load_document", return_value=loaded),
            patch(f"{_INGEST_MOD}.should_attempt_ocr", return_value=False),
            patch(f"{_INGEST_MOD}.apply_ocr_fallback", return_value=loaded),
            patch(f"{_INGEST_MOD}.apply_figure_assets", return_value=after_assets) as assets_fn,
            patch(
                f"{_INGEST_MOD}.apply_figure_captions", return_value=after_captions
            ) as captions_fn,
        ):
            pipeline.ingest_file(path)

        assets_fn.assert_called_once()
        captions_fn.assert_called_once_with(after_assets)
        service.prepare.assert_called_once_with(after_captions)

    def test_skip_path_calls_captions_after_assets(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.md"
        content = "unchanged content"
        path.write_text(content)
        source = str(path.resolve())
        loaded = Document(id="doc-1", source=source, content=content, metadata={})
        after_assets = loaded.model_copy(update={"metadata": {"figures": []}})
        after_captions = after_assets.model_copy(
            update={"metadata": {"figures": [{FIGURE_CAPTION_KEY: "x"}]}}
        )
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=content_hash(source, content),
            chunk_count=1,
        )
        metadata.get_chunk_ids.return_value = ["c1"]
        pipeline, service, *_ = mock_ingestion_pipeline(metadata=metadata)

        with (
            patch(f"{_INGEST_MOD}.load_document", return_value=loaded),
            patch(f"{_INGEST_MOD}.should_attempt_ocr", return_value=False),
            patch(f"{_INGEST_MOD}.apply_figure_assets", return_value=after_assets) as assets_fn,
            patch(
                f"{_INGEST_MOD}.apply_figure_captions", return_value=after_captions
            ) as captions_fn,
        ):
            result = pipeline.ingest_file(path)

        assert result.skipped is True
        assets_fn.assert_called_once()
        captions_fn.assert_called_once_with(after_assets)
        service.prepare.assert_not_called()

    def test_skip_path_persists_captions_via_sidecar(self, tmp_path: Path) -> None:
        """Skip path must durable-store VLM captions even without reindex."""
        path = tmp_path / "doc.md"
        content = "unchanged content"
        path.write_text(content)
        source = str(path.resolve())
        asset = _asset(tmp_path, "fig.png")
        loaded = Document(
            id="doc-1",
            source=source,
            content=content,
            metadata={
                "figures": [{FIGURE_ID_KEY: "figure-1", ASSET_PATH_KEY: str(asset)}],
            },
        )
        metadata = MagicMock()
        metadata.get_by_source.return_value = MagicMock(
            id="doc-1",
            content_hash=content_hash(source, content),
            chunk_count=1,
        )
        metadata.get_chunk_ids.return_value = ["c1"]
        pipeline, service, *_ = mock_ingestion_pipeline(metadata=metadata)
        provider = MagicMock()
        provider.caption_image.return_value = "skip-path caption"

        with (
            patch(f"{_INGEST_MOD}.load_document", return_value=loaded),
            patch(f"{_INGEST_MOD}.should_attempt_ocr", return_value=False),
            patch(f"{_INGEST_MOD}.apply_figure_assets", side_effect=lambda doc, *_a, **_k: doc),
            patch(
                "src.infrastructure.vision.get_vision_provider",
                return_value=provider,
            ),
            patch("src.core.settings.settings", _settings(enabled=True)),
        ):
            first = pipeline.ingest_file(path)
            second = pipeline.ingest_file(path)

        assert first.skipped is True
        assert second.skipped is True
        assert caption_sidecar_path(asset).read_text(encoding="utf-8") == "skip-path caption"
        assert provider.caption_image.call_count == 1
        service.prepare.assert_not_called()
