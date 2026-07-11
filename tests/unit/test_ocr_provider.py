"""T-220 / T-221 / T-222 — OCR provider factory and provider tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.core.settings import Settings
from src.infrastructure import ocr as ocr_module
from src.infrastructure.ocr import (
    AzureDocumentIntelligenceOcr,
    DoclingOcrProvider,
    EasyOcrProvider,
    TesseractOcrProvider,
    clear_ocr_provider_cache,
    get_ocr_provider,
)
from src.infrastructure.ocr import docling_backed as backed_module
from src.infrastructure.ocr.docling_backed import DoclingBackedOcr, create_ocr_converter

_BACKED = "src.infrastructure.ocr.docling_backed"
_ENDPOINT = "https://example.cognitiveservices.azure.com"
_API_KEY = "test-key"


@pytest.fixture(autouse=True)
def _clear_ocr_cache() -> Generator[None]:
    clear_ocr_provider_cache()
    yield
    clear_ocr_provider_cache()


def _ocr_settings(
    *,
    enabled: bool = False,
    provider: str = "tesseract",
    azure_endpoint: str = "",
    azure_api_key: str = "",
) -> Settings:
    return Settings(
        parsing={
            "ocr": {
                "enabled": enabled,
                "provider": provider,
                "azure_di": {
                    "endpoint": azure_endpoint,
                    "api_key": azure_api_key,
                },
            }
        }
    )


def _azure_settings(*, enabled: bool = True) -> Settings:
    return _ocr_settings(
        enabled=enabled,
        provider="azure_di",
        azure_endpoint=_ENDPOINT,
        azure_api_key=_API_KEY,
    )


def _make_result(
    *,
    markdown: str = "hello",
    status_name: str | None = "SUCCESS",
) -> SimpleNamespace:
    status = None if status_name is None else SimpleNamespace(name=status_name)
    document = SimpleNamespace(export_to_markdown=lambda: markdown)
    return SimpleNamespace(status=status, document=document)


def _make_converter(result: SimpleNamespace | None = None) -> MagicMock:
    converter = MagicMock()
    converter.convert.return_value = result or _make_result()
    return converter


class TestGetOcrProvider:
    def test_returns_none_when_disabled(self) -> None:
        assert get_ocr_provider(_ocr_settings(enabled=False)) is None

    def test_returns_none_when_disabled_without_explicit_settings(self) -> None:
        disabled = _ocr_settings(enabled=False)
        with patch("src.infrastructure.ocr._settings", return_value=disabled):
            assert get_ocr_provider() is None

    def test_reads_live_settings_when_not_passed(self) -> None:
        from src.evals.e2e.technique_benchmark import temporary_config

        with temporary_config({"PARSING__OCR__ENABLED": "false"}):
            assert get_ocr_provider() is None

    def test_cache_returns_none_for_same_disabled_settings(self) -> None:
        settings = _ocr_settings(enabled=False, provider="tesseract")
        first = get_ocr_provider(settings)
        second = get_ocr_provider(settings)
        assert first is None
        assert second is None

    def test_clear_ocr_provider_cache_allows_reload(self) -> None:
        settings = _ocr_settings(enabled=False)
        assert get_ocr_provider(settings) is None
        clear_ocr_provider_cache()
        assert get_ocr_provider(settings) is None

    @pytest.mark.parametrize(
        ("provider", "cls"),
        [
            ("tesseract", TesseractOcrProvider),
            ("easyocr", EasyOcrProvider),
            ("docling", DoclingOcrProvider),
        ],
    )
    def test_returns_self_hosted_provider(self, provider: str, cls: type[DoclingBackedOcr]) -> None:
        settings = _ocr_settings(enabled=True, provider=provider)
        result = get_ocr_provider(settings)
        assert isinstance(result, cls)
        assert result.engine == provider

    def test_caches_enabled_provider_instance(self) -> None:
        settings = _ocr_settings(enabled=True, provider="tesseract")
        first = get_ocr_provider(settings)
        second = get_ocr_provider(settings)
        assert first is second

    def test_returns_azure_di_provider(self) -> None:
        result = get_ocr_provider(_azure_settings())
        assert isinstance(result, AzureDocumentIntelligenceOcr)
        assert result.endpoint == _ENDPOINT
        assert result.api_key == _API_KEY

    def test_azure_di_missing_credentials_raises(self) -> None:
        settings = _ocr_settings(enabled=True, provider="azure_di")
        with pytest.raises(ConfigurationError, match="requires parsing.ocr.azure_di"):
            get_ocr_provider(settings)

    def test_unknown_provider_raises_configuration_error(self) -> None:
        settings = _ocr_settings(enabled=True, provider="unknown")
        with pytest.raises(ConfigurationError, match="Unknown OCR provider"):
            get_ocr_provider(settings)

    def test_failed_enabled_lookup_does_not_poison_disabled_cache(self) -> None:
        enabled = _ocr_settings(enabled=True, provider="azure_di")
        with pytest.raises(ConfigurationError):
            get_ocr_provider(enabled)
        assert get_ocr_provider(_ocr_settings(enabled=False)) is None

    def test_module_exports(self) -> None:
        assert set(ocr_module.__all__) == {
            "AzureDocumentIntelligenceOcr",
            "DoclingOcrProvider",
            "EasyOcrProvider",
            "TesseractOcrProvider",
            "clear_ocr_provider_cache",
            "get_ocr_provider",
        }


class TestDoclingBackedOcr:
    def test_ocr_returns_text(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.png"
        path.write_bytes(b"fake-image")
        provider = TesseractOcrProvider(converter=_make_converter(_make_result(markdown="  hi  ")))
        assert provider.ocr(path) == "hi"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.docx"
        path.write_text("x")
        with pytest.raises(DocumentLoadError, match="does not support"):
            EasyOcrProvider(converter=_make_converter()).ocr(path)

    def test_empty_text_returns_empty_string(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_path / "blank.pdf"
        path.write_bytes(b"%PDF-1.4")
        provider = DoclingOcrProvider(converter=_make_converter(_make_result(markdown="   ")))
        with caplog.at_level("WARNING"):
            assert provider.ocr(path) == ""
        assert "No OCR text extracted" in caplog.text

    def test_failure_status_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.4")
        converter = _make_converter(_make_result(status_name="FAILURE"))
        with pytest.raises(DocumentLoadError, match="OCR conversion failed"):
            TesseractOcrProvider(converter=converter).ocr(path)

    def test_status_none_treated_as_success(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.jpg"
        path.write_bytes(b"img")
        provider = EasyOcrProvider(
            converter=_make_converter(_make_result(markdown="ok", status_name=None))
        )
        assert provider.ocr(path) == "ok"

    def test_re_raises_document_load_error(self, tmp_path: Path) -> None:
        path = tmp_path / "err.pdf"
        path.write_bytes(b"%PDF-1.4")
        converter = MagicMock()
        converter.convert.side_effect = DocumentLoadError("already wrapped")
        with pytest.raises(DocumentLoadError, match="already wrapped"):
            DoclingOcrProvider(converter=converter).ocr(path)

    def test_wraps_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.png"
        path.write_bytes(b"img")
        with (
            patch(f"{_BACKED}.create_ocr_converter", side_effect=ConfigurationError("missing")),
            pytest.raises(DocumentLoadError, match="not configured") as exc_info,
        ):
            TesseractOcrProvider().ocr(path)
        assert isinstance(exc_info.value.cause, ConfigurationError)

    def test_wraps_unexpected_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "boom.webp"
        path.write_bytes(b"img")
        converter = MagicMock()
        converter.convert.side_effect = RuntimeError("boom")
        with pytest.raises(DocumentLoadError, match="Cannot OCR") as exc_info:
            EasyOcrProvider(converter=converter).ocr(path)
        assert exc_info.value.cause is not None

    def test_lazy_converter_reused(self, tmp_path: Path) -> None:
        path = tmp_path / "reuse.tif"
        path.write_bytes(b"img")
        converter = _make_converter(_make_result(markdown="once"))
        with patch(f"{_BACKED}.create_ocr_converter", return_value=converter) as create:
            provider = DoclingOcrProvider()
            assert provider.ocr(path) == "once"
            assert provider.ocr(path) == "once"
            create.assert_called_once_with("docling")


class TestCreateOcrConverter:
    def test_builds_converter_with_engine_options(self) -> None:
        mock_converter_cls = MagicMock(return_value="converter-instance")
        mock_pdf_fmt = MagicMock()
        mock_img_fmt = MagicMock()
        mock_pipeline = MagicMock()
        mock_input = SimpleNamespace(PDF="pdf", IMAGE="image")
        mock_tesseract = MagicMock(return_value="tess-opts")
        mock_easy = MagicMock(return_value="easy-opts")
        mock_auto = MagicMock(return_value="auto-opts")

        pipeline_mod = MagicMock()
        pipeline_mod.PdfPipelineOptions = mock_pipeline
        pipeline_mod.TesseractCliOcrOptions = mock_tesseract
        pipeline_mod.EasyOcrOptions = mock_easy
        pipeline_mod.OcrAutoOptions = mock_auto

        doc_converter_mod = MagicMock()
        doc_converter_mod.DocumentConverter = mock_converter_cls
        doc_converter_mod.PdfFormatOption = mock_pdf_fmt
        doc_converter_mod.ImageFormatOption = mock_img_fmt

        base_mod = MagicMock()
        base_mod.InputFormat = mock_input

        with patch.dict(
            "sys.modules",
            {
                "docling": MagicMock(),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.base_models": base_mod,
                "docling.datamodel.pipeline_options": pipeline_mod,
                "docling.document_converter": doc_converter_mod,
            },
        ):
            assert create_ocr_converter("tesseract") == "converter-instance"
            assert create_ocr_converter("easyocr") == "converter-instance"
            assert create_ocr_converter("docling") == "converter-instance"

        assert mock_tesseract.call_count == 1
        assert mock_easy.call_count == 1
        assert mock_auto.call_count == 1
        assert mock_converter_cls.call_count == 3

    def test_unknown_engine_raises(self) -> None:
        pipeline_mod = MagicMock()
        pipeline_mod.PdfPipelineOptions = MagicMock()
        pipeline_mod.TesseractCliOcrOptions = MagicMock()
        pipeline_mod.EasyOcrOptions = MagicMock()
        pipeline_mod.OcrAutoOptions = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "docling": MagicMock(),
                    "docling.datamodel": MagicMock(),
                    "docling.datamodel.base_models": MagicMock(),
                    "docling.datamodel.pipeline_options": pipeline_mod,
                    "docling.document_converter": MagicMock(),
                },
            ),
            pytest.raises(ConfigurationError, match="Unknown Docling OCR engine"),
        ):
            create_ocr_converter("nope")

    def test_missing_docling_raises_configuration_error(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("docling"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_blocked),
            pytest.raises(ConfigurationError, match="uv pip install docling"),
        ):
            create_ocr_converter("tesseract")

    def test_ocr_options_for_engine_unknown(self) -> None:
        pipeline_mod = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "docling": MagicMock(),
                    "docling.datamodel": MagicMock(),
                    "docling.datamodel.pipeline_options": pipeline_mod,
                },
            ),
            pytest.raises(ConfigurationError, match="Unknown Docling OCR engine"),
        ):
            backed_module._ocr_options_for_engine("nope")


def _azure_provider(
    *,
    client: httpx.Client | MagicMock | None = None,
    sleeper: MagicMock | None = None,
    clock: MagicMock | None = None,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 1.0,
) -> AzureDocumentIntelligenceOcr:
    return AzureDocumentIntelligenceOcr(
        endpoint=_ENDPOINT,
        api_key=_API_KEY,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        client=client,  # type: ignore[arg-type]
        sleeper=sleeper,
        clock=clock,
    )


def _http_response(
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    json_data: dict[str, Any] | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test")
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        json=json_data,
        request=request,
    )


class TestAzureDocumentIntelligenceOcr:
    def test_init_requires_credentials(self) -> None:
        with pytest.raises(ConfigurationError, match="requires parsing.ocr.azure_di"):
            AzureDocumentIntelligenceOcr(endpoint="", api_key="")
        with pytest.raises(ConfigurationError, match="requires parsing.ocr.azure_di"):
            AzureDocumentIntelligenceOcr(endpoint=_ENDPOINT, api_key="  ")

    def test_from_settings_uses_global_config(self) -> None:
        mock_settings = _azure_settings()
        with patch("src.core.settings.settings", mock_settings):
            provider = AzureDocumentIntelligenceOcr.from_settings()
        assert provider.endpoint == _ENDPOINT
        assert provider.api_key == _API_KEY
        assert provider.model_id == "prebuilt-read"

    def test_ocr_happy_path(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/1"},
        )
        client.get.return_value = _http_response(
            json_data={
                "status": "succeeded",
                "analyzeResult": {"content": "  hello azure  "},
            }
        )
        provider = _azure_provider(client=client)
        assert provider.ocr(path) == "hello azure"
        client.post.assert_called_once()
        body = client.post.call_args.kwargs["json"]
        assert "base64Source" in body

    def test_ocr_polls_until_succeeded(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.png"
        path.write_bytes(b"img")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/2"},
        )
        client.get.side_effect = [
            _http_response(json_data={"status": "running"}),
            _http_response(json_data={"status": "succeeded", "analyzeResult": {"content": "done"}}),
        ]
        sleeper = MagicMock()
        clock = MagicMock(side_effect=[0.0, 0.5, 1.0])
        provider = _azure_provider(client=client, sleeper=sleeper, clock=clock)
        assert provider.ocr(path) == "done"
        sleeper.assert_called_once_with(1.0)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.docx"
        path.write_text("x")
        with pytest.raises(DocumentLoadError, match="does not support"):
            _azure_provider(client=MagicMock()).ocr(path)

    def test_empty_content_returns_empty_string(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "blank.pdf"
        path.write_bytes(b"%PDF")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/3"},
        )
        client.get.return_value = _http_response(
            json_data={"status": "succeeded", "analyzeResult": {"content": "   "}}
        )
        with caplog.at_level("WARNING"):
            assert _azure_provider(client=client).ocr(path) == ""
        assert "No OCR text extracted" in caplog.text

    def test_missing_analyze_result_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.jpg"
        path.write_bytes(b"img")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/4"},
        )
        client.get.return_value = _http_response(json_data={"status": "succeeded"})
        assert _azure_provider(client=client).ocr(path) == ""

    def test_failed_status_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/5"},
        )
        client.get.return_value = _http_response(
            json_data={"status": "failed", "error": {"message": "boom"}}
        )
        with pytest.raises(DocumentLoadError, match="Azure DI analysis failed"):
            _azure_provider(client=client).ocr(path)

    def test_missing_operation_location_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF")
        client = MagicMock()
        client.post.return_value = _http_response(status_code=202, headers={})
        with pytest.raises(DocumentLoadError, match="Operation-Location"):
            _azure_provider(client=client).ocr(path)

    def test_analyze_http_error(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF")
        client = MagicMock()
        client.post.return_value = _http_response(status_code=401, json_data={"error": "nope"})
        with pytest.raises(DocumentLoadError, match="HTTP 401") as exc_info:
            _azure_provider(client=client).ocr(path)
        assert exc_info.value.cause is not None

    def test_poll_http_error(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/6"},
        )
        client.get.return_value = _http_response(status_code=500, json_data={"error": "x"})
        with pytest.raises(DocumentLoadError, match="HTTP 500"):
            _azure_provider(client=client).ocr(path)

    def test_timeout_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/7"},
        )
        client.get.return_value = _http_response(json_data={"status": "running"})
        sleeper = MagicMock()
        clock = MagicMock(side_effect=[0.0, 5.0])
        provider = _azure_provider(client=client, sleeper=sleeper, clock=clock, timeout_seconds=5.0)
        with pytest.raises(DocumentLoadError, match="timed out"):
            provider.ocr(path)

    def test_wraps_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.pdf"
        path.write_bytes(b"%PDF")
        provider = _azure_provider(client=MagicMock())
        with (
            patch.object(
                provider,
                "_start_analyze",
                side_effect=ConfigurationError("creds"),
            ),
            pytest.raises(DocumentLoadError, match="not configured") as exc_info,
        ):
            provider.ocr(path)
        assert isinstance(exc_info.value.cause, ConfigurationError)

    def test_wraps_unexpected_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.webp"
        path.write_bytes(b"img")
        client = MagicMock()
        client.post.side_effect = RuntimeError("network down")
        with pytest.raises(DocumentLoadError, match="Cannot OCR") as exc_info:
            _azure_provider(client=client).ocr(path)
        assert exc_info.value.cause is not None

    def test_lazy_client_created_and_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.tif"
        path.write_bytes(b"img")
        mock_client = MagicMock()
        mock_client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/8"},
        )
        mock_client.get.return_value = _http_response(
            json_data={"status": "succeeded", "analyzeResult": {"content": "x"}}
        )
        with patch("httpx.Client", return_value=mock_client) as client_cls:
            provider = AzureDocumentIntelligenceOcr(endpoint=_ENDPOINT, api_key=_API_KEY)
            assert provider.ocr(path) == "x"
            client_cls.assert_called_once()
            provider.close()
            mock_client.close.assert_called_once()
            provider.close()  # second close is a no-op

    def test_injected_client_not_closed(self) -> None:
        client = MagicMock()
        provider = _azure_provider(client=client)
        provider.close()
        client.close.assert_not_called()

    def test_null_content_field(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.bmp"
        path.write_bytes(b"img")
        client = MagicMock()
        client.post.return_value = _http_response(
            status_code=202,
            headers={"Operation-Location": f"{_ENDPOINT}/ops/9"},
        )
        client.get.return_value = _http_response(
            json_data={"status": "succeeded", "analyzeResult": {"content": None}}
        )
        assert _azure_provider(client=client).ocr(path) == ""
