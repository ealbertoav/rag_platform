"""T-231 — vision provider factory and OpenAI/Gemini caption clients."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ConfigurationError, GenerationError
from src.core.settings import (
    FigureCaptionSettings,
    GeminiVisionConfig,
    OpenAIVisionConfig,
    ParsingSettings,
    Settings,
)
from src.domain.repositories.vision_repository import VisionRepository
from src.infrastructure.vision import (
    GeminiVisionProvider,
    OpenAIVisionProvider,
    clear_vision_provider_cache,
    get_vision_provider,
)
from src.infrastructure.vision import gemini_vision_provider as gemini_mod
from src.infrastructure.vision import openai_vision_provider as openai_mod


@pytest.fixture(autouse=True)
def _clear_vision_cache() -> Generator[None]:
    clear_vision_provider_cache()
    yield
    clear_vision_provider_cache()


@pytest.fixture(autouse=True)
def no_tenacity_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)


def _caption_settings(
    *,
    enabled: bool = True,
    provider: str = "openai",
    openai_api_key: str = "sk-test",
    openai_model: str = "gpt-4o-mini",
    gemini_api_key: str = "gemini-key",
    gemini_model: str = "gemini-2.0-flash",
) -> Settings:
    return Settings(
        parsing=ParsingSettings(
            figure_captions=FigureCaptionSettings(
                enabled=enabled,
                provider=provider,  # type: ignore[arg-type]
                openai=OpenAIVisionConfig(api_key=openai_api_key, model=openai_model),
                gemini=GeminiVisionConfig(api_key=gemini_api_key, model=gemini_model),
            )
        )
    )


def _png(tmp_path: Path, name: str = "fig.png", data: bytes = b"\x89PNG\r\n") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestGetVisionProvider:
    def test_disabled_returns_none(self) -> None:
        assert get_vision_provider(_caption_settings(enabled=False)) is None

    def test_uses_global_settings_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.infrastructure.vision._settings",
            lambda: _caption_settings(enabled=False),
        )
        assert get_vision_provider() is None

    def test_openai_provider(self) -> None:
        provider = get_vision_provider(_caption_settings(provider="openai"))
        assert isinstance(provider, OpenAIVisionProvider)
        assert provider.model == "gpt-4o-mini"

    def test_gemini_provider(self) -> None:
        provider = get_vision_provider(_caption_settings(provider="gemini"))
        assert isinstance(provider, GeminiVisionProvider)
        assert provider.model == "gemini-2.0-flash"

    def test_caches_same_instance(self) -> None:
        settings = _caption_settings(provider="openai")
        assert get_vision_provider(settings) is get_vision_provider(settings)

    def test_rotating_api_key_rebuilds(self) -> None:
        first = get_vision_provider(_caption_settings(openai_api_key="sk-one"))
        second = get_vision_provider(_caption_settings(openai_api_key="sk-two"))
        assert first is not second

    def test_rotating_gemini_identity_rebuilds(self) -> None:
        first = get_vision_provider(_caption_settings(provider="gemini", gemini_api_key="a"))
        second = get_vision_provider(_caption_settings(provider="gemini", gemini_api_key="b"))
        assert first is not second

    def test_clear_cache(self) -> None:
        settings = _caption_settings()
        first = get_vision_provider(settings)
        clear_vision_provider_cache()
        second = get_vision_provider(settings)
        assert first is not second

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(
            parsing=ParsingSettings(
                figure_captions=FigureCaptionSettings.model_construct(
                    enabled=True,
                    provider="claude",
                    openai=OpenAIVisionConfig(api_key="sk"),
                    gemini=GeminiVisionConfig(api_key="gk"),
                )
            )
        )
        with pytest.raises(ConfigurationError, match="Unknown vision provider"):
            get_vision_provider(settings)

    def test_missing_openai_key_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="openai"):
            get_vision_provider(_caption_settings(openai_api_key=""))

    def test_missing_gemini_key_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="gemini"):
            get_vision_provider(_caption_settings(provider="gemini", gemini_api_key=""))


class TestOpenAIVisionProvider:
    def test_implements_repository(self) -> None:
        assert isinstance(OpenAIVisionProvider(api_key="sk-test"), VisionRepository)

    def test_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.core.settings.settings",
            _caption_settings(openai_api_key="sk-from-settings", openai_model="gpt-4o"),
        )
        provider = OpenAIVisionProvider.from_settings()
        assert provider.api_key == "sk-from-settings"
        assert provider.model == "gpt-4o"

    def test_caption_image_success(self, tmp_path: Path) -> None:
        path = _png(tmp_path, name="photo.jpg")
        provider = OpenAIVisionProvider(api_key="sk-test", model="gpt-4o-mini")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  a chart  "))]
        )
        provider._client = mock_client

        assert provider.caption_image(path) == "a chart"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        content = call_kwargs["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_caption_image_custom_prompt(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = OpenAIVisionProvider(api_key="sk-test")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        provider._client = mock_client

        provider.caption_image(path, prompt="Be brief.")
        content = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert content[0]["text"] == "Be brief."

    def test_empty_asset_raises(self, tmp_path: Path) -> None:
        path = _png(tmp_path, data=b"")
        provider = OpenAIVisionProvider(api_key="sk-test")
        provider._client = MagicMock()
        with pytest.raises(GenerationError, match="empty"):
            provider.caption_image(path)

    def test_non_text_content_raises(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = OpenAIVisionProvider(api_key="sk-test")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        provider._client = mock_client
        with pytest.raises(GenerationError, match="non-text"):
            provider.caption_image(path)

    def test_missing_package_raises(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = OpenAIVisionProvider(api_key="sk-test")
        real_import = __import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "openai" or name.startswith("openai."):
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            pytest.raises(GenerationError, match="openai package"),
        ):
            provider.caption_image(path)

    def test_wraps_unexpected_errors(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = OpenAIVisionProvider(api_key="sk-test")
        with (
            patch.object(provider, "_call_with_retry", side_effect=RuntimeError("net")),
            pytest.raises(GenerationError, match="OpenAI vision caption failed"),
        ):
            provider.caption_image(path)

    def test_reraises_generation_error(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = OpenAIVisionProvider(api_key="sk-test")
        with (
            patch.object(provider, "_call_with_retry", side_effect=GenerationError("already")),
            pytest.raises(GenerationError, match="already"),
        ):
            provider.caption_image(path)

    def test_rate_limit_helper_with_openai_type(self) -> None:
        class RateLimitError(Exception):
            pass

        fake_openai = SimpleNamespace(RateLimitError=RateLimitError)
        with patch.dict("sys.modules", {"openai": fake_openai}):
            # Force re-import path by calling helper; import happens inside function
            assert openai_mod._is_rate_limit(RateLimitError("429")) is True

    def test_rate_limit_helper_without_openai(self) -> None:
        real_import = __import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "openai":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            assert openai_mod._is_rate_limit(Exception("rate limit exceeded")) is True
            assert openai_mod._is_rate_limit(Exception("other")) is False

    def test_mime_type_defaults_to_png(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"data")
        assert openai_mod._mime_type_for(path) == "image/png"

    def test_creates_client_lazily(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = OpenAIVisionProvider(api_key="sk-test")
        mock_openai_cls = MagicMock()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        fake_mod = SimpleNamespace(OpenAI=mock_openai_cls)
        with patch.dict("sys.modules", {"openai": fake_mod}):
            assert provider.caption_image(path) == "ok"
        mock_openai_cls.assert_called_once_with(api_key="sk-test")


class TestGeminiVisionProvider:
    def test_implements_repository(self) -> None:
        assert isinstance(GeminiVisionProvider(api_key="gk"), VisionRepository)

    def test_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.core.settings.settings",
            _caption_settings(
                provider="gemini",
                gemini_api_key="gk-settings",
                gemini_model="gemini-pro",
            ),
        )
        provider = GeminiVisionProvider.from_settings()
        assert provider.api_key == "gk-settings"
        assert provider.model == "gemini-pro"

    def test_caption_image_success(self, tmp_path: Path) -> None:
        path = _png(tmp_path, name="shot.webp")
        provider = GeminiVisionProvider(api_key="gk", model="gemini-2.0-flash")
        with patch.object(provider, "_call_with_retry", return_value="  diagram  "):
            assert provider.caption_image(path) == "diagram"

    def test_caption_image_via_retry(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = GeminiVisionProvider(api_key="gk")
        with patch.object(provider, "_call_api", return_value="via retry"):
            assert provider.caption_image(path) == "via retry"

    def test_call_api_success(self, tmp_path: Path) -> None:
        path = _png(tmp_path, name="shot.webp")
        provider = GeminiVisionProvider(api_key="gk")
        mock_model = MagicMock()
        mock_model.generate_content.return_value = SimpleNamespace(text="diagram")
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            assert provider._call_api(path, "Describe") == "diagram"
        mock_genai.configure.assert_called_once_with(api_key="gk")
        args = mock_model.generate_content.call_args.args[0]
        assert args[0] == "Describe"
        assert args[1]["mime_type"] == "image/webp"

    def test_caption_image_custom_prompt(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = GeminiVisionProvider(api_key="gk")
        with patch.object(provider, "_call_with_retry", return_value="ok") as call:
            provider.caption_image(path, prompt="short")
        call.assert_called_once_with(path, "short")

    def test_empty_asset_raises(self, tmp_path: Path) -> None:
        path = _png(tmp_path, data=b"")
        provider = GeminiVisionProvider(api_key="gk")
        with (
            patch.dict("sys.modules", {"google.generativeai": MagicMock()}),
            pytest.raises(GenerationError, match="empty"),
        ):
            provider._call_api(path, "prompt")

    def test_non_text_content_raises(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = GeminiVisionProvider(api_key="gk")
        mock_model = MagicMock()
        mock_model.generate_content.return_value = SimpleNamespace(text=None)
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        with (
            patch.dict("sys.modules", {"google.generativeai": mock_genai}),
            pytest.raises(GenerationError, match="non-text"),
        ):
            provider._call_api(path, "prompt")

    def test_missing_package_raises(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = GeminiVisionProvider(api_key="gk")
        real_import = __import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in {"google.generativeai", "google"}:
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            pytest.raises(GenerationError, match="google-generativeai"),
        ):
            provider._call_api(path, "prompt")

    def test_wraps_unexpected_errors(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = GeminiVisionProvider(api_key="gk")
        with (
            patch.object(provider, "_call_with_retry", side_effect=RuntimeError("net")),
            pytest.raises(GenerationError, match="Gemini vision caption failed"),
        ):
            provider.caption_image(path)

    def test_reraises_generation_error(self, tmp_path: Path) -> None:
        path = _png(tmp_path)
        provider = GeminiVisionProvider(api_key="gk")
        with (
            patch.object(provider, "_call_with_retry", side_effect=GenerationError("already")),
            pytest.raises(GenerationError, match="already"),
        ):
            provider.caption_image(path)

    def test_rate_limit_helper_with_resource_exhausted(self) -> None:
        class ResourceExhausted(Exception):
            pass

        fake_exc_mod = SimpleNamespace(ResourceExhausted=ResourceExhausted)
        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.api_core": SimpleNamespace(exceptions=fake_exc_mod),
                "google.api_core.exceptions": fake_exc_mod,
            },
        ):
            assert gemini_mod._is_rate_limit(ResourceExhausted("quota")) is True

    def test_rate_limit_helper_without_api_core(self) -> None:
        real_import = __import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "google.api_core.exceptions":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            assert gemini_mod._is_rate_limit(Exception("resource_exhausted")) is True
            assert gemini_mod._is_rate_limit(Exception("other")) is False

    def test_mime_type_defaults_to_png(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"data")
        assert gemini_mod._mime_type_for(path) == "image/png"
