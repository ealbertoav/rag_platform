"""Azure Document Intelligence OCR provider (T-222).

Calls the Document Intelligence REST API (``prebuilt-read`` by default) via
httpx — not the OCR Form Tools labeling UI. Local files are uploaded as
``base64Source``; results are polled from the ``Operation-Location`` header.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

import httpx

from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.domain.repositories.ocr_repository import OcrRepository

logger = logging.getLogger(__name__)

_OCR_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)
_DEFAULT_API_VERSION = "2024-11-30"
_DEFAULT_MODEL_ID = "prebuilt-read"
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_POLL_INTERVAL_SECONDS = 1.0
_SUBSCRIPTION_KEY_HEADER = "Ocp-Apim-Subscription-Key"

__all__ = ["AzureDocumentIntelligenceOcr"]


class AzureDocumentIntelligenceOcr(OcrRepository):
    """``OcrRepository`` backed by Azure Document Intelligence REST API."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str = _DEFAULT_API_VERSION,
        model_id: str = _DEFAULT_MODEL_ID,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        normalized_endpoint = endpoint.strip().rstrip("/")
        normalized_key = api_key.strip()
        if not normalized_endpoint or not normalized_key:
            raise ConfigurationError(
                "OCR provider 'azure_di' requires parsing.ocr.azure_di.endpoint "
                "and parsing.ocr.azure_di.api_key"
            )
        self.endpoint: str = normalized_endpoint
        self.api_key: str = normalized_key
        self.api_version: str = api_version
        self.model_id: str = model_id
        self.timeout_seconds: float = timeout_seconds
        self.poll_interval_seconds: float = poll_interval_seconds
        self._client: httpx.Client | None = client
        self._owns_client: bool = client is None
        self._sleeper: Callable[[float], None] = sleeper or time.sleep
        self._clock: Callable[[], float] = clock or time.monotonic

    @classmethod
    def from_settings(cls) -> AzureDocumentIntelligenceOcr:
        """Build a provider from ``settings.parsing.ocr.azure_di``."""
        from src.core.settings import settings

        cfg = settings.parsing.ocr.azure_di
        return cls(
            endpoint=cfg.endpoint,
            api_key=cfg.api_key.get_secret_value(),
            api_version=cfg.api_version,
            model_id=cfg.model_id,
            timeout_seconds=cfg.timeout_seconds,
            poll_interval_seconds=cfg.poll_interval_seconds,
        )

    @override
    def ocr(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext not in _OCR_EXTENSIONS:
            raise DocumentLoadError(
                f"OCR provider 'azure_di' does not support '{ext}'. "
                f"Supported: {sorted(_OCR_EXTENSIONS)}"
            )

        try:
            payload = path.read_bytes()
            encoded = base64.b64encode(payload).decode("ascii")
            operation_url = self._start_analyze(encoded)
            text = self._poll_result(operation_url).strip()
            if not text:
                logger.warning("No OCR text extracted from %s", path.name)
            return text
        except DocumentLoadError:
            raise
        except ConfigurationError as exc:
            raise DocumentLoadError(
                f"OCR provider 'azure_di' is not configured for {path.name}",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise DocumentLoadError(f"Cannot OCR with 'azure_di': {path}", cause=exc) from exc

    def close(self) -> None:
        """Close the owned httpx client, if any."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_seconds)
            self._owns_client = True
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            _SUBSCRIPTION_KEY_HEADER: self.api_key,
            "Content-Type": "application/json",
        }

    def _analyze_url(self) -> str:
        return f"{self.endpoint}/documentintelligence/documentModels/{self.model_id}:analyze"

    def _start_analyze(self, base64_source: str) -> str:
        response = self._http().post(
            self._analyze_url(),
            params={"api-version": self.api_version},
            headers=self._headers(),
            json={"base64Source": base64_source},
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DocumentLoadError(
                f"Azure DI analyze request failed with HTTP {exc.response.status_code}",
                cause=exc,
            ) from exc

        # httpx header lookup is case-insensitive.
        operation_url = response.headers.get("Operation-Location")
        if not operation_url:
            raise DocumentLoadError("Azure DI analyze response missing Operation-Location header")
        return str(operation_url)

    def _poll_result(self, operation_url: str) -> str:
        deadline = self._clock() + self.timeout_seconds
        while True:
            response = self._http().get(operation_url, headers=self._headers())
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise DocumentLoadError(
                    f"Azure DI result poll failed with HTTP {exc.response.status_code}",
                    cause=exc,
                ) from exc

            data: dict[str, Any] = response.json()
            status = str(data.get("status", "")).lower()
            if status == "succeeded":
                analyze_result = data.get("analyzeResult")
                if not isinstance(analyze_result, dict):
                    return ""
                content = analyze_result.get("content")
                return str(content) if content is not None else ""
            if status == "failed":
                error = data.get("error") or data.get("analyzeResult") or data
                raise DocumentLoadError(f"Azure DI analysis failed: {error}")

            if self._clock() >= deadline:
                raise DocumentLoadError(
                    f"Azure DI analysis timed out after {self.timeout_seconds:.0f}s"
                )
            self._sleeper(self.poll_interval_seconds)
