# OCR Providers

This platform supports optical character recognition for scanned PDFs and images
during ingestion (Phases 22+). Providers implement
`OcrRepository.ocr(path) -> str` and are selected via `parsing.ocr` in
`configs/parsing.yaml`.

## Feature flag

OCR is **off by default**. Enable it and pick a provider:

```yaml
# configs/parsing.yaml
parsing:
  ocr:
    enabled: true
    provider: tesseract   # tesseract | easyocr | docling | azure_di
    min_chars: 50
```

Or via environment variables:

```bash
PARSING__OCR__ENABLED=true
PARSING__OCR__PROVIDER=tesseract
PARSING__OCR__MIN_CHARS=50
```

When disabled, `get_ocr_provider()` returns `None` and scanned-PDF fallback
(`apply_ocr_fallback`, T-223) is a no-op.

## Self-hosted (T-221)

| Provider | Engine | Notes |
|---|---|---|
| `tesseract` | Docling + Tesseract CLI | Default; install Tesseract on the host |
| `easyocr` | Docling + EasyOCR | Heavier GPU/CPU footprint |
| `docling` | Docling auto engine | Lets Docling pick an available OCR backend |

Install Docling separately:

```bash
uv pip install docling
```

Prefer self-hosted when documents must stay on-prem, cost must stay near zero,
or you already run Tesseract/EasyOCR in the environment.

## Azure Document Intelligence (T-222)

| Provider | Backend |
|---|---|
| `azure_di` | Azure Document Intelligence REST API (`prebuilt-read` by default) |

This uses the cloud **Document Intelligence** analyze API (the Form OCR Tools /
FOTT *backend*), not the labeling UI. Local files are uploaded as
`base64Source`; the client polls `Operation-Location` until the operation
succeeds and returns `analyzeResult.content`.

### Credentials

```yaml
# configs/parsing.yaml (or env overrides)
parsing:
  ocr:
    enabled: true
    provider: azure_di
    azure_di:
      endpoint: https://<resource>.cognitiveservices.azure.com
      api_key: ""                 # prefer env: PARSING__OCR__AZURE_DI__API_KEY
      api_version: "2024-11-30"
      model_id: prebuilt-read
      timeout_seconds: 120
      poll_interval_seconds: 1
```

```bash
PARSING__OCR__ENABLED=true
PARSING__OCR__PROVIDER=azure_di
PARSING__OCR__AZURE_DI__ENDPOINT=https://<resource>.cognitiveservices.azure.com
PARSING__OCR__AZURE_DI__API_KEY=<key>
```

Missing endpoint or API key raises `ConfigurationError` at factory construction.
`apply_ocr_fallback` treats that as a soft failure and keeps extractable text.

### When to prefer Azure DI

- Scanned / low-quality documents where self-hosted OCR accuracy is insufficient
- Multi-language or handwriting-heavy corpora
- You already have an Azure AI services resource and accept cloud egress

Prefer self-hosted when latency, cost, or data residency rules rule out cloud OCR.

## Factory

```python
from src.infrastructure.ocr import get_ocr_provider

provider = get_ocr_provider()  # None when parsing.ocr.enabled=false
if provider is not None:
    text = provider.ocr(path)
```

`get_ocr_provider()` caches by `(enabled, provider, azure_di identity)`.
For `azure_di`, the identity includes endpoint, API key, API version, model
ID, timeout, and poll interval so credential/config rotations rebuild the
client. Call `clear_ocr_provider_cache()` after settings reloads in tests
(also done automatically by `temporary_config` / `reload_settings_module`).

## Ingest wiring

See README — [Scanned-PDF OCR Fallback (T-223)](../README.md#scanned-pdf-ocr-fallback-t-223).
With `azure_di` registered, no further pipeline changes are required: low-text
PDFs call `get_ocr_provider().ocr(path)` the same way as self-hosted engines.
