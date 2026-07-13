"""Figure asset extraction and Chunk builders (T-230).

When "parsing.figure_assets.enabled" is true, persist figure bytes from
layout "figures[]" (Docling PDF/DOCX) or PPTX picture shapes under a
:class:`~src.rag.ingestion.local_asset_store.LocalAssetStore`, then set
"figures[].asset_path" and produce :class:`~src.domain.entities.chunk.Chunk`
instances with "asset_path" / "figure_id".

Docling picture export enables "generate_picture_images" for PDF
(:class:`~docling.document_converter.PdfFormatOption`) and DOCX
(:class:`~docling.document_converter.WordFormatOption` +
"PaginatedPipelineOptions"). DOCX also falls back to embedded
"python-docx" image parts when Docling conversion fails, Docling is
unavailable, or "PictureItem.get_image()" returns no bytes (common when
layout metadata exists but ImageRef was not attached). Figure slot N uses
Docling picture N with DOCX embedded image N as backfill so assets stay
aligned in document order.

Soft-fails per figure when bytes cannot be exported. VLM captions are
T-231; caption indexing is T-232.
"""

from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Any, Protocol, cast

from src.core.constants import (
    ASSET_PATH_KEY,
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_KEY,
    FIGURE_ID_KEY,
    MODALITY_FIGURE,
)
from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.core.settings import FigureAssetSettings, Settings
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.metadata import chunk_metadata
from src.rag.ingestion.local_asset_store import LocalAssetStore, document_asset_key

logger = logging.getLogger(__name__)

_PPTX_EXTENSION = ".pptx"
_DOCLING_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})
_FIGURE_CHUNK_NAMESPACE = uuid.UUID("c4e91b7a-2d6f-4a18-9e3b-8f0d5c1a7b42")
_DEFAULT_FIGURE_TEXT = "[figure]"


class _PilImage(Protocol):
    def save(self, fp: Any, *args: Any, **kwargs: Any) -> None: ...


class _PptxImage(Protocol):
    blob: bytes
    ext: str | None


class _PictureShape(Protocol):
    image: _PptxImage


class _DoclingPicture(Protocol):
    prov: list[Any]

    def caption_text(self, doc: object) -> str: ...

    def get_image(self, doc: object) -> _PilImage | None: ...


class _DoclingDocument(Protocol):
    pictures: list[_DoclingPicture]


class _DoclingConversionStatus(Protocol):
    name: str


class _DoclingConversionResult(Protocol):
    status: _DoclingConversionStatus | None
    document: _DoclingDocument


class _DoclingConverter(Protocol):
    def convert(self, source: str) -> _DoclingConversionResult: ...


def figure_chunk_id(source: str, figure_id: str) -> str:
    """Stable chunk ID for a figure asset at a *source*."""
    return str(uuid.uuid5(_FIGURE_CHUNK_NAMESPACE, f"{source}:{figure_id}"))


def is_figure_chunk(chunk: Chunk) -> bool:
    """Return True when *chunk* is a figure asset index point."""
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_FIGURE


def build_figure_chunks(document: Document) -> list[Chunk]:
    """Build figure chunks from document metadata that already has asset paths."""
    figures = document.metadata.get("figures")
    if not isinstance(figures, list) or not figures:
        return []

    chunks: list[Chunk] = []
    for index, raw_entry in enumerate(figures):
        if not isinstance(raw_entry, dict):
            logger.debug("Skipping non-dict figure entry at index %d", index)
            continue
        figure_id = raw_entry.get(FIGURE_ID_KEY)
        asset_path = raw_entry.get(ASSET_PATH_KEY) or raw_entry.get("asset_path")
        if not figure_id or not asset_path:
            continue

        caption = raw_entry.get("caption")
        if isinstance(caption, str) and caption.strip():
            text = caption.strip()
        else:
            text = _DEFAULT_FIGURE_TEXT

        metadata = chunk_metadata(document.metadata)
        metadata[CHUNK_TYPE_KEY] = CHUNK_TYPE_FIGURE
        metadata[FIGURE_ID_KEY] = str(figure_id)
        metadata[ASSET_PATH_KEY] = str(asset_path)
        metadata[CHUNK_SOURCE_KEY] = document.source
        if CHUNK_PAGE_KEY in raw_entry:
            metadata[CHUNK_PAGE_KEY] = raw_entry[CHUNK_PAGE_KEY]
        if BBOX_KEY in raw_entry:
            metadata[BBOX_KEY] = raw_entry[BBOX_KEY]

        chunks.append(
            Chunk(
                id=figure_chunk_id(document.source, str(figure_id)),
                document_id=document.id,
                text=text,
                metadata=metadata,
                modality=MODALITY_FIGURE,
                asset_path=str(asset_path),
            )
        )
    return chunks


def apply_figure_assets(
    document: Document,
    path: Path,
    *,
    app_settings: Settings | None = None,
    store: LocalAssetStore | None = None,
) -> Document:
    """Persist figure bytes and attach "asset_path" onto "figures[]" metadata.

    No-op when figure assets are disabled. Soft-fails (logs and returns the
    original document) on configuration / conversion errors, so ingest continues.
    """
    cfg = _figure_asset_settings(app_settings)
    if not cfg.enabled:
        return document

    asset_store = store or LocalAssetStore(cfg.store_dir)
    try:
        suffix = path.suffix.lower()
        if suffix == _PPTX_EXTENSION:
            return _apply_pptx_figures(document, path, asset_store)
        if suffix in _DOCLING_EXTENSIONS:
            return _apply_docling_figures(document, path, asset_store)
        logger.debug("Skipping figure assets for unsupported type %s", path.name)
        return document
    except ConfigurationError as exc:
        logger.warning(
            "Figure asset extraction misconfigured for %s: %s — continuing without assets",
            path.name,
            exc,
        )
        return document
    except DocumentLoadError as exc:
        logger.warning(
            "Figure asset extraction failed for %s: %s — continuing without assets",
            path.name,
            exc,
        )
        return document
    except Exception as exc:
        logger.warning(
            "Unexpected figure asset error for %s: %s — continuing without assets",
            path.name,
            exc,
        )
        return document


def _figure_asset_settings(app_settings: Settings | None) -> FigureAssetSettings:
    if app_settings is None:
        from src.core.settings import settings as default_settings

        return default_settings.parsing.figure_assets
    return app_settings.parsing.figure_assets


def _figure_entry_needs_asset(entry: dict[str, Any]) -> bool:
    """Return True when a figures[] dict has an id but no persisted asset_path."""
    if not entry.get(FIGURE_ID_KEY):
        return False
    asset_path = entry.get(ASSET_PATH_KEY) or entry.get("asset_path")
    return not asset_path


def _apply_pptx_figures(
    document: Document,
    path: Path,
    store: LocalAssetStore,
) -> Document:
    extracted = _extract_pptx_picture_bytes(path)
    existing = document.metadata.get("figures")
    if not extracted:
        pending = 0
        if isinstance(existing, list):
            pending = sum(
                1
                for entry in existing
                if isinstance(entry, dict) and _figure_entry_needs_asset(entry)
            )
        if pending:
            logger.warning(
                "PPTX %s has %d figure(s) without asset_path but no extractable pictures — "
                "continuing without assets",
                path.name,
                pending,
            )
        return document

    doc_key = document_asset_key(document.source)
    figures: list[dict[str, Any]] = []
    existing_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(existing, list):
        for entry in existing:
            if isinstance(entry, dict) and entry.get(FIGURE_ID_KEY):
                existing_by_id[str(entry[FIGURE_ID_KEY])] = dict(entry)

    for index, (blob, extension, page) in enumerate(extracted, start=1):
        figure_id = f"figure-{index}"
        entry = dict(existing_by_id.get(figure_id, {FIGURE_ID_KEY: figure_id}))
        entry[FIGURE_ID_KEY] = figure_id
        entry[CHUNK_PAGE_KEY] = page
        try:
            asset_path = store.save(doc_key, figure_id, blob, extension=extension)
            entry[ASSET_PATH_KEY] = str(asset_path)
        except Exception as exc:
            logger.warning(
                "Failed to store PPTX figure %s from %s: %s",
                figure_id,
                path.name,
                exc,
            )
        figures.append(entry)

    metadata = dict(document.metadata)
    metadata["figures"] = figures
    return document.model_copy(update={"metadata": metadata})


def _extract_pptx_picture_bytes(path: Path) -> list[tuple[bytes, str, int]]:
    try:
        import pptx as python_pptx
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError as exc:
        raise ConfigurationError(
            "PPTX figure extraction requires python-pptx. Install with: uv pip install python-pptx"
        ) from exc

    try:
        presentation = python_pptx.Presentation(str(path))
    except Exception as exc:
        raise DocumentLoadError(
            f"Cannot open PPTX for figure extraction: {path}",
            cause=exc,
        ) from exc

    pictures: list[tuple[bytes, str, int]] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        for shape in slide.shapes:
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            try:
                image = cast(_PictureShape, shape).image
                blob = image.blob
                extension = (image.ext or "png").lstrip(".").lower() or "png"
            except Exception as exc:
                logger.warning(
                    "Skipping unreadable PPTX picture on slide %s in %s: %s",
                    slide_index,
                    path.name,
                    exc,
                )
                continue
            if blob:
                pictures.append((blob, extension, slide_index))
    return pictures


def _apply_docling_figures(
    document: Document,
    path: Path,
    store: LocalAssetStore,
) -> Document:
    existing = document.metadata.get("figures")
    if not isinstance(existing, list) or not existing:
        logger.debug("No layout figures[] for %s — skipping Docling asset export", path.name)
        return document

    figure_entries = [entry for entry in existing if isinstance(entry, dict)]
    expected_count = len(figure_entries)
    pictures = _load_docling_pictures_or_empty(path)
    docx_blobs = _docx_fallback_blobs(path)

    if not pictures and not docx_blobs:
        logger.warning(
            "Layout figures[] has %d entr%s for %s but Docling returned no pictures — "
            "continuing without asset_path",
            expected_count,
            "y" if expected_count == 1 else "ies",
            path.name,
        )
        return document

    # Docling pictures are primary; DOCX embedded parts only backfill missing rasters.
    # Both lists are document-order aligned: figure slot N uses pictures[N] / docx_blobs[N].
    available = max(len(pictures), len(docx_blobs)) if docx_blobs else len(pictures)
    if expected_count > available:
        if docx_blobs and not pictures:
            logger.warning(
                "DOCX embedded images returned %d picture(s) for %s but layout figures[] "
                "has %d dict entr%s — unmatched figures leave without asset_path",
                len(docx_blobs),
                path.name,
                expected_count,
                "y" if expected_count == 1 else "ies",
            )
        else:
            logger.warning(
                "Docling returned %d picture(s) for %s but layout figures[] has %d dict entr%s — "
                "unmatched figures leave without asset_path",
                len(pictures),
                path.name,
                expected_count,
                "y" if expected_count == 1 else "ies",
            )

    doc_key = document_asset_key(document.source)
    figures: list[dict[str, Any]] = []
    slot = 0
    for raw_entry in existing:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        figure_id = str(entry.get(FIGURE_ID_KEY) or f"figure-{len(figures) + 1}")
        entry[FIGURE_ID_KEY] = figure_id

        blob: bytes | None = None
        extension = "png"
        attempted_docling = False
        if slot < len(pictures):
            attempted_docling = True
            picture, doc = pictures[slot]
            try:
                blob = _picture_to_png_bytes(picture, doc)
            except Exception as exc:
                logger.warning(
                    "Failed to export Docling figure %s from %s: %s",
                    figure_id,
                    path.name,
                    exc,
                )

        if not blob and slot < len(docx_blobs):
            blob, extension = docx_blobs[slot]
            logger.debug(
                "Using python-docx embedded image for %s in %s",
                figure_id,
                path.name,
            )

        slot += 1

        if not blob:
            if attempted_docling:
                logger.warning(
                    "No image bytes for %s in %s — leaving without asset_path",
                    figure_id,
                    path.name,
                )
            else:
                logger.warning(
                    "No Docling picture for %s in %s — leaving without asset_path",
                    figure_id,
                    path.name,
                )
            figures.append(entry)
            continue

        try:
            asset_path = store.save(doc_key, figure_id, blob, extension=extension)
            entry[ASSET_PATH_KEY] = str(asset_path)
        except Exception as exc:
            logger.warning(
                "Failed to export Docling figure %s from %s: %s",
                figure_id,
                path.name,
                exc,
            )
        figures.append(entry)

    metadata = dict(document.metadata)
    metadata["figures"] = figures
    return document.model_copy(update={"metadata": metadata})


def _docx_fallback_blobs(path: Path) -> list[tuple[bytes, str]]:
    """Return embedded DOCX image blobs for figure-asset fallback, else []."""
    if path.suffix.lower() != ".docx":
        return []
    try:
        return _extract_docx_picture_bytes(path)
    except ConfigurationError:
        raise
    except DocumentLoadError as exc:
        logger.warning(
            "DOCX embedded-image fallback unavailable for %s: %s",
            path.name,
            exc,
        )
        return []
    except Exception as exc:
        logger.warning(
            "DOCX embedded-image fallback failed for %s: %s",
            path.name,
            exc,
        )
        return []


def _extension_from_image_content_type(content_type: str) -> str:
    normalized = (content_type or "").lower().strip()
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "image/webp": "webp",
        "image/x-emf": "emf",
        "image/x-wmf": "wmf",
    }
    if normalized in mapping:
        return mapping[normalized]
    if "/" in normalized:
        subtype = normalized.rsplit("/", 1)[-1]
        if subtype.startswith("x-"):
            subtype = subtype[2:]
        if subtype:
            return subtype
    return "png"


def _extract_docx_picture_bytes(path: Path) -> list[tuple[bytes, str]]:
    """Extract embedded image parts from a DOCX via python-docx (document order)."""
    try:
        import docx as python_docx
    except ImportError as exc:
        raise ConfigurationError(
            "DOCX figure fallback requires python-docx. Install with: uv pip install python-docx"
        ) from exc

    try:
        document = python_docx.Document(str(path))
    except Exception as exc:
        raise DocumentLoadError(
            f"Cannot open DOCX for figure extraction: {path}",
            cause=exc,
        ) from exc

    pictures: list[tuple[bytes, str]] = []
    seen_rids: set[str] = set()
    for rel in document.part.rels.values():
        reltype = str(getattr(rel, "reltype", "") or "")
        if "image" not in reltype.lower():
            continue
        rid = str(getattr(rel, "rId", "") or "")
        if rid and rid in seen_rids:
            continue
        try:
            target = rel.target_part
            blob = bytes(getattr(target, "blob", b"") or b"")
            content_type = str(getattr(target, "content_type", "") or "")
            extension = _extension_from_image_content_type(content_type)
        except Exception as exc:
            logger.warning(
                "Skipping unreadable DOCX embedded image in %s: %s",
                path.name,
                exc,
            )
            continue
        if not blob:
            continue
        if rid:
            seen_rids.add(rid)
        pictures.append((blob, extension))
    return pictures


def _load_docling_pictures_or_empty(
    path: Path,
) -> list[tuple[_DoclingPicture, _DoclingDocument]]:
    """Load Docling pictures, soft-failing so DOCX embedded-image fallback can run.

    "DocumentLoadError" always soft-fails to an empty list. "ConfigurationError"
    (e.g. missing Docling) soft-fails only for ".docx", where python-docx can still
    supply bytes; for other types it re-raises so ingest reports misconfiguration.
    """
    try:
        return _load_docling_pictures(path)
    except DocumentLoadError as exc:
        logger.warning(
            "Docling figure conversion failed for %s: %s — "
            "trying embedded-image fallback if available",
            path.name,
            exc,
        )
        return []
    except ConfigurationError:
        if path.suffix.lower() != ".docx":
            raise
        logger.warning(
            "Docling unavailable for %s — trying python-docx embedded-image fallback",
            path.name,
        )
        return []


def _load_docling_pictures(path: Path) -> list[tuple[_DoclingPicture, _DoclingDocument]]:
    converter = _create_picture_converter()
    try:
        result = converter.convert(str(path))
    except ConfigurationError:
        raise
    except Exception as exc:
        raise DocumentLoadError(
            f"Cannot convert {path.name} for figure asset extraction",
            cause=exc,
        ) from exc

    status = getattr(result, "status", None)
    if status is not None and getattr(status, "name", str(status)) == "FAILURE":
        raise DocumentLoadError(f"Docling conversion failed for figure assets: {path}")

    doc = result.document
    pictures = getattr(doc, "pictures", None) or []
    return [(picture, doc) for picture in pictures]


def _picture_to_png_bytes(picture: _DoclingPicture, doc: _DoclingDocument) -> bytes | None:
    image = picture.get_image(doc)
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    return data or None


def _create_picture_converter() -> _DoclingConverter:
    try:
        # Optional runtime dependency (same pattern as DoclingLayoutParser / OCR).
        # Install separately: uv pip install docling
        import importlib

        base_models = importlib.import_module("docling.datamodel.base_models")
        pipeline_options_mod = importlib.import_module("docling.datamodel.pipeline_options")
        document_converter_mod = importlib.import_module("docling.document_converter")
        input_format = base_models.InputFormat
        pdf_pipeline_options = pipeline_options_mod.PdfPipelineOptions
        paginated_pipeline_options = pipeline_options_mod.PaginatedPipelineOptions
        document_converter = document_converter_mod.DocumentConverter
        pdf_format_option = document_converter_mod.PdfFormatOption
        word_format_option = document_converter_mod.WordFormatOption
    except ImportError as exc:
        raise ConfigurationError(
            "Figure asset extraction for PDF/DOCX requires the docling package. "
            "Install with: uv pip install docling"
        ) from exc

    pdf_options = pdf_pipeline_options()
    pdf_options.generate_picture_images = True
    # DOCX uses SimplePipeline; PaginatedPipelineOptions carries generate_picture_images
    # so PictureItem ImageRefs / get_image() can yield raster bytes when supported.
    docx_options = paginated_pipeline_options()
    docx_options.generate_picture_images = True
    return cast(
        _DoclingConverter,
        document_converter(
            allowed_formats=[input_format.PDF, input_format.DOCX],
            format_options={
                input_format.PDF: pdf_format_option(pipeline_options=pdf_options),
                input_format.DOCX: word_format_option(pipeline_options=docx_options),
            },
        ),
    )
