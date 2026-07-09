from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pptx as python_pptx
from pptx.shapes.base import BaseShape
from pptx.slide import Slide
from pptx.text.text import TextFrame

from src.core.constants import CHUNK_SECTION_KEY
from src.core.exceptions import DocumentLoadError
from src.domain.entities.document import Document


class _TextFrameShape(Protocol):
    has_text_frame: bool
    text_frame: TextFrame


def _shape_text(shape: BaseShape) -> str:
    if not shape.has_text_frame:
        return ""
    text_frame = cast(_TextFrameShape, shape).text_frame
    paragraphs = [p.text for p in text_frame.paragraphs]
    return "\n".join(p for p in paragraphs if p.strip())


def _slide_title(slide: Slide) -> str | None:
    title_shape = slide.shapes.title
    if title_shape is None:
        return None
    title = _shape_text(title_shape).strip()
    return title or None


def _slide_text(slide: Slide) -> str:
    parts: list[str] = []
    for shape in slide.shapes:
        text = _shape_text(shape)
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


class PptxLoader:
    """Loads text from PPTX files using python-pptx."""

    @staticmethod
    def load(path: Path) -> Document:
        try:
            presentation = python_pptx.Presentation(str(path))

            slide_texts: list[str] = []
            section_titles: list[str] = []
            for slide in presentation.slides:
                title = _slide_title(slide)
                if title:
                    section_titles.append(title)
                text = _slide_text(slide)
                if text.strip():
                    slide_texts.append(text)

            content = "\n\n---\n\n".join(slide_texts)

            metadata: dict[str, object] = {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "loader": "pptx",
                "slide_count": len(presentation.slides),
                "sections": section_titles,
            }
            if section_titles:
                metadata[CHUNK_SECTION_KEY] = section_titles[0]

            return Document(
                source=str(path.resolve()),
                content=content,
                metadata=metadata,
            )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Cannot load PPTX: {path}", cause=exc) from exc
