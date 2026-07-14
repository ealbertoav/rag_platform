from __future__ import annotations

from pathlib import Path

from src.core.constants import CHUNK_SECTION_KEY
from src.core.exceptions import DocumentLoadError
from src.core.markdown_headings import extract_markdown_headings
from src.domain.entities.document import Document


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


class MarkdownLoader:
    """Loads raw Markdown files.

    The raw text is preserved (not converted to HTML) because chunkers and LLMs
    handle Markdown natively, and headings serve as natural section boundaries.
    """

    @staticmethod
    def load(path: Path) -> Document:
        try:
            content = _read_text(path).strip()

            headings = extract_markdown_headings(content)

            metadata: dict[str, object] = {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "loader": "markdown",
                "heading_count": len(headings),
                "headings": headings,
            }
            if headings:
                metadata[CHUNK_SECTION_KEY] = headings[0]

            return Document(
                source=str(path.resolve()),
                content=content,
                metadata=metadata,
            )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Cannot load Markdown: {path}", cause=exc) from exc
