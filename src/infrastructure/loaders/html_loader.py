from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from src.core.exceptions import DocumentLoadError
from src.domain.entities.document import Document

# Tags whose content is navigational/decorative and should be stripped.
_STRIP_TAGS = frozenset({
    "script", "style", "nav", "header", "footer", "aside",
    "noscript", "iframe", "svg", "form",
})


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


class HtmlLoader:
    """Loads text from HTML files, stripping boilerplate tags using BeautifulSoup."""

    @staticmethod
    def load(path: Path) -> Document:
        try:
            raw = _read_text(path)
            soup = BeautifulSoup(raw, "html.parser")

            for tag in soup.find_all(_STRIP_TAGS):
                if isinstance(tag, Tag):
                    tag.decompose()

            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            content = soup.get_text(separator="\n", strip=True)

            return Document(
                source=str(path.resolve()),
                content=content,
                metadata={
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "loader": "html",
                    "title": title,
                },
            )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Cannot load HTML: {path}", cause=exc) from exc
