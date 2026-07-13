"""Local filesystem asset store for extracted figure bytes (T-230)."""

from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path

from src.core.constants import ASSETS_DIR, ROOT

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def document_asset_key(source: str) -> str:
    """Return a stable, filesystem-safe key for a document source path."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    stem = Path(source).stem or "document"
    safe_stem = _SAFE_SEGMENT_RE.sub("_", stem).strip("._-") or "document"
    return f"{safe_stem}-{digest}"


def sanitize_segment(value: str) -> str:
    """Return a filesystem-safe path segment."""
    cleaned = _SAFE_SEGMENT_RE.sub("_", value).strip("._-")
    return cleaned or "asset"


def resolve_store_root(store_dir: str | Path) -> Path:
    """Resolve *store_dir* relative to the project root when not absolute."""
    path = Path(store_dir)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


class LocalAssetStore:
    """Persist binary assets under "{root}/{document_key}/{figure_id}.{ext}"."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root: Path = resolve_store_root(root) if root is not None else ASSETS_DIR.resolve()

    def path_for(
        self,
        document_key: str,
        figure_id: str,
        *,
        extension: str = "png",
    ) -> Path:
        """Return the absolute path for a figure asset without writing."""
        ext = extension.lstrip(".").lower() or "png"
        return self.root / sanitize_segment(document_key) / f"{sanitize_segment(figure_id)}.{ext}"

    def save(
        self,
        document_key: str,
        figure_id: str,
        data: bytes,
        *,
        extension: str = "png",
    ) -> Path:
        """Write *data* to the disk and return the absolute asset path.

        When an existing file at the destination is replaced with different
        bytes, any adjacent "{stem}.caption.txt" sidecar is removed so T-231
        captioning cannot reuse a stale VLM caption for the new image.
        """
        if not data:
            raise ValueError("Cannot store empty asset bytes")
        path = self.path_for(document_key, figure_id, extension=extension)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            try:
                previous = path.read_bytes()
            except OSError:
                previous = None
            if previous is not None and previous != data:
                sidecar = path.with_name(f"{path.stem}.caption.txt")
                with contextlib.suppress(OSError):
                    sidecar.unlink(missing_ok=True)
        path.write_bytes(data)
        return path
