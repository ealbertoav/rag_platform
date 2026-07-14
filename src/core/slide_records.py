"""Shared PPTX slide-record type (loader output, section-chunker input)."""

from __future__ import annotations

from typing import NamedTuple


class SlideRecord(NamedTuple):
    """A single PPTX slide's loader-authoritative title and body text."""

    title: str | None
    text: str
