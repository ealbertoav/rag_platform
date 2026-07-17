"""T-280 — Multimodal golden dataset builder: QA pairs scoped to table/figure chunks.

Reuses the LLM-driven QA generation and dedup logic from
:class:`~src.evals.golden_dataset.SyntheticDatasetBuilder` (T-040), restricted
to chunks whose resolved modality is table or figure, and tags each resulting
pair with that modality so downstream modality-specific metrics (T-281) can
filter by it. Written as JSON Lines rather than the T-040 JSON array to keep
the multimodal golden appendable/streamable independently of the main QA
golden.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from src.core.constants import CHUNK_TYPE_KEY, MODALITY_FIGURE, MODALITY_TABLE
from src.domain.entities.chunk import Chunk
from src.domain.entities.source_reference import resolve_modality
from src.evals.golden_dataset import QAPair, SyntheticDatasetBuilder

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_PATH = (
    Path(__file__).parents[2] / "datasets" / "goldens" / "multimodal_qa_dataset.jsonl"
)

MULTIMODAL_MODALITIES: frozenset[str] = frozenset({MODALITY_TABLE, MODALITY_FIGURE})


@dataclasses.dataclass
class MultimodalQAPair:
    """A QAPair (T-040) tagged with its source chunk's table/figure modality."""

    question: str
    answer: str
    relevant_chunks: list[str]
    modality: str
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


def chunk_modality(chunk: Chunk) -> str:
    """Resolve *chunk*'s effective modality (explicit field, else metadata.type)."""
    return resolve_modality(modality=chunk.modality, chunk_type=chunk.metadata.get(CHUNK_TYPE_KEY))


def filter_multimodal_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Return only chunks whose resolved modality is table or figure."""
    return [chunk for chunk in chunks if chunk_modality(chunk) in MULTIMODAL_MODALITIES]


def build_multimodal_golden(
    builder: SyntheticDatasetBuilder,
    chunks: Iterable[Chunk],
) -> list[MultimodalQAPair]:
    """Generate QA pairs from *chunks*, restricted to table/figure modalities."""
    multimodal_chunks = filter_multimodal_chunks(chunks)
    if not multimodal_chunks:
        return []

    pairs = builder.generate_from_chunks(multimodal_chunks)
    chunks_by_id = {chunk.id: chunk for chunk in multimodal_chunks}
    return _tag_modality(pairs, chunks_by_id)


def _tag_modality(
    pairs: list[QAPair],
    chunks_by_id: dict[str, Chunk],
) -> list[MultimodalQAPair]:
    """Attach each pair's source chunk modality; drop pairs with no resolvable source."""
    tagged: list[MultimodalQAPair] = []
    for pair in pairs:
        chunk = next(
            (chunks_by_id[cid] for cid in pair.relevant_chunks if cid in chunks_by_id),
            None,
        )
        if chunk is None:
            logger.debug(
                "Dropping QA pair with no resolvable multimodal source chunk: %r",
                pair.question,
            )
            continue
        tagged.append(
            MultimodalQAPair(
                question=pair.question,
                answer=pair.answer,
                relevant_chunks=pair.relevant_chunks,
                modality=chunk_modality(chunk),
                source=pair.source,
            )
        )
    return tagged


def save_jsonl(pairs: list[MultimodalQAPair], path: Path | None = None) -> None:
    """Write *pairs* as JSON Lines (default: datasets/goldens/multimodal_qa_dataset.jsonl)."""
    target = path or _DEFAULT_OUTPUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False))
            fh.write("\n")
    logger.info("Saved %d multimodal QA pairs to %s", len(pairs), target)


def load_jsonl(path: Path | None = None) -> list[dict[str, object]]:
    """Load multimodal golden rows from a JSONL file; returns [] when missing."""
    source = path or _DEFAULT_OUTPUT_PATH
    if not source.exists():
        return []

    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if isinstance(row, dict):
                rows.append(row)
    return rows
