from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import Any

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_SOURCE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.recursive_chunker import RecursiveChunker

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "ingestion" / "extract_propositions.txt"

_GRADE_PROMPT = """Evaluate the proposition below against the original text.

Rate each category from 1 to 10:
- accuracy: how well the proposition reflects the original text
- clarity: how easy it is to understand without additional context
- completeness: whether necessary details (dates, qualifiers) are included
- conciseness: whether the proposition is concise without losing information

Original text:
{original_text}

Proposition:
{proposition}

Output ONLY valid JSON with integer fields: accuracy, clarity, completeness, conciseness.
"""


def load_extract_template(path: Path | None = None) -> Template:
    """Load the proposition-extraction prompt from the disk."""
    template_path = path or _PROMPT_PATH
    return Template(template_path.read_text(encoding="utf-8").strip())


def extract_propositions(
    text: str,
    llm: LLMRepository,
    template: Template | None = None,
) -> list[str]:
    """Return atomic propositions extracted from *text* using the LLM."""
    tmpl = template or load_extract_template()
    prompt = tmpl.substitute(text=text.strip())
    response = llm.generate(prompt=prompt, context="").strip()
    return _parse_propositions(response)


def grade_proposition(
    proposition: str,
    original_text: str,
    llm: LLMRepository,
) -> dict[str, int] | None:
    """Return quality scores for *proposition* or None when grading fails."""
    prompt = _GRADE_PROMPT.format(
        original_text=original_text.strip(),
        proposition=proposition.strip(),
    )
    response = llm.generate(prompt=prompt, context="").strip()
    return _parse_scores(response)


def passes_quality_threshold(scores: dict[str, int], threshold: int) -> bool:
    """True when every score category meets a *threshold*."""
    required = ("accuracy", "clarity", "completeness", "conciseness")
    return all(scores.get(key, 0) >= threshold for key in required)


class PropositionChunker:
    """Extracts and quality-filters atomic factual propositions via LLM calls.

    Documents are first split into processing segments (recursive chunker) that
    are never indexed directly. Each segment is decomposed into propositions;
    low-scoring propositions are discarded before indexing.
    """

    def __init__(
        self,
        llm: LLMRepository,
        chunk_size: int = 500,
        overlap: int = 50,
        quality_threshold: int = 7,
        template: Template | None = None,
    ) -> None:
        if not 1 <= quality_threshold <= 10:
            raise ValueError("quality_threshold must be between 1 and 10")
        self._llm = llm
        self._quality_threshold = quality_threshold
        self._template = template
        self._segmenter = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        segments = self._segmenter.chunk(document)
        if not segments:
            return []

        chunks: list[Chunk] = []
        proposition_index = 0

        for segment in segments:
            try:
                propositions = extract_propositions(segment.text, self._llm, self._template)
            except Exception as exc:
                logger.warning(
                    "Proposition extraction failed for segment %s in %s: %s",
                    segment.metadata.get(CHUNK_INDEX_KEY),
                    document.source,
                    exc,
                )
                continue

            for proposition in propositions:
                text = proposition.strip()
                if not text:
                    continue

                try:
                    scores = grade_proposition(text, segment.text, self._llm)
                except Exception as exc:
                    logger.warning(
                        "Proposition grading failed in %s: %s",
                        document.source,
                        exc,
                    )
                    continue
                if scores is None:
                    logger.debug("Skipping ungradable proposition in %s", document.source)
                    continue
                if not passes_quality_threshold(scores, self._quality_threshold):
                    continue

                metadata = {
                    **document.metadata,
                    CHUNK_SOURCE_KEY: document.source,
                    CHUNK_INDEX_KEY: proposition_index,
                    "proposition_scores": scores,
                }
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=text,
                        metadata=metadata,
                    )
                )
                proposition_index += 1

        return chunks


def _parse_propositions(text: str) -> list[str]:
    parsed = _load_json_list(text)
    if parsed is not None:
        return parsed

    logger.warning("Could not parse propositions from LLM response")
    return []


def _parse_scores(text: str) -> dict[str, int] | None:
    for candidate in (text.strip(), _extract_json_object(text)):
        if not candidate:
            continue
        try:
            data: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        scores = _normalise_scores(data)
        if scores is not None:
            return scores
    logger.warning("Could not parse proposition quality scores from LLM response")
    return None


def _load_json_list(text: str) -> list[str] | None:
    for candidate in (text.strip(), _extract_json_array(text)):
        if not candidate:
            continue
        try:
            data: Any = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        propositions = [item.strip() for item in data if isinstance(item, str) and item.strip()]
        if propositions:
            return propositions
    return None


def _normalise_scores(data: Any) -> dict[str, int] | None:
    if not isinstance(data, dict):
        return None

    keys = ("accuracy", "clarity", "completeness", "conciseness")
    scores: dict[str, int] = {}
    for key in keys:
        value = _coerce_score(data.get(key))
        if value is None:
            return None
        scores[key] = value
    return scores


def _coerce_score(value: Any) -> int | None:
    """Convert LLM score values to integers (accepts whole-number floats)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _extract_json_array(text: str) -> str | None:
    match = re.search(r"\[.*?]", text, re.DOTALL)
    return match.group() if match else None


def _extract_json_object(text: str) -> str | None:
    match = re.search(r"\{.*}", text, re.DOTALL)
    return match.group() if match else None
