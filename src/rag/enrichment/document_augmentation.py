from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from string import Template

from src.core.constants import (
    CHUNK_RAW_TEXT_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_SYNTHETIC,
    SOURCE_CHUNK_ID_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.repositories.vector_store_repository import SearchResult
from src.rag.chunking.contextual_headers import chunk_context_text

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "ingestion" / "generate_chunk_questions.txt"


def load_question_template(path: Path | None = None) -> Template:
    """Load the synthetic-question generation prompt from disk."""
    template_path = path or _PROMPT_PATH
    return Template(template_path.read_text(encoding="utf-8").strip())


def is_synthetic_question(chunk: Chunk) -> bool:
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_SYNTHETIC


def make_question_chunk(source: Chunk, question: str) -> Chunk:
    """Build an indexable question chunk pointing at a *source*."""
    metadata = {
        k: v for k, v in source.metadata.items() if k not in {CHUNK_RAW_TEXT_KEY, CHUNK_TYPE_KEY}
    }
    metadata[CHUNK_TYPE_KEY] = CHUNK_TYPE_SYNTHETIC
    metadata[SOURCE_CHUNK_ID_KEY] = source.id
    return Chunk(document_id=source.document_id, text=question, metadata=metadata)


def generate_questions(
    chunk: Chunk,
    llm: LLMRepository,
    n: int,
    template: Template | None = None,
) -> list[str]:
    """Return up to *n* synthetic questions for *chunk* using the LLM."""
    tmpl = template or load_question_template()
    passage = chunk_context_text(chunk)
    prompt = tmpl.substitute(n=n, passage=passage)
    response = llm.generate(prompt=prompt, context="").strip()
    return _parse_questions(response, n)


def resolve_synthetic_questions(
    results: list[SearchResult],
    lookup: Callable[[str], Chunk | None],
) -> list[SearchResult]:
    """Map synthetic question hits back to their source chunks for fusion."""
    resolved: list[SearchResult] = []
    for chunk, score in results:
        if not is_synthetic_question(chunk):
            resolved.append((chunk, score))
            continue

        source_id = chunk.metadata.get(SOURCE_CHUNK_ID_KEY)
        if not isinstance(source_id, str):
            logger.debug("Synthetic question %s missing source_chunk_id", chunk.id)
            continue

        source = lookup(source_id)
        if source is None:
            logger.debug("Source chunk %s not found for synthetic question %s", source_id, chunk.id)
            continue

        resolved.append((source, score))
    return resolved


class DocumentAugmentor:
    """Generates and embeds synthetic question chunks at ingested time."""

    def __init__(
        self,
        llm: LLMRepository,
        embedder: EmbeddingRepository,
        n_questions: int = 3,
        template: Template | None = None,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._n_questions = n_questions
        self._template = template

    def augment(self, source_chunks: list[Chunk]) -> list[Chunk]:
        """Return embedded synthetic question chunks for *source_chunks*."""
        question_chunks: list[Chunk] = []
        for chunk in source_chunks:
            try:
                questions = generate_questions(chunk, self._llm, self._n_questions, self._template)
            except Exception as exc:
                logger.warning("Augmentation failed for chunk %s: %s", chunk.id, exc)
                continue

            for question in questions[: self._n_questions]:
                if question.strip():
                    question_chunks.append(make_question_chunk(chunk, question.strip()))

        if not question_chunks:
            return []

        texts = [c.text for c in question_chunks]
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both(texts)
        except Exception as exc:
            logger.warning("Embedding augmented questions failed: %s", exc)
            return []

        return [
            chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
            for chunk, dense, sparse in zip(question_chunks, dense_vecs, sparse_vecs, strict=True)
        ]


def _parse_questions(text: str, max_count: int) -> list[str]:
    """Extract question strings from LLM JSON output."""
    parsed = _load_question_json(text)
    if parsed is not None:
        return parsed[:max_count]

    logger.warning("Could not parse synthetic questions from LLM response")
    return []


def _load_question_json(text: str) -> list[str] | None:
    for candidate in (text.strip(), _extract_json_array(text)):
        if not candidate:
            continue
        try:
            data: object = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        questions = _normalise_questions(data)
        if questions:
            return questions
    return None


def _extract_json_array(text: str) -> str | None:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    return match.group() if match else None


def _normalise_questions(data: object) -> list[str]:
    if not isinstance(data, list):
        return []

    questions: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            questions.append(item.strip())
        elif isinstance(item, dict):
            question = item.get("question")
            if isinstance(question, str) and question.strip():
                questions.append(question.strip())
    return questions
