from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from string import Template
from typing import TypedDict

import numpy as np

from src.domain.entities.chunk import Chunk
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "evaluation" / "generate_qa.txt"


class _RawPair(TypedDict):
    """Expected JSON structure returned by the LLM for each QA pair."""

    question: str
    answer: str


@dataclasses.dataclass
class QAPair:
    question: str
    answer: str
    relevant_chunks: list[str]
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)  # type: ignore[return-value]


class SyntheticDatasetBuilder:
    """Generate synthetic QA pairs from ingested chunks via an LLM.

    Each chunk is sent to the LLM, which returns N (question, answer) pairs.
    Questions that are too similar (cosine similarity ≥ "dedup_threshold")
    to an earlier question are discarded, keeping the dataset diverse.
    """

    # Class-level annotations so basedpyright can type-check subclasses.
    _llm: LLMRepository
    _embedder: EmbeddingRepository
    _n: int
    _dedup_threshold: float
    _template: Template | None

    def __init__(
        self,
        llm: LLMRepository,
        embedder: EmbeddingRepository,
        n_pairs_per_chunk: int = 3,
        dedup_threshold: float = 0.95,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._n = n_pairs_per_chunk
        self._dedup_threshold = dedup_threshold
        self._template = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate_from_chunks(self, chunks: list[Chunk]) -> list[QAPair]:
        """Generate and deduplicate QA pairs across all *chunks*."""
        all_pairs: list[QAPair] = []
        for chunk in chunks:
            pairs = self._generate_for_chunk(chunk)
            all_pairs.extend(pairs)
            logger.debug("Chunk %s: +%d pairs (total %d)", chunk.id, len(pairs), len(all_pairs))

        deduped = self._deduplicate(all_pairs)
        logger.info(
            "Generated %d pairs from %d chunks → %d after dedup",
            len(all_pairs),
            len(chunks),
            len(deduped),
        )
        return deduped

    @staticmethod
    def save(pairs: list[QAPair], path: Path) -> None:
        """Write pairs to a *path* as a human-reviewable JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump([p.to_dict() for p in pairs], fh, indent=2, ensure_ascii=False)
        logger.info("Saved %d QA pairs to %s", len(pairs), path)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_template(self) -> Template:
        if self._template is not None:
            return self._template
        template = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
        self._template = template
        return template

    def _generate_for_chunk(self, chunk: Chunk) -> list[QAPair]:
        prompt = self._get_template().substitute(n=self._n, passage=chunk.text)
        try:
            response = self._llm.generate(prompt=prompt, context="")
            raw_pairs: list[_RawPair] = _parse_json_pairs(response)
        except Exception as exc:
            logger.warning("LLM call failed for chunk %s: %s", chunk.id, exc)
            return []

        return [
            QAPair(
                question=p["question"].strip(),
                answer=p["answer"].strip(),
                relevant_chunks=[chunk.id],
                source=str(chunk.metadata.get("source", "")),
            )
            for p in raw_pairs
            if p.get("question") and p.get("answer")
        ]

    def _deduplicate(self, pairs: list[QAPair]) -> list[QAPair]:
        if len(pairs) <= 1:
            return pairs

        questions = [p.question for p in pairs]
        try:
            vecs = np.array(self._embedder.embed(questions), dtype=np.float32)
        except Exception as exc:
            logger.warning("Embedding failed during dedup; skipping: %s", exc)
            return pairs

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalised = vecs / np.where(norms > 0, norms, 1.0)
        sim = normalised @ normalised.T

        keep: list[QAPair] = []
        removed: set[int] = set()
        for i in range(len(pairs)):
            if i in removed:
                continue
            keep.append(pairs[i])
            for j in range(i + 1, len(pairs)):
                if j not in removed and sim[i, j] >= self._dedup_threshold:
                    removed.add(j)

        return keep


# ── helpers ────────────────────────────────────────────────────────────────────


def _parse_json_pairs(text: str) -> list[_RawPair]:
    """Extract a JSON array of {question, answer} dicts from LLM output."""

    def _to_raw_pairs(obj: object) -> list[_RawPair] | None:
        if not isinstance(obj, list):
            return None
        result: list[_RawPair] = []
        for item in obj:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                result.append(_RawPair(question=str(item["question"]), answer=str(item["answer"])))
        return result

    # Try the whole response first.
    try:
        pairs = _to_raw_pairs(json.loads(text.strip()))
        if pairs is not None:
            return pairs
    except json.JSONDecodeError:
        pass

    # Fall back to the first JSON array found anywhere in the text.
    match = re.search(r"\[.*?]", text, re.DOTALL)
    if match:
        try:
            pairs = _to_raw_pairs(json.loads(match.group()))
            if pairs is not None:
                return pairs
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON pairs from LLM response")
    return []
