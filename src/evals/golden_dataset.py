from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from string import Template
from typing import TypedDict, cast

import numpy as np

from src.domain.entities.chunk import Chunk
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "evaluation" / "generate_qa.txt"
MIN_QA_PAIRS = 20
_DEFAULT_RETRIEVAL_PATH = (
    Path(__file__).parents[2] / "datasets" / "goldens" / "retrieval_dataset.json"
)
_DEFAULT_QA_GOLDEN_PATH = Path(__file__).parents[2] / "datasets" / "goldens" / "qa_dataset.json"


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
        return cast(dict[str, object], dataclasses.asdict(self))


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


# ── Golden dataset helpers (T-152) ───────────────────────────────────────────


def is_placeholder_chunk_ids(chunk_ids: list[object]) -> bool:
    """Return True when every ID is a placeholder (chunk_id_*)."""
    # pyrefly: ignore [unnecessary-type-conversion]
    str_ids = [str(r) for r in chunk_ids if isinstance(r, str)]
    return bool(str_ids) and all(r.startswith("chunk_id_") for r in str_ids)


def is_placeholder_qa_pair(pair: dict[str, object]) -> bool:
    """Return True when all relevant_chunks are placeholder IDs (chunk_id_*)."""
    chunks = pair.get("relevant_chunks")
    if not isinstance(chunks, list) or not chunks:
        return False
    return is_placeholder_chunk_ids(chunks)


def is_evaluable_qa_pair(pair: dict[str, object]) -> bool:
    """Return True when a QA row should participate in benchmarks."""
    if not pair.get("question"):
        return False
    chunks = pair.get("relevant_chunks")
    if not isinstance(chunks, list):
        return True
    if not chunks:
        return False
    return not is_placeholder_chunk_ids(chunks)


def is_placeholder_retrieval_row(entry: dict[str, object]) -> bool:
    """Return True when all relevant_chunk_ids are placeholder IDs."""
    raw_ids = entry.get("relevant_chunk_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return False
    return is_placeholder_chunk_ids(raw_ids)


def filter_real_qa_pairs(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop rows without a question, empty relevant_chunks, and placeholder-only IDs."""
    return [pair for pair in pairs if is_evaluable_qa_pair(pair)]


def qa_pairs_to_retrieval_rows(pairs: list[QAPair]) -> list[dict[str, object]]:
    """Convert QA pairs to retrieval golden rows (query + relevant_chunk_ids)."""
    return _retrieval_rows_from_qa_content(
        [(pair.question, list(pair.relevant_chunks)) for pair in pairs]
    )


def qa_dicts_to_retrieval_rows(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert evaluable QA dict rows to retrieval golden rows."""
    evaluable = filter_real_qa_pairs(pairs)
    content = []
    for pair in evaluable:
        raw_chunks = pair.get("relevant_chunks")
        if isinstance(raw_chunks, list):
            # pyrefly: ignore [unnecessary-type-conversion]
            chunk_ids = [str(chunk_id) for chunk_id in raw_chunks if isinstance(chunk_id, str)]
        else:
            chunk_ids = []
        content.append((str(pair["question"]), chunk_ids))
    return _retrieval_rows_from_qa_content(content)


def _retrieval_rows_from_qa_content(
    rows: list[tuple[str, list[str]]],
) -> list[dict[str, object]]:
    return [
        {
            "id": f"retrieval_{index + 1:03d}",
            "query": question,
            "relevant_chunk_ids": chunk_ids,
        }
        for index, (question, chunk_ids) in enumerate(rows)
    ]


def load_qa_dicts(path: Path | None = None) -> list[dict[str, object]]:
    """Load QA golden rows from disk."""
    qa_path = path or _DEFAULT_QA_GOLDEN_PATH
    try:
        raw: object = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def retrieval_rows_match_qa(
    qa_pairs: list[dict[str, object]],
    retrieval_rows: Sequence[object],
) -> bool:
    """Return True when retrieval rows are the sync output of evaluable QA pairs."""
    expected = qa_dicts_to_retrieval_rows(qa_pairs)
    actual = [
        {
            "query": row.get("query"),
            "relevant_chunk_ids": row.get("relevant_chunk_ids"),
        }
        for row in retrieval_rows
        if isinstance(row, dict)
    ]
    normalized_expected = [
        {"query": row["query"], "relevant_chunk_ids": row["relevant_chunk_ids"]} for row in expected
    ]
    return normalized_expected == actual


def save_retrieval_dataset(rows: list[dict[str, object]], path: Path | None = None) -> None:
    """Write retrieval golden rows (default: datasets/goldens/retrieval_dataset.json)."""
    target = path or _DEFAULT_RETRIEVAL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d retrieval rows to %s", len(rows), target)


def sync_retrieval_from_qa(
    qa_path: Path | None = None,
    retrieval_path: Path | None = None,
) -> int:
    """Derive retrieval golden rows from evaluable QA rows and write them to disk."""
    qa_pairs = load_qa_dicts(qa_path)
    rows = qa_dicts_to_retrieval_rows(qa_pairs)
    save_retrieval_dataset(rows, retrieval_path)
    return len(rows)


def dedup_retention_estimate(dedup_threshold: float) -> float:
    """Estimate the fraction of generated QA pairs that survive deduplication.

    Lower cosine thresholds remove more near-duplicate questions across chunks.
    """
    clamped = max(0.5, min(1.0, dedup_threshold))
    return 0.5 + 0.5 * clamped


def chunks_needed_for_min_pairs(
    min_pairs: int,
    n_pairs_per_chunk: int,
    *,
    dedup_threshold: float = 0.95,
) -> int:
    """Return the minimum chunk count likely to reach *min_pairs* after dedup."""
    if min_pairs <= 0 or n_pairs_per_chunk <= 0:
        return 1
    retention = dedup_retention_estimate(dedup_threshold)
    raw_pairs_needed = math.ceil(min_pairs / retention)
    return max(1, math.ceil(raw_pairs_needed / n_pairs_per_chunk))


def resolve_max_chunks(
    total_chunks: int,
    *,
    min_pairs: int,
    n_pairs_per_chunk: int,
    max_chunks: int | None = None,
    dedup_threshold: float = 0.95,
) -> int:
    """Pick how many chunks to process, accounting for expected dedup losses."""
    if max_chunks is not None:
        return min(total_chunks, max(1, max_chunks))
    needed = chunks_needed_for_min_pairs(
        min_pairs,
        n_pairs_per_chunk,
        dedup_threshold=dedup_threshold,
    )
    return min(total_chunks, needed)


def resolve_retrieval_output_path(
    qa_output: Path,
    *,
    retrieval_output: Path | None = None,
    qa_golden_path: Path | None = None,
    retrieval_golden_path: Path | None = None,
) -> Path:
    """Return the retrieval golden path paired with *qa_output*.

    Custom QA outputs sync retrieval alongside the QA file instead of
    overwriting the committed "datasets/goldens/retrieval_dataset.json".
    """
    default_qa = qa_golden_path or _DEFAULT_QA_GOLDEN_PATH
    default_retrieval = retrieval_golden_path or _DEFAULT_RETRIEVAL_PATH
    if retrieval_output is not None:
        return retrieval_output
    if qa_output.resolve() == default_qa.resolve():
        return default_retrieval
    return qa_output.parent / "retrieval_dataset.json"


def generate_until_min_pairs(
    builder: SyntheticDatasetBuilder,
    chunks: Sequence[Chunk] | None = None,
    *,
    total_chunks: int | None = None,
    iter_chunks: Callable[[], Iterator[Chunk]] | None = None,
    min_pairs: int,
    n_pairs_per_chunk: int,
    dedup_threshold: float = 0.95,
    max_chunks: int | None = None,
) -> tuple[list[QAPair], int]:
    """Generate QA pairs, expanding the chunk window until *min_pairs* or exhaustion."""
    if chunks is not None:
        corpus_size = len(chunks)

        def _window(window_size: int) -> list[Chunk]:
            return list(chunks[:window_size])
    elif iter_chunks is not None and total_chunks is not None:
        corpus_size = total_chunks

        def _window(window_size: int) -> list[Chunk]:
            from itertools import islice

            return list(islice(iter_chunks(), window_size))
    else:
        return [], 0

    if corpus_size == 0:
        return [], 0

    if max_chunks is not None:
        limit = min(corpus_size, max(1, max_chunks))
        return builder.generate_from_chunks(_window(limit)), limit

    limit = resolve_max_chunks(
        corpus_size,
        min_pairs=min_pairs,
        n_pairs_per_chunk=n_pairs_per_chunk,
        max_chunks=None,
        dedup_threshold=dedup_threshold,
    )
    while True:
        pairs = builder.generate_from_chunks(_window(limit))
        if len(pairs) >= min_pairs or limit >= corpus_size:
            return pairs, limit
        shortfall = min_pairs - len(pairs)
        extra = chunks_needed_for_min_pairs(
            shortfall,
            n_pairs_per_chunk,
            dedup_threshold=dedup_threshold,
        )
        next_limit = min(corpus_size, limit + extra)
        if next_limit <= limit:
            next_limit = corpus_size
        limit = next_limit


def count_real_qa_pairs(path: Path | None = None) -> int:
    """Count non-placeholder QA rows in the golden dataset file."""
    qa_path = path or _DEFAULT_QA_GOLDEN_PATH
    try:
        raw: object = json.loads(qa_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(raw, list):
        return 0
    candidates = [item for item in raw if isinstance(item, dict)]
    return len(filter_real_qa_pairs(candidates))


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
