"""T-040 — SyntheticDatasetBuilder tests."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np

from src.domain.entities.chunk import Chunk
from src.evals.golden_dataset import (
    MIN_QA_PAIRS,
    QAPair,
    SyntheticDatasetBuilder,
    chunks_needed_for_min_pairs,
    count_real_qa_pairs,
    dedup_retention_estimate,
    filter_real_qa_pairs,
    is_evaluable_qa_pair,
    is_placeholder_chunk_ids,
    is_placeholder_qa_pair,
    is_placeholder_retrieval_row,
    load_qa_dicts,
    qa_dicts_to_retrieval_rows,
    qa_pairs_to_retrieval_rows,
    resolve_max_chunks,
    retrieval_rows_match_qa,
    save_retrieval_dataset,
    sync_retrieval_from_qa,
)


def _internal(module: str, name: str) -> object:
    return getattr(importlib.import_module(module), name)


parse_json_pairs = cast(
    Callable[[str], list[dict[str, str]]],
    _internal("src.evals.golden_dataset", "_parse_json_pairs"),
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int, text: str = "relevant passage text") -> Chunk:
    return Chunk(
        id=f"c{i}",
        document_id="doc",
        text=text,
        metadata={"source": f"doc{i}.pdf"},
    )


_VALID_JSON = (
    '[{"question": "What is X?", "answer": "X is Y."},'
    ' {"question": "Why Z?", "answer": "Because W."}]'
)


def _llm_mock(response: str = _VALID_JSON) -> MagicMock:
    m = MagicMock()
    m.generate.return_value = response
    return m


def _embedder_mock(dim: int = 4) -> MagicMock:
    m = MagicMock()
    rng = np.random.default_rng(42)
    m.embed.side_effect = lambda texts: rng.random((len(texts), dim)).tolist()
    return m


def _builder(
    response: str = _VALID_JSON,
    n: int = 2,
    dedup: float = 0.95,
) -> SyntheticDatasetBuilder:
    return SyntheticDatasetBuilder(
        llm=_llm_mock(response),
        embedder=_embedder_mock(),
        n_pairs_per_chunk=n,
        dedup_threshold=dedup,
    )


# ── T-152 golden dataset helpers ───────────────────────────────────────────────


class TestPlaceholderDetection:
    def test_placeholder_chunk_ids_true(self):
        assert is_placeholder_chunk_ids(["chunk_id_1", "chunk_id_2"])

    def test_placeholder_chunk_ids_false_mixed(self):
        assert not is_placeholder_chunk_ids(["c0", "chunk_id_1"])

    def test_placeholder_chunk_ids_false_empty(self):
        assert not is_placeholder_chunk_ids([])

    def test_placeholder_qa_pair(self):
        assert is_placeholder_qa_pair({"relevant_chunks": ["chunk_id_1"]})
        assert not is_placeholder_qa_pair({"relevant_chunks": ["rag_c001"]})
        assert not is_placeholder_qa_pair({"relevant_chunks": []})

    def test_is_evaluable_qa_pair(self):
        assert is_evaluable_qa_pair({"question": "Real?", "relevant_chunks": ["c0"]})
        assert not is_evaluable_qa_pair({"question": "", "relevant_chunks": ["c0"]})
        assert not is_evaluable_qa_pair({"question": "Empty?", "relevant_chunks": []})
        assert not is_evaluable_qa_pair(
            {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]}
        )
        assert is_evaluable_qa_pair({"question": "Legacy?", "relevant_chunks": "bad"})

    def test_placeholder_retrieval_row(self):
        assert is_placeholder_retrieval_row({"relevant_chunk_ids": ["chunk_id_1"]})
        assert not is_placeholder_retrieval_row({"relevant_chunk_ids": ["rag_c001"]})

    def test_placeholder_retrieval_row_empty_or_missing(self):
        assert not is_placeholder_retrieval_row({})
        assert not is_placeholder_retrieval_row({"relevant_chunk_ids": []})
        assert not is_placeholder_retrieval_row({"relevant_chunk_ids": "bad"})


class TestFilterRealQaPairs:
    def test_filters_placeholders_and_empty_questions(self):
        pairs = filter_real_qa_pairs(
            [
                {"question": "Real?", "relevant_chunks": ["c0"]},
                {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
                {"question": "", "relevant_chunks": ["c1"]},
            ]
        )
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Real?"

    def test_filters_empty_relevant_chunks(self):
        pairs = filter_real_qa_pairs(
            [
                {"question": "Real?", "relevant_chunks": ["c0"]},
                {"question": "Missing chunks?", "relevant_chunks": []},
            ]
        )
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Real?"


class TestRetrievalSync:
    def test_qa_pairs_to_retrieval_rows(self):
        pairs = [
            QAPair(question="What is RAG?", answer="Retrieval augmented.", relevant_chunks=["c0"])
        ]
        rows = qa_pairs_to_retrieval_rows(pairs)
        assert rows[0]["id"] == "retrieval_001"
        assert rows[0]["query"] == "What is RAG?"
        assert rows[0]["relevant_chunk_ids"] == ["c0"]

    def test_qa_dicts_to_retrieval_rows_filters_non_evaluable(self):
        pairs: list[dict[str, object]] = [
            {"question": "Real?", "relevant_chunks": ["c0"]},
            {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
            {"question": "Empty?", "relevant_chunks": []},
            {"question": "Legacy?", "relevant_chunks": "bad"},
        ]
        rows = qa_dicts_to_retrieval_rows(pairs)
        assert len(rows) == 2
        assert rows[0]["query"] == "Real?"
        assert rows[1]["query"] == "Legacy?"
        assert rows[1]["relevant_chunk_ids"] == []

    def test_retrieval_rows_match_qa(self):
        qa_pairs: list[dict[str, object]] = [
            {"question": "Real?", "relevant_chunks": ["c0"]},
            {"question": "Also real?", "relevant_chunks": ["c1"]},
        ]
        expected = qa_dicts_to_retrieval_rows(qa_pairs)
        assert retrieval_rows_match_qa(qa_pairs, expected)
        assert not retrieval_rows_match_qa(qa_pairs, expected[:1])

    def test_retrieval_rows_match_qa_ignores_non_dict_rows(self):
        qa_pairs: list[dict[str, object]] = [
            {"question": "Real?", "relevant_chunks": ["c0"]},
        ]
        expected = qa_dicts_to_retrieval_rows(qa_pairs)
        assert retrieval_rows_match_qa(qa_pairs, [*expected, "bad"])

    def test_sync_retrieval_from_qa(self, tmp_path: Path):
        qa_path = tmp_path / "qa.json"
        retrieval_path = tmp_path / "retrieval.json"
        qa_path.write_text(
            json.dumps([{"question": "What is RAG?", "relevant_chunks": ["rag_c001"]}]),
            encoding="utf-8",
        )
        count = sync_retrieval_from_qa(qa_path, retrieval_path)
        assert count == 1
        rows = json.loads(retrieval_path.read_text())
        assert rows[0]["query"] == "What is RAG?"

    def test_load_qa_dicts(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps([{"question": "Q?", "relevant_chunks": ["c0"]}, "bad"]),
            encoding="utf-8",
        )
        assert len(load_qa_dicts(path)) == 1

    def test_load_qa_dicts_missing_or_invalid(self, tmp_path: Path):
        assert load_qa_dicts(tmp_path / "missing.json") == []
        path = tmp_path / "qa.json"
        path.write_text("not json", encoding="utf-8")
        assert load_qa_dicts(path) == []
        path.write_text('{"not": "a list"}', encoding="utf-8")
        assert load_qa_dicts(path) == []

    def test_save_retrieval_dataset(self, tmp_path: Path):
        rows: list[dict[str, object]] = [{"id": "r1", "query": "q?", "relevant_chunk_ids": ["c0"]}]
        out = tmp_path / "retrieval.json"
        save_retrieval_dataset(rows, out)
        data = json.loads(out.read_text())
        assert data[0]["query"] == "q?"


class TestCountRealQaPairs:
    def test_counts_non_placeholder_rows(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(
            json.dumps(
                [
                    {"question": "Real?", "relevant_chunks": ["c0"]},
                    {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
                ]
            ),
            encoding="utf-8",
        )
        assert count_real_qa_pairs(path) == 1

    def test_missing_file_returns_zero(self, tmp_path: Path):
        assert count_real_qa_pairs(tmp_path / "missing.json") == 0

    def test_non_list_json_returns_zero(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text('{"not": "a list"}', encoding="utf-8")
        assert count_real_qa_pairs(path) == 0

    def test_min_qa_pairs_constant(self):
        assert MIN_QA_PAIRS == 20


class TestChunkEstimation:
    def test_dedup_retention_clamped(self):
        assert dedup_retention_estimate(1.0) == 1.0
        assert dedup_retention_estimate(0.95) == 0.975
        assert dedup_retention_estimate(0.5) == 0.75
        assert dedup_retention_estimate(0.0) == 0.75

    def test_chunks_needed_accounts_for_dedup(self):
        naive = max(1, (MIN_QA_PAIRS + 2) // 3)
        with_dedup = chunks_needed_for_min_pairs(MIN_QA_PAIRS, 3, dedup_threshold=0.95)
        assert with_dedup >= naive

    def test_chunks_needed_minimum_one(self):
        assert chunks_needed_for_min_pairs(0, 0) == 1

    def test_resolve_max_chunks_uses_explicit_cap(self):
        assert resolve_max_chunks(100, min_pairs=20, n_pairs_per_chunk=3, max_chunks=5) == 5

    def test_resolve_max_chunks_defaults_with_dedup_headroom(self):
        default = resolve_max_chunks(100, min_pairs=20, n_pairs_per_chunk=3)
        naive = max(1, (20 + 3 - 1) // 3)
        assert default >= naive

    def test_resolve_max_chunks_capped_by_available(self):
        assert resolve_max_chunks(2, min_pairs=20, n_pairs_per_chunk=3) == 2


# ── _parse_json_pairs ──────────────────────────────────────────────────────────


class TestParseJsonPairs:
    def test_valid_json_array(self):
        result = parse_json_pairs(_VALID_JSON)
        assert len(result) == 2
        assert result[0]["question"] == "What is X?"

    def test_json_embedded_in_text(self):
        text = f"Here are the pairs:\n{_VALID_JSON}\nThat's all."
        result = parse_json_pairs(text)
        assert len(result) == 2

    def test_empty_response_returns_empty(self):
        assert parse_json_pairs("") == []

    def test_invalid_json_returns_empty(self):
        assert parse_json_pairs("not json") == []

    def test_non_list_returns_empty(self):
        assert parse_json_pairs('{"question": "q", "answer": "a"}') == []

    def test_invalid_embedded_json_array_returns_empty(self):
        assert parse_json_pairs("Pairs:\n[not valid json]\nDone.") == []


# ── QAPair ─────────────────────────────────────────────────────────────────────


class TestQAPair:
    def test_to_dict_has_required_keys(self):
        pair = QAPair(question="q?", answer="a.", relevant_chunks=["c0"])
        d = pair.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "relevant_chunks" in d

    def test_to_dict_values(self):
        pair = QAPair(question="q?", answer="a.", relevant_chunks=["c0"], source="doc.pdf")
        d = pair.to_dict()
        assert d["question"] == "q?"
        assert d["source"] == "doc.pdf"


# ── SyntheticDatasetBuilder ────────────────────────────────────────────────────


class TestGenerateFromChunks:
    def test_returns_list_of_qa_pairs(self):
        result = _builder().generate_from_chunks([_chunk(0)])
        assert isinstance(result, list)
        assert all(isinstance(p, QAPair) for p in result)

    def test_generates_pairs_per_chunk(self):
        result = _builder().generate_from_chunks([_chunk(0)])
        assert len(result) >= 1

    def test_multiple_chunks_combined(self):
        result = _builder().generate_from_chunks([_chunk(0), _chunk(1)])
        assert len(result) >= 2

    def test_chunk_id_in_relevant_chunks(self):
        result = _builder().generate_from_chunks([_chunk(7)])
        assert all("c7" in p.relevant_chunks for p in result)

    def test_source_from_chunk_metadata(self):
        result = _builder().generate_from_chunks([_chunk(0)])
        assert all(p.source == "doc0.pdf" for p in result)

    def test_llm_failure_returns_empty(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        builder = SyntheticDatasetBuilder(llm=llm, embedder=_embedder_mock())
        assert builder.generate_from_chunks([_chunk(0)]) == []

    def test_malformed_llm_response_skipped(self):
        result = _builder("not valid json").generate_from_chunks([_chunk(0)])
        assert result == []

    def test_skips_pairs_with_empty_question(self):
        bad_json = '[{"question": "", "answer": "a."}, {"question": "Real Q?", "answer": "a."}]'
        result = _builder(bad_json).generate_from_chunks([_chunk(0)])
        assert all(p.question != "" for p in result)

    def test_empty_chunks_returns_empty(self):
        assert _builder().generate_from_chunks([]) == []


class TestDeduplicate:
    def test_exact_duplicate_removed(self):
        # Make the embedder return identical vectors for all questions
        llm = _llm_mock('[{"question":"Q","answer":"A"},{"question":"Q","answer":"A"}]')
        embedder = MagicMock()
        # All embeddings are identical → cosine sim = 1.0 → second is removed
        embedder.embed.return_value = [[1.0, 0.0, 0.0, 0.0]] * 2
        builder = SyntheticDatasetBuilder(llm=llm, embedder=embedder, dedup_threshold=0.95)
        result = builder.generate_from_chunks([_chunk(0)])
        assert len(result) == 1

    def test_dissimilar_questions_kept(self):
        pairs_json = (
            '[{"question":"What is A?","answer":"A."},{"question":"Why does B?","answer":"B."}]'
        )
        llm = _llm_mock(pairs_json)
        embedder = MagicMock()
        # Orthogonal vectors → cosine sim = 0.0 → both kept
        embedder.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
        builder = SyntheticDatasetBuilder(llm=llm, embedder=embedder, dedup_threshold=0.95)
        result = builder.generate_from_chunks([_chunk(0)])
        assert len(result) == 2

    def test_embedding_failure_returns_original(self):
        llm = _llm_mock()
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("embed failed")
        builder = SyntheticDatasetBuilder(llm=llm, embedder=embedder)
        result = builder.generate_from_chunks([_chunk(0)])
        assert len(result) >= 1  # no dedup, returns all


class TestSave:
    def test_creates_output_file(self, tmp_path: Path):
        pairs = [QAPair(question="q?", answer="a.", relevant_chunks=["c0"])]
        out = tmp_path / "qa.json"
        _builder().save(pairs, out)
        assert out.exists()

    def test_output_is_valid_json(self, tmp_path: Path):
        pairs = [QAPair(question="q?", answer="a.", relevant_chunks=["c0"])]
        out = tmp_path / "qa.json"
        _builder().save(pairs, out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert data[0]["question"] == "q?"

    def test_creates_parent_dirs(self, tmp_path: Path):
        pairs = [QAPair(question="q?", answer="a.", relevant_chunks=["c0"])]
        out = tmp_path / "nested" / "dir" / "qa.json"
        _builder().save(pairs, out)
        assert out.exists()
