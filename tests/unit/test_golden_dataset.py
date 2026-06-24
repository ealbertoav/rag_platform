"""T-040 — SyntheticDatasetBuilder tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from src.domain.entities.chunk import Chunk
from src.evals.golden_dataset import QAPair, SyntheticDatasetBuilder, _parse_json_pairs

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


def _embedder_mock(n_questions: int = 2, dim: int = 4) -> MagicMock:
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


# ── _parse_json_pairs ──────────────────────────────────────────────────────────


class TestParseJsonPairs:
    def test_valid_json_array(self):
        result = _parse_json_pairs(_VALID_JSON)
        assert len(result) == 2
        assert result[0]["question"] == "What is X?"

    def test_json_embedded_in_text(self):
        text = f"Here are the pairs:\n{_VALID_JSON}\nThat's all."
        result = _parse_json_pairs(text)
        assert len(result) == 2

    def test_empty_response_returns_empty(self):
        assert _parse_json_pairs("") == []

    def test_invalid_json_returns_empty(self):
        assert _parse_json_pairs("not json") == []

    def test_non_list_returns_empty(self):
        assert _parse_json_pairs('{"question": "q", "answer": "a"}') == []


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
