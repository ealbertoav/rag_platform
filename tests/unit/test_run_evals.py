"""Unit tests for scripts/run_evals.py (T-152)."""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.domain.entities.chunk import Chunk
from src.evals.golden_dataset import (
    MIN_QA_PAIRS,
    QAPair,
    SyntheticDatasetBuilder,
    chunks_needed_for_min_pairs,
)


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"passage {i}")


def _qa_pair(i: int) -> QAPair:
    return QAPair(
        question=f"Question {i}?",
        answer=f"Answer {i}.",
        relevant_chunks=[f"c{i}"],
        source="doc.md",
    )


def _mock_builder(pairs: list[QAPair]) -> MagicMock:
    builder = MagicMock()
    builder.generate_from_chunks.return_value = pairs
    builder.save.side_effect = SyntheticDatasetBuilder.save
    return builder


def _golden_pairs(count: int = MIN_QA_PAIRS) -> list[QAPair]:
    return [_qa_pair(i) for i in range(count)]


def _bm25_mock(chunk_count: int = 10) -> MagicMock:
    chunk_list = [_chunk(i) for i in range(chunk_count)]
    bm25 = MagicMock()
    bm25.size = chunk_count
    bm25.iter_chunks = lambda: iter(chunk_list)
    return bm25


def _output_paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "qa.json", tmp_path / "retrieval.json"


@contextmanager
def _patch_run_evals(bm25: MagicMock, builder: MagicMock) -> Generator[None]:
    with (
        patch("src.infrastructure.vectordb.bm25.BM25Index.load_or_create", return_value=bm25),
        patch(
            "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings",
            return_value=MagicMock(),
        ),
        patch(
            "src.infrastructure.embeddings.bge_m3.BGEM3EmbeddingProvider.from_settings",
            return_value=MagicMock(),
        ),
        patch("run_evals.SyntheticDatasetBuilder", return_value=builder),
    ):
        yield


class TestRunEvalsMain:
    def test_exits_when_bm25_empty(self, monkeypatch: pytest.MonkeyPatch):
        import run_evals

        monkeypatch.setattr(sys, "argv", ["run_evals.py"])
        with (
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=MagicMock(size=0),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            run_evals.main()
        assert exc.value.code == 1

    def test_exits_when_below_min_pairs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        import run_evals

        bm25 = MagicMock()
        bm25.size = 2
        bm25.iter_chunks = lambda: iter([_chunk(0), _chunk(1)])

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_evals.py", "--min-pairs", "5", "--output", str(tmp_path / "qa.json")],
        )
        with (
            _patch_run_evals(bm25, _mock_builder([_qa_pair(0)])),
            pytest.raises(SystemExit) as exc,
        ):
            run_evals.main()
        assert exc.value.code == 1

    def test_writes_qa_and_retrieval_goldens(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        import run_evals

        pairs = _golden_pairs()
        bm25 = _bm25_mock()
        qa_out, retrieval_out = _output_paths(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_evals.py",
                "--output",
                str(qa_out),
                "--retrieval-output",
                str(retrieval_out),
            ],
        )
        with _patch_run_evals(bm25, _mock_builder(pairs)):
            run_evals.main()

        qa_data = json.loads(qa_out.read_text())
        retrieval_data = json.loads(retrieval_out.read_text())
        assert len(qa_data) == MIN_QA_PAIRS
        assert len(retrieval_data) == MIN_QA_PAIRS
        assert retrieval_data[0]["query"] == pairs[0].question

    def test_skips_retrieval_sync_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import run_evals

        pairs = _golden_pairs()
        bm25 = _bm25_mock()
        qa_out, retrieval_out = _output_paths(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_evals.py",
                "--output",
                str(qa_out),
                "--retrieval-output",
                str(retrieval_out),
                "--no-sync-retrieval",
            ],
        )
        with _patch_run_evals(bm25, _mock_builder(pairs)):
            run_evals.main()

        assert qa_out.exists()
        assert not retrieval_out.exists()

    def test_custom_output_syncs_sibling_retrieval_not_committed_golden(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import run_evals

        pairs = _golden_pairs()
        bm25 = _bm25_mock()
        committed_retrieval = tmp_path / "goldens" / "retrieval_dataset.json"
        committed_retrieval.parent.mkdir(parents=True, exist_ok=True)
        committed_retrieval.write_text('{"sentinel": true}', encoding="utf-8")

        qa_out = tmp_path / "custom" / "qa.json"
        sibling_retrieval = tmp_path / "custom" / "retrieval_dataset.json"

        monkeypatch.setattr(
            run_evals,
            "_default_retrieval_output",
            lambda: committed_retrieval,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_evals.py", "--output", str(qa_out)],
        )
        with _patch_run_evals(bm25, _mock_builder(pairs)):
            run_evals.main()

        assert json.loads(committed_retrieval.read_text()) == {"sentinel": True}
        assert sibling_retrieval.exists()
        retrieval_data = json.loads(sibling_retrieval.read_text())
        assert len(retrieval_data) == MIN_QA_PAIRS

    def test_expands_chunks_when_initial_batch_below_min_pairs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import run_evals

        chunk_count = 100
        bm25 = _bm25_mock(chunk_count)
        initial = chunks_needed_for_min_pairs(MIN_QA_PAIRS, 3)
        qa_out = tmp_path / "qa.json"

        builder = _mock_builder(_golden_pairs())

        def _generate(batch: list[Chunk]) -> list[QAPair]:
            if len(batch) <= initial:
                return [_qa_pair(i) for i in range(5)]
            return _golden_pairs()

        builder.generate_from_chunks.side_effect = _generate

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_evals.py", "--output", str(qa_out), "--no-sync-retrieval"],
        )
        with _patch_run_evals(bm25, builder):
            run_evals.main()

        assert builder.generate_from_chunks.call_count >= 2
        final_batch = builder.generate_from_chunks.call_args[0][0]
        assert len(final_batch) > initial

    def test_default_max_chunks_uses_dedup_headroom(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import run_evals

        pairs = _golden_pairs()
        chunk_count = 100
        bm25 = _bm25_mock(chunk_count)
        expected = chunks_needed_for_min_pairs(MIN_QA_PAIRS, 3)

        qa_out = tmp_path / "qa.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_evals.py", "--output", str(qa_out), "--no-sync-retrieval"],
        )
        builder = _mock_builder(pairs)
        with _patch_run_evals(bm25, builder):
            run_evals.main()

        passed_chunks = builder.generate_from_chunks.call_args[0][0]
        assert len(passed_chunks) == expected
