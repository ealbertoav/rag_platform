"""Unit tests for scripts/build_multimodal_golden.py (T-280)."""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import CHUNK_TYPE_KEY, CHUNK_TYPE_TABLE, MODALITY_FIGURE, MODALITY_TABLE
from src.domain.entities.chunk import Chunk
from src.evals.golden_dataset import QAPair


def _table_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"table{i}",
        document_id="doc",
        text=f"table text {i}",
        metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_TABLE},
    )


def _figure_chunk(i: int) -> Chunk:
    return Chunk(id=f"figure{i}", document_id="doc", text=f"figure {i}", modality=MODALITY_FIGURE)


def _text_chunk(i: int) -> Chunk:
    return Chunk(id=f"text{i}", document_id="doc", text=f"prose {i}")


def _bm25_mock(chunks: list[Chunk]) -> MagicMock:
    bm25 = MagicMock()
    bm25.size = len(chunks)
    bm25.iter_chunks = lambda: iter(chunks)
    return bm25


def _mock_builder(pairs: list[QAPair]) -> MagicMock:
    builder = MagicMock()
    builder.generate_from_chunks.return_value = pairs
    return builder


@contextmanager
def _patch_script(bm25: MagicMock, builder: MagicMock) -> Generator[None]:
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
        patch("build_multimodal_golden.SyntheticDatasetBuilder", return_value=builder),
    ):
        yield


class TestBuildMultimodalGoldenMain:
    def test_exits_when_bm25_empty(self, monkeypatch: pytest.MonkeyPatch):
        import build_multimodal_golden

        monkeypatch.setattr(sys, "argv", ["build_multimodal_golden.py"])
        with (
            patch(
                "src.infrastructure.vectordb.bm25.BM25Index.load_or_create",
                return_value=MagicMock(size=0),
            ),
            pytest.raises(SystemExit) as exc,
        ):
            build_multimodal_golden.main()
        assert exc.value.code == 1

    def test_exits_when_no_multimodal_chunks(self, monkeypatch: pytest.MonkeyPatch):
        import build_multimodal_golden

        bm25 = _bm25_mock([_text_chunk(0), _text_chunk(1)])
        monkeypatch.setattr(sys, "argv", ["build_multimodal_golden.py"])
        with (
            _patch_script(bm25, _mock_builder([])),
            pytest.raises(SystemExit) as exc,
        ):
            build_multimodal_golden.main()
        assert exc.value.code == 1

    def test_exits_when_no_pairs_generated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        import build_multimodal_golden

        bm25 = _bm25_mock([_table_chunk(0)])
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_multimodal_golden.py", "--output", str(tmp_path / "out.jsonl")],
        )
        with (
            _patch_script(bm25, _mock_builder([])),
            pytest.raises(SystemExit) as exc,
        ):
            build_multimodal_golden.main()
        assert exc.value.code == 1

    def test_writes_jsonl_golden_from_table_and_figure_chunks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        import build_multimodal_golden

        table = _table_chunk(0)
        figure = _figure_chunk(0)
        bm25 = _bm25_mock([table, figure, _text_chunk(0)])
        pairs = [
            QAPair(question="Table Q", answer="A", relevant_chunks=["table0"]),
            QAPair(question="Figure Q", answer="A", relevant_chunks=["figure0"]),
        ]
        output = tmp_path / "multimodal_qa_dataset.jsonl"

        monkeypatch.setattr(
            sys,
            "argv",
            ["build_multimodal_golden.py", "--output", str(output)],
        )
        with _patch_script(bm25, _mock_builder(pairs)):
            build_multimodal_golden.main()

        lines = output.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        rows = [json.loads(line) for line in lines]
        modalities = {row["modality"] for row in rows}
        assert modalities == {MODALITY_TABLE, MODALITY_FIGURE}
