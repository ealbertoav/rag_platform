"""T-280 — MultimodalGoldenDataset tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.core.constants import (
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_TABLE,
    MODALITY_FIGURE,
    MODALITY_TABLE,
    MODALITY_TEXT,
)
from src.domain.entities.chunk import Chunk
from src.evals.golden_dataset import QAPair
from src.evals.multimodal_golden_dataset import (
    MultimodalQAPair,
    build_multimodal_golden,
    chunk_modality,
    filter_multimodal_chunks,
    load_jsonl,
    save_jsonl,
)


def _table_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"table{i}",
        document_id="doc",
        text=f"table text {i}",
        metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_TABLE},
    )


def _figure_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"figure{i}",
        document_id="doc",
        text=f"figure caption {i}",
        modality=MODALITY_FIGURE,
    )


def _caption_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"caption{i}",
        document_id="doc",
        text=f"caption text {i}",
        metadata={CHUNK_TYPE_KEY: CHUNK_TYPE_CAPTION},
    )


def _text_chunk(i: int) -> Chunk:
    return Chunk(id=f"text{i}", document_id="doc", text=f"prose {i}")


class TestChunkModality:
    def test_table_chunk_via_metadata_type(self):
        assert chunk_modality(_table_chunk(0)) == MODALITY_TABLE

    def test_figure_chunk_via_explicit_modality(self):
        assert chunk_modality(_figure_chunk(0)) == MODALITY_FIGURE

    def test_text_chunk_defaults_to_text(self):
        assert chunk_modality(_text_chunk(0)) == MODALITY_TEXT


class TestFilterMultimodalChunks:
    def test_keeps_only_table_and_figure_chunks(self):
        chunks = [_table_chunk(0), _figure_chunk(0), _caption_chunk(0), _text_chunk(0)]
        filtered = filter_multimodal_chunks(chunks)
        assert {c.id for c in filtered} == {"table0", "figure0"}

    def test_empty_input_returns_empty(self):
        assert filter_multimodal_chunks([]) == []


class TestBuildMultimodalGolden:
    def test_generates_only_from_multimodal_chunks(self):
        table = _table_chunk(0)
        figure = _figure_chunk(0)
        text = _text_chunk(0)

        builder = MagicMock()
        builder.generate_from_chunks.return_value = [
            QAPair(question="What is in the table?", answer="X", relevant_chunks=["table0"]),
            QAPair(question="What does the figure show?", answer="Y", relevant_chunks=["figure0"]),
        ]

        result = build_multimodal_golden(builder, [table, figure, text])

        passed_chunks = builder.generate_from_chunks.call_args[0][0]
        assert {c.id for c in passed_chunks} == {"table0", "figure0"}
        assert len(result) == 2
        assert all(isinstance(pair, MultimodalQAPair) for pair in result)

    def test_tags_pairs_with_source_chunk_modality(self):
        table = _table_chunk(0)
        figure = _figure_chunk(0)

        builder = MagicMock()
        builder.generate_from_chunks.return_value = [
            QAPair(question="Table Q", answer="A", relevant_chunks=["table0"]),
            QAPair(question="Figure Q", answer="A", relevant_chunks=["figure0"]),
        ]

        result = build_multimodal_golden(builder, [table, figure])

        by_question = {pair.question: pair for pair in result}
        assert by_question["Table Q"].modality == MODALITY_TABLE
        assert by_question["Figure Q"].modality == MODALITY_FIGURE

    def test_drops_pairs_whose_source_chunk_is_unresolvable(self):
        table = _table_chunk(0)

        builder = MagicMock()
        builder.generate_from_chunks.return_value = [
            QAPair(question="Orphan Q", answer="A", relevant_chunks=["missing-chunk"]),
        ]

        result = build_multimodal_golden(builder, [table])
        assert result == []

    def test_no_multimodal_chunks_skips_llm_call(self):
        builder = MagicMock()
        result = build_multimodal_golden(builder, [_text_chunk(0)])

        builder.generate_from_chunks.assert_not_called()
        assert result == []


class TestJsonlRoundTrip:
    def test_save_and_load_jsonl(self, tmp_path: Path):
        pairs = [
            MultimodalQAPair(
                question="Q1",
                answer="A1",
                relevant_chunks=["table0"],
                modality=MODALITY_TABLE,
                source="doc.pdf",
            ),
            MultimodalQAPair(
                question="Q2",
                answer="A2",
                relevant_chunks=["figure0"],
                modality=MODALITY_FIGURE,
            ),
        ]
        path = tmp_path / "multimodal_qa_dataset.jsonl"

        save_jsonl(pairs, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["question"] == "Q1"

        loaded = load_jsonl(path)
        assert loaded == [pair.to_dict() for pair in pairs]

    def test_load_jsonl_missing_file_returns_empty(self, tmp_path: Path):
        assert load_jsonl(tmp_path / "nope.jsonl") == []

    def test_load_jsonl_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"question": "Q1"}\n\n{"question": "Q2"}\n', encoding="utf-8")
        loaded = load_jsonl(path)
        assert [row["question"] for row in loaded] == ["Q1", "Q2"]
