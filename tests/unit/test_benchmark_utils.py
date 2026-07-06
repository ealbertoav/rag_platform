"""Unit tests for scripts/_benchmark_utils.py."""

from __future__ import annotations

import json
from pathlib import Path

from _benchmark_utils import resolve_qa_pairs


def _write_qa(path: Path, pairs: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(pairs), encoding="utf-8")


class TestResolveQaPairs:
    def test_filters_placeholders_before_max_samples(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        goldens = tmp_path / "datasets" / "goldens"
        goldens.mkdir(parents=True)
        _write_qa(
            goldens / "qa_dataset.json",
            [
                {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
                {"question": "Placeholder 2?", "relevant_chunks": ["chunk_id_2"]},
                {"question": "Real 1?", "answer": "A1", "relevant_chunks": ["c0"]},
                {"question": "Real 2?", "answer": "A2", "relevant_chunks": ["c1"]},
                {"question": "Real 3?", "answer": "A3", "relevant_chunks": ["c2"]},
            ],
        )

        import src.core.constants as constants

        monkeypatch.setattr(constants, "DATASETS_DIR", tmp_path / "datasets")

        pairs = resolve_qa_pairs("qa_dataset.json", max_samples=2)
        assert pairs is not None
        assert len(pairs) == 2
        assert pairs[0]["question"] == "Real 1?"
        assert pairs[1]["question"] == "Real 2?"

    def test_returns_none_when_file_empty(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        goldens = tmp_path / "datasets" / "goldens"
        goldens.mkdir(parents=True)
        _write_qa(goldens / "qa_dataset.json", [])

        import src.core.constants as constants

        monkeypatch.setattr(constants, "DATASETS_DIR", tmp_path / "datasets")

        assert resolve_qa_pairs("qa_dataset.json", max_samples=10) is None
        assert "no QA pairs found" in capsys.readouterr().err

    def test_returns_empty_list_when_only_placeholders(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        goldens = tmp_path / "datasets" / "goldens"
        goldens.mkdir(parents=True)
        _write_qa(
            goldens / "qa_dataset.json",
            [{"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]}],
        )

        import src.core.constants as constants

        monkeypatch.setattr(constants, "DATASETS_DIR", tmp_path / "datasets")

        pairs = resolve_qa_pairs("qa_dataset.json", max_samples=5)
        assert pairs == []

    def test_skips_placeholder_filter_when_disabled(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        goldens = tmp_path / "datasets" / "goldens"
        goldens.mkdir(parents=True)
        _write_qa(
            goldens / "qa_dataset.json",
            [
                {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
                {"question": "Real?", "answer": "A", "relevant_chunks": ["c0"]},
            ],
        )

        import src.core.constants as constants

        monkeypatch.setattr(constants, "DATASETS_DIR", tmp_path / "datasets")

        pairs = resolve_qa_pairs(
            "qa_dataset.json",
            max_samples=1,
            filter_placeholders=False,
        )
        assert pairs is not None
        assert len(pairs) == 1
        assert pairs[0]["question"] == "Placeholder?"

    def test_absolute_path(self, tmp_path: Path):
        path = tmp_path / "custom.json"
        _write_qa(path, [{"question": "Q?", "answer": "A", "relevant_chunks": ["c0"]}])
        pairs = resolve_qa_pairs(str(path), max_samples=None)
        assert pairs is not None
        assert len(pairs) == 1
