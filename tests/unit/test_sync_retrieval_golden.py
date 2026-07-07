"""Unit tests for scripts/sync_retrieval_golden.py (T-152)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evals.golden_dataset import sync_retrieval_from_qa


class TestSyncRetrievalFromQa:
    def test_writes_retrieval_rows_from_qa(self, tmp_path: Path):
        qa_path = tmp_path / "qa.json"
        retrieval_path = tmp_path / "retrieval.json"
        qa_path.write_text(
            json.dumps(
                [
                    {"question": "What is RAG?", "relevant_chunks": ["rag_c001"]},
                    {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
                ]
            ),
            encoding="utf-8",
        )

        count = sync_retrieval_from_qa(qa_path, retrieval_path)

        assert count == 1
        rows = json.loads(retrieval_path.read_text())
        assert rows[0]["query"] == "What is RAG?"
        assert rows[0]["relevant_chunk_ids"] == ["rag_c001"]


class TestSyncRetrievalGoldenMain:
    def test_main_prints_sync_summary(self, tmp_path: Path, capsys):
        import sync_retrieval_golden

        qa_path = tmp_path / "qa.json"
        retrieval_path = tmp_path / "retrieval.json"
        qa_path.write_text(
            json.dumps([{"question": "Real?", "relevant_chunks": ["rag_c001"]}]),
            encoding="utf-8",
        )

        with patch.object(
            sys,
            "argv",
            [
                "sync_retrieval_golden.py",
                "--qa",
                str(qa_path),
                "--retrieval-output",
                str(retrieval_path),
            ],
        ):
            sync_retrieval_golden.main()

        output = capsys.readouterr().out
        assert "1 retrieval rows synced" in output
        assert str(retrieval_path) in output

    def test_main_uses_default_paths(self, monkeypatch: pytest.MonkeyPatch, capsys):
        import sync_retrieval_golden

        monkeypatch.setattr(sys, "argv", ["sync_retrieval_golden.py"])
        with patch.object(
            sync_retrieval_golden,
            "sync_retrieval_from_qa",
            return_value=22,
        ) as mock_sync:
            sync_retrieval_golden.main()

        qa_path, retrieval_path = mock_sync.call_args[0]
        assert qa_path == sync_retrieval_golden._default_qa_path()
        assert retrieval_path == sync_retrieval_golden._default_retrieval_path()
        output = capsys.readouterr().out
        assert "22 retrieval rows synced" in output

    def test_script_entrypoint(self, monkeypatch: pytest.MonkeyPatch):
        import runpy
        from pathlib import Path

        monkeypatch.setattr(sys, "argv", ["sync_retrieval_golden.py"])
        with patch("sync_retrieval_golden.sync_retrieval_from_qa", return_value=1):
            runpy.run_path(
                str(Path("scripts/sync_retrieval_golden.py")),
                run_name="__main__",
            )
