"""T-151 — ChunkSizeSweep unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.domain.entities.chunk import Chunk
from src.domain.entities.evaluation import BenchmarkRun
from src.evals.e2e.chunk_size_sweep import (
    ChunkSizeResult,
    ChunkSizeSweep,
    ChunkSizeSweepReport,
    SweepPlanEntry,
    SweepWeights,
    bm25_cache_path,
    build_chunk_size_overrides,
    build_sweep_pipeline,
    build_sweep_plan,
    chunk_cache_path,
    chunk_documents_from_source,
    clear_vector_index,
    collection_name_for_size,
    compute_weighted_score,
    embed_chunks,
    index_chunks_for_size,
    iter_source_files,
    load_chunk_cache,
    load_sweep_sizes,
    load_sweep_weights,
    recommend_size,
    remap_relevant_chunks,
    run_chunk_size_sweep,
    save_chunk_cache,
)
from tests.unit.e2e_benchmark_helpers import metric_mock, pipeline_mock


def _qa(
    question: str = "What is EKS?",
    answer: str = "Kubernetes on AWS.",
    relevant: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": relevant or ["c0"],
    }


def _chunk(
    chunk_id: str = "c0",
    text: str = "Kubernetes on AWS runs EKS.",
    document_id: str = "doc1",
) -> Chunk:
    return Chunk(id=chunk_id, document_id=document_id, text=text, metadata={"source": "test.md"})


def _benchmark(faith: float = 0.9, relev: float = 0.85) -> ChunkSizeSweep:
    return ChunkSizeSweep(
        faithfulness=metric_mock(faith),
        relevance=metric_mock(relev),
        weights=SweepWeights(0.35, 0.35, 0.20, 0.10),
    )


# ── config / paths ─────────────────────────────────────────────────────────────


class TestConfigLoading:
    def test_load_sweep_sizes_from_yaml(self, tmp_path: Path):
        cfg = tmp_path / "evals.yaml"
        cfg.write_text(yaml.dump({"evals": {"chunk_size_sweep": {"sizes": [128, 256]}}}))
        assert load_sweep_sizes(cfg) == [128, 256]

    def test_load_sweep_sizes_defaults_when_missing(self, tmp_path: Path):
        assert load_sweep_sizes(tmp_path / "missing.yaml") == [256, 500, 768, 1024]

    def test_load_sweep_weights_from_yaml(self, tmp_path: Path):
        cfg = tmp_path / "evals.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "evals": {
                        "chunk_size_sweep": {
                            "weights": {
                                "recall": 0.5,
                                "faithfulness": 0.5,
                                "relevance": 0.0,
                                "latency": 0.0,
                            }
                        }
                    }
                }
            )
        )
        weights = load_sweep_weights(cfg)
        assert weights.recall == pytest.approx(0.5)
        assert weights.faithfulness == pytest.approx(0.5)

    def test_load_sweep_weights_defaults(self, tmp_path: Path):
        weights = load_sweep_weights(tmp_path / "missing.yaml")
        total = weights.recall + weights.faithfulness + weights.relevance + weights.latency
        assert total == pytest.approx(1.0)

    def test_collection_name_for_size(self):
        assert collection_name_for_size(500) == "rag_documents_cs500"

    def test_build_chunk_size_overrides(self):
        overrides = build_chunk_size_overrides(768)
        assert overrides["CHUNKING__CHUNK_SIZE"] == "768"
        assert overrides["QDRANT__COLLECTION"] == "rag_documents_cs768"

    def test_cache_paths(self, tmp_path: Path):
        assert chunk_cache_path(256, tmp_path).name == "chunks.json"
        assert bm25_cache_path(256, tmp_path).name == "bm25_index.json"


# ── chunk cache ────────────────────────────────────────────────────────────────


class TestChunkCache:
    def test_save_and_load_cache(self, tmp_path: Path):
        chunks = [_chunk("c1"), _chunk("c2", text="Other.")]
        path = save_chunk_cache(chunks, 256, tmp_path)
        assert path.exists()
        loaded = load_chunk_cache(256, tmp_path)
        assert loaded is not None
        assert [c.id for c in loaded] == ["c1", "c2"]
        assert loaded[0].embedding is None

    def test_load_missing_cache(self, tmp_path: Path):
        assert load_chunk_cache(999, tmp_path) is None

    def test_load_invalid_cache(self, tmp_path: Path):
        path = chunk_cache_path(256, tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("not json")
        assert load_chunk_cache(256, tmp_path) is None

    def test_load_non_list_cache(self, tmp_path: Path):
        path = chunk_cache_path(256, tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text('{"a": 1}')
        assert load_chunk_cache(256, tmp_path) is None

    def test_load_invalid_chunk_entries(self, tmp_path: Path):
        path = chunk_cache_path(256, tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([{"id": "c1"}, {"not": "a chunk"}]))
        assert load_chunk_cache(256, tmp_path) is None


# ── remapping / scoring ────────────────────────────────────────────────────────


class TestRemapAndScore:
    def test_remap_uses_existing_ids(self):
        pair = _qa(relevant=["c0", "missing"])
        assert remap_relevant_chunks(pair, {"c0", "c1"}, {"c0": _chunk()}) == ["c0"]

    def test_remap_by_answer_overlap(self):
        pair = _qa(answer="EKS service", relevant=["old_id"])
        chunks = {"c9": _chunk("c9", text="Amazon EKS service manages clusters.")}
        assert remap_relevant_chunks(pair, {"c9"}, chunks) == ["c9"]

    def test_remap_empty_when_no_match(self):
        pair = _qa(answer="Unrelated", relevant=["old"])
        assert remap_relevant_chunks(pair, {"c0"}, {"c0": _chunk(text="Different")}) == []

    def test_remap_empty_answer(self):
        pair: dict[str, object] = {
            "question": "Q?",
            "answer": "  ",
            "relevant_chunks": ["missing"],
        }
        assert remap_relevant_chunks(pair, {"c0"}, {"c0": _chunk()}) == []

    def test_compute_weighted_score(self):
        result = ChunkSizeResult(
            chunk_size=256,
            total_samples=1,
            mean_recall_at_5=0.8,
            mean_faithfulness=0.9,
            mean_relevance=0.7,
            mean_latency_ms=100.0,
        )
        weights = SweepWeights(0.25, 0.25, 0.25, 0.25)
        score = compute_weighted_score(result, weights, latency_scores={256: 1.0})
        assert score == pytest.approx(0.85)

    def test_compute_weighted_score_error_is_zero(self):
        result = ChunkSizeResult(
            chunk_size=256,
            total_samples=0,
            mean_recall_at_5=0.0,
            mean_faithfulness=0.0,
            mean_relevance=0.0,
            mean_latency_ms=0.0,
            error="boom",
        )
        score = compute_weighted_score(result, SweepWeights(1, 1, 1, 1), latency_scores={256: 1.0})
        assert score == 0.0

    def test_recommend_size(self):
        results = [
            ChunkSizeResult(256, 1, 0.5, 0.8, 0.7, 120.0),
            ChunkSizeResult(500, 1, 0.9, 0.85, 0.8, 80.0),
        ]
        assert recommend_size(results, SweepWeights(0.35, 0.35, 0.20, 0.10)) == 500
        assert results[1].weighted_score > results[0].weighted_score

    def test_recommend_size_all_errors(self):
        results = [
            ChunkSizeResult(256, 0, 0.0, 0.0, 0.0, 0.0, error="fail"),
        ]
        assert recommend_size(results, SweepWeights(1, 1, 1, 1)) is None

    def test_sweep_weights_normalized_zero_total(self):
        assert SweepWeights(0, 0, 0, 0).normalized().recall == pytest.approx(0.25)


# ── sweep plan / report ────────────────────────────────────────────────────────


class TestSweepPlanAndReport:
    def test_build_sweep_plan_with_cache(self, tmp_path: Path):
        save_chunk_cache([_chunk()], 256, tmp_path)
        plan = build_sweep_plan([256], cache_dir=tmp_path)
        assert plan[0].action == "load cache + index"

    def test_build_sweep_plan_with_source(self, tmp_path: Path):
        plan = build_sweep_plan([256], ingest_source=Path("data/raw"), cache_dir=tmp_path)
        assert plan[0].action == "chunk source + cache + index"

    def test_build_sweep_plan_missing_cache(self, tmp_path: Path):
        plan = build_sweep_plan([256], cache_dir=tmp_path)
        assert "missing cache" in plan[0].action

    def test_report_save_and_summary(self, tmp_path: Path):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[256, 500],
            results=[
                ChunkSizeResult(256, 1, 0.5, 0.8, 0.7, 50.0, weighted_score=0.6),
                ChunkSizeResult(500, 1, 0.7, 0.85, 0.75, 40.0, weighted_score=0.8),
            ],
            recommended_size=500,
        )
        out = tmp_path / "report.json"
        report.save(out)
        data = json.loads(out.read_text())
        assert data["recommended_size"] == 500
        assert "Recommended chunk_size: 500" in report.summary()

    def test_report_save_dry_run_plan(self, tmp_path: Path):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[256],
            results=[],
            dry_run=True,
            plan=[
                SweepPlanEntry(
                    chunk_size=256,
                    collection="rag_documents_cs256",
                    cache_path=tmp_path / "chunks.json",
                    action="load cache + index",
                )
            ],
        )
        out = tmp_path / "dry_run.json"
        report.save(out)
        data = json.loads(out.read_text())
        assert data["plan"][0]["cache_path"].endswith("chunks.json")

    def test_report_dry_run_summary(self):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[256],
            results=[],
            dry_run=True,
            plan=[
                SweepPlanEntry(
                    chunk_size=256,
                    collection="rag_documents_cs256",
                    cache_path=Path("/tmp/chunks.json"),
                    action="load cache + index",
                )
            ],
        )
        assert "dry-run" in report.summary().lower()

    def test_report_skipped_summary(self):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[],
            results=[],
            skipped=True,
            skip_reason="no data",
        )
        assert "skipped" in report.summary().lower()

    def test_print_table_dry_run(self, capsys):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[256],
            results=[],
            dry_run=True,
            plan=[
                SweepPlanEntry(
                    chunk_size=256,
                    collection="rag_documents_cs256",
                    cache_path=Path("cache.json"),
                    action="load cache + index",
                )
            ],
        )
        report.print_table()
        assert "256" in capsys.readouterr().out

    def test_print_table_skipped(self, capsys):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[],
            results=[],
            skipped=True,
            skip_reason="placeholder",
        )
        report.print_table()
        assert "skipped" in capsys.readouterr().out.lower()

    def test_print_table_with_results(self, capsys):
        report = ChunkSizeSweepReport(
            timestamp="T",
            sizes=[256],
            results=[
                ChunkSizeResult(256, 1, 0.5, 0.8, 0.7, 50.0, weighted_score=0.7),
                ChunkSizeResult(500, 0, 0.0, 0.0, 0.0, 0.0, error="fail"),
            ],
            recommended_size=256,
        )
        report.print_table()
        out = capsys.readouterr().out
        assert "256" in out
        assert "Recommended chunk_size: 256" in out


# ── source iteration / chunking ────────────────────────────────────────────────


class TestSourceChunking:
    def test_iter_source_files_file(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("# Hi")
        assert iter_source_files(md) == [md]

    def test_iter_source_files_unsupported(self, tmp_path: Path):
        bad = tmp_path / "doc.txt"
        bad.write_text("nope")
        assert iter_source_files(bad) == []

    def test_iter_source_files_directory(self, tmp_path: Path):
        md = tmp_path / "a.md"
        md.write_text("# A")
        txt = tmp_path / "b.txt"
        txt.write_text("skip")
        assert iter_source_files(tmp_path) == [md]

    def test_chunk_documents_from_source(self, tmp_path: Path):
        md = tmp_path / "doc.md"
        md.write_text("Paragraph one.\n\nParagraph two with more content here.")

        with (
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
            patch("src.core.settings.settings") as mock_settings,
        ):
            mock_settings.chunking.strategy = "recursive"
            mock_settings.chunking.contextual_headers.enabled = False
            mock_settings.chunking.overlap = 10
            with patch("src.rag.chunking.get_chunker") as mock_get:
                mock_get.return_value.chunk.return_value = [_chunk()]
                chunks = chunk_documents_from_source(tmp_path, 256)
        assert len(chunks) == 1

    def test_chunk_documents_empty_source_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No supported documents"):
            chunk_documents_from_source(tmp_path, 256)


# ── indexing helpers ───────────────────────────────────────────────────────────


class TestIndexingHelpers:
    def test_clear_vector_index_uses_recreate_collection(self):
        store = MagicMock()
        clear_vector_index(store)
        store.recreate_collection.assert_called_once()
        store.drop_collection.assert_not_called()

    def test_clear_vector_index_falls_back_to_drop_collection(self):
        store = MagicMock(spec=["drop_collection"])
        clear_vector_index(store)
        store.drop_collection.assert_called_once()

    def test_clear_vector_index_propagates_recreate_errors(self):
        from src.core.exceptions import VectorStoreError

        store = MagicMock()
        store.recreate_collection.side_effect = VectorStoreError("purge failed")
        with pytest.raises(VectorStoreError, match="purge failed"):
            clear_vector_index(store)

    def test_clear_vector_index_noop_without_clear_methods(self):
        clear_vector_index(MagicMock(spec=[]))

    def test_embed_chunks(self):
        chunk = _chunk()
        embedder = MagicMock()
        embedder.embed_both.return_value = ([[0.1, 0.2]], [{1: 0.5}])
        with patch(
            "src.infrastructure.embeddings.get_embedding_provider",
            return_value=embedder,
        ):
            embedded = embed_chunks([chunk])
        assert embedded[0].embedding == [0.1, 0.2]
        assert embedded[0].sparse_vector == {1: 0.5}

    def test_embed_chunks_empty(self):
        assert embed_chunks([]) == []

    def test_index_chunks_for_size(self, tmp_path: Path):
        store = MagicMock()
        bm25 = MagicMock()
        embedded = [_chunk()]
        with (
            patch("src.evals.e2e.chunk_size_sweep.embed_chunks", return_value=embedded),
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
            patch(
                "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings",
                return_value=store,
            ),
            patch("src.infrastructure.vectordb.bm25.BM25Index", return_value=bm25),
        ):
            out_store, out_bm25, out_chunks = index_chunks_for_size(
                256, [_chunk()], cache_dir=tmp_path
            )
        store.recreate_collection.assert_called_once()
        store.upsert.assert_called_once_with(embedded)
        bm25.index.assert_called_once_with(embedded)
        bm25.save.assert_called_once()
        assert out_store is store
        assert out_bm25 is bm25
        assert out_chunks == embedded

    def test_index_chunks_for_size_aborts_when_clear_fails(self, tmp_path: Path):
        from src.core.exceptions import VectorStoreError

        store = MagicMock()
        store.recreate_collection.side_effect = VectorStoreError("clear failed")
        bm25 = MagicMock()
        embedded = [_chunk()]
        with (
            patch("src.evals.e2e.chunk_size_sweep.embed_chunks", return_value=embedded),
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
            patch(
                "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings",
                return_value=store,
            ),
            patch("src.infrastructure.vectordb.bm25.BM25Index", return_value=bm25),
            pytest.raises(VectorStoreError, match="clear failed"),
        ):
            index_chunks_for_size(256, [_chunk()], cache_dir=tmp_path)
        store.upsert.assert_not_called()
        bm25.index.assert_not_called()

    def test_build_sweep_pipeline(self):
        store = MagicMock()
        bm25 = MagicMock()
        with patch("src.rag.pipelines.chat_pipeline.ChatPipeline.from_settings") as mock:
            mock.return_value = MagicMock()
            pipeline = build_sweep_pipeline(vector_store=store, bm25_index=bm25)
        mock.assert_called_once_with(bm25_index=bm25, vector_store=store)
        assert pipeline is mock.return_value


# ── ChunkSizeSweep.run ─────────────────────────────────────────────────────────


class TestChunkSizeSweepRun:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        report = await _benchmark().run([], [256, 500], dry_run=True)
        assert report.dry_run is True
        assert len(report.plan) == 2

    @pytest.mark.asyncio
    async def test_skips_placeholder_dataset(self):
        report = await _benchmark().run([], [256])
        assert report.skipped is True

    @pytest.mark.asyncio
    async def test_runs_size_with_cache(self, tmp_path: Path):
        save_chunk_cache([_chunk("c0")], 256, tmp_path)
        pipeline = pipeline_mock(sources=["c0"])
        store = MagicMock()
        bm25 = MagicMock()

        def fake_prepare(*_args, **_kwargs):
            return [_chunk("c0")]

        sweep = _benchmark()
        with (
            patch.object(sweep, "_prepare_chunks", side_effect=fake_prepare),
            patch(
                "src.evals.e2e.chunk_size_sweep.index_chunks_for_size",
                return_value=(store, bm25, [_chunk("c0")]),
            ),
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
        ):
            factory = MagicMock(return_value=pipeline)
            report = await sweep.run(
                [_qa(relevant=["c0"], answer="EKS")],
                [256],
                cache_dir=tmp_path,
                pipeline_factory=factory,
            )

        assert report.skipped is False
        assert len(report.results) == 1
        assert report.results[0].mean_recall_at_5 == pytest.approx(1.0)
        assert report.recommended_size == 256

    @pytest.mark.asyncio
    async def test_size_failure_recorded_as_error(self, tmp_path: Path):
        sweep = _benchmark()
        with patch.object(
            sweep,
            "_prepare_chunks",
            side_effect=RuntimeError("chunk failed"),
        ):
            report = await sweep.run([_qa()], [256], cache_dir=tmp_path)

        assert report.results[0].error == "chunk failed"

    @pytest.mark.asyncio
    async def test_index_clear_failure_recorded_as_error(self, tmp_path: Path):
        from src.core.exceptions import VectorStoreError

        sweep = _benchmark()
        with (
            patch.object(sweep, "_prepare_chunks", return_value=[_chunk()]),
            patch(
                "src.evals.e2e.chunk_size_sweep.index_chunks_for_size",
                side_effect=VectorStoreError("clear failed"),
            ),
        ):
            report = await sweep.run([_qa()], [256], cache_dir=tmp_path)

        assert report.results[0].error == "clear failed"

    @pytest.mark.asyncio
    async def test_pipeline_failure_recorded_as_zero(self, tmp_path: Path):
        pipeline = pipeline_mock(fail=True)
        store = MagicMock()
        bm25 = MagicMock()
        sweep = _benchmark()

        with (
            patch.object(sweep, "_prepare_chunks", return_value=[_chunk()]),
            patch(
                "src.evals.e2e.chunk_size_sweep.index_chunks_for_size",
                return_value=(store, bm25, [_chunk()]),
            ),
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
        ):
            report = await sweep.run(
                [_qa()],
                [256],
                cache_dir=tmp_path,
                pipeline_factory=MagicMock(return_value=pipeline),
            )

        assert report.results[0].mean_recall_at_5 == pytest.approx(0.0)

    def test_prepare_chunks_uses_cache_when_present(self, tmp_path: Path):
        save_chunk_cache([_chunk("cached")], 256, tmp_path)
        sweep = _benchmark()
        chunks = sweep._prepare_chunks(
            256,
            ingest_source=None,
            cache_dir=tmp_path,
            force_rechunk=False,
        )
        assert chunks[0].id == "cached"

    def test_prepare_chunks_from_source(self, tmp_path: Path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "doc.md").write_text("# Doc")

        sweep = _benchmark()
        with patch(
            "src.evals.e2e.chunk_size_sweep.chunk_documents_from_source",
            return_value=[_chunk()],
        ) as mock_chunk:
            chunks = sweep._prepare_chunks(
                256,
                ingest_source=source,
                cache_dir=tmp_path,
                force_rechunk=True,
            )
        mock_chunk.assert_called_once_with(source, 256)
        assert len(chunks) == 1
        assert load_chunk_cache(256, tmp_path) is not None

    def test_prepare_chunks_missing_cache_raises(self, tmp_path: Path):
        sweep = _benchmark()
        with pytest.raises(ValueError, match="No chunk cache"):
            sweep._prepare_chunks(
                256,
                ingest_source=None,
                cache_dir=tmp_path,
                force_rechunk=False,
            )

    def test_prepare_chunks_empty_after_chunking_raises(self, tmp_path: Path):
        source = tmp_path / "src"
        source.mkdir()
        sweep = _benchmark()
        with (
            patch(
                "src.evals.e2e.chunk_size_sweep.chunk_documents_from_source",
                return_value=[],
            ),
            pytest.raises(ValueError, match="produced no chunks"),
        ):
            sweep._prepare_chunks(
                256,
                ingest_source=source,
                cache_dir=tmp_path,
                force_rechunk=True,
            )

    @pytest.mark.asyncio
    async def test_skips_empty_question_in_evaluate(self, tmp_path: Path):
        pipeline = pipeline_mock()
        store = MagicMock()
        bm25 = MagicMock()
        sweep = _benchmark()

        with (
            patch.object(sweep, "_prepare_chunks", return_value=[_chunk()]),
            patch(
                "src.evals.e2e.chunk_size_sweep.index_chunks_for_size",
                return_value=(store, bm25, [_chunk()]),
            ),
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
        ):
            report = await sweep.run(
                [_qa(), {"question": "", "relevant_chunks": ["c0"]}],
                [256],
                cache_dir=tmp_path,
                pipeline_factory=MagicMock(return_value=pipeline),
            )

        assert report.results[0].total_samples == 1
        pipeline.benchmark.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_answer_without_latency_uses_elapsed(self, tmp_path: Path):
        pipeline = MagicMock()
        answer = MagicMock()
        answer.text = "A"
        answer.sources = ["c0"]
        answer.latency_ms = None
        pipeline.benchmark = AsyncMock(
            return_value=BenchmarkRun(answer=answer, context_texts=["ctx"])
        )
        store = MagicMock()
        bm25 = MagicMock()
        sweep = _benchmark()

        with (
            patch.object(sweep, "_prepare_chunks", return_value=[_chunk()]),
            patch(
                "src.evals.e2e.chunk_size_sweep.index_chunks_for_size",
                return_value=(store, bm25, [_chunk()]),
            ),
            patch(
                "src.evals.e2e.chunk_size_sweep.temporary_config",
                return_value=_noop_context(),
            ),
        ):
            report = await sweep.run(
                [_qa(relevant=["c0"])],
                [256],
                cache_dir=tmp_path,
                pipeline_factory=MagicMock(return_value=pipeline),
            )

        assert report.results[0].mean_latency_ms >= 0.0


class TestRunChunkSizeSweep:
    def test_run_chunk_size_sweep_saves_report(self, tmp_path: Path):
        async def _fake_run(*_args, **_kwargs):
            return ChunkSizeSweepReport(
                timestamp="T",
                sizes=[256],
                results=[
                    ChunkSizeResult(256, 1, 1.0, 0.9, 0.85, 10.0, weighted_score=0.9),
                ],
                recommended_size=256,
            )

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_fake_run)

        report = run_chunk_size_sweep([256], [_qa()], output_dir=tmp_path, sweep=runner)
        assert report.skipped is False
        assert list(tmp_path.glob("chunk_size_sweep_*.json"))

    def test_skipped_sweep_does_not_save(self, tmp_path: Path):
        async def _fake_run(*_args, **_kwargs):
            return ChunkSizeSweepReport(
                timestamp="T",
                sizes=[],
                results=[],
                skipped=True,
                skip_reason="placeholder",
            )

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_fake_run)

        report = run_chunk_size_sweep([], [], output_dir=tmp_path, sweep=runner)
        assert report.skipped is True
        assert not list(tmp_path.glob("chunk_size_sweep_*.json"))

    def test_dry_run_does_not_save(self, tmp_path: Path):
        async def _fake_run(*_args, **_kwargs):
            return ChunkSizeSweepReport(
                timestamp="T",
                sizes=[256],
                results=[],
                dry_run=True,
            )

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_fake_run)

        report = run_chunk_size_sweep([256], [], output_dir=tmp_path, sweep=runner, dry_run=True)
        assert report.dry_run is True
        assert not list(tmp_path.glob("chunk_size_sweep_*.json"))


def _noop_context():
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield

    return _cm()
