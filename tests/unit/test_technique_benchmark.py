"""T-150 — TechniqueBenchmark unit tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.domain.entities.answer import Answer
from src.evals.e2e.technique_benchmark import (
    FeedbackComparison,
    TechniqueBenchmark,
    TechniqueBenchmarkReport,
    TechniqueConfig,
    TechniqueResult,
    build_benchmark_pipeline,
    build_feedback_boost_overrides,
    filter_qa_pairs,
    has_real_qa_data,
    is_placeholder_qa_pair,
    load_qa_pairs,
    load_technique_configs,
    merge_technique_overrides,
    pre_seed_feedback_scores,
    prepare_qa_pairs,
    reload_settings_module,
    run_technique_matrix,
    temporary_config,
)

_DEFAULT_TECHNIQUE_NAMES = (
    "baseline",
    "multi_query",
    "hyde",
    "cch",
    "reliable_rag",
    "self_rag",
    "feedback_loop",
)


def _qa(
    question: str = "What is EKS?",
    answer: str = "Kubernetes on AWS.",
    relevant: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": relevant or ["c0", "c1"],
    }


def _pipeline_mock(
    *,
    sources: list[str] | None = None,
    text: str = "Answer.",
    context: list[str] | None = None,
    latency_ms: float = 42.0,
    fail: bool = False,
) -> MagicMock:
    pipeline = MagicMock()
    if fail:
        pipeline.benchmark = AsyncMock(side_effect=RuntimeError("pipeline down"))
    else:
        answer = Answer(
            query_id="q1",
            text=text,
            sources=sources if sources is not None else ["c0"],
            latency_ms=latency_ms,
        )
        pipeline.benchmark = AsyncMock(return_value=(answer, context or ["ctx"]))
    return pipeline


def _metric_mock(score: float) -> MagicMock:
    mock = MagicMock()
    result = MagicMock()
    result.score = score
    mock.score.return_value = result
    return mock


def _benchmark(faith: float = 0.9, relev: float = 0.85) -> TechniqueBenchmark:
    return TechniqueBenchmark(
        faithfulness=_metric_mock(faith),
        relevance=_metric_mock(relev),
    )


# ── placeholder / QA loading ───────────────────────────────────────────────────


class TestPlaceholderHelpers:
    def test_is_placeholder_true(self):
        assert is_placeholder_qa_pair({"relevant_chunks": ["chunk_id_1", "chunk_id_2"]})

    def test_is_placeholder_false_real_ids(self):
        assert not is_placeholder_qa_pair({"relevant_chunks": ["c0", "chunk_id_1"]})

    def test_is_placeholder_false_empty(self):
        assert not is_placeholder_qa_pair({"relevant_chunks": []})

    def test_is_placeholder_false_missing(self):
        assert not is_placeholder_qa_pair({})

    def test_filter_qa_pairs(self):
        pairs: list[dict[str, object]] = [
            _qa(),
            {"question": "", "relevant_chunks": ["c0"]},
            {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
        ]
        filtered = filter_qa_pairs(pairs)
        assert len(filtered) == 1

    def test_prepare_qa_pairs_filters_before_max_samples(self):
        pairs: list[dict[str, object]] = [
            {"question": "Placeholder 1?", "relevant_chunks": ["chunk_id_1"]},
            {"question": "Placeholder 2?", "relevant_chunks": ["chunk_id_2"]},
            _qa(question="Real 1?", relevant=["c0"]),
            _qa(question="Real 2?", relevant=["c1"]),
            _qa(question="Real 3?", relevant=["c2"]),
        ]
        prepared = prepare_qa_pairs(pairs, max_samples=2)
        assert len(prepared) == 2
        assert prepared[0]["question"] == "Real 1?"
        assert prepared[1]["question"] == "Real 2?"

    def test_prepare_qa_pairs_no_cap(self):
        pairs: list[dict[str, object]] = [
            _qa(),
            {"question": "Placeholder?", "relevant_chunks": ["chunk_id_1"]},
        ]
        assert len(prepare_qa_pairs(pairs)) == 1

    def test_has_real_qa_data(self):
        assert has_real_qa_data([_qa()])
        assert not has_real_qa_data([])

    def test_load_qa_pairs_from_file(self, tmp_path: Path):
        path = tmp_path / "qa.json"
        path.write_text(json.dumps([_qa(), {"question": "P?", "relevant_chunks": ["chunk_id_x"]}]))
        assert len(load_qa_pairs(path)) == 1

    def test_load_qa_pairs_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        assert load_qa_pairs(path) == []

    def test_load_qa_pairs_non_list(self, tmp_path: Path):
        path = tmp_path / "obj.json"
        path.write_text('{"a": 1}')
        assert load_qa_pairs(path) == []


# ── config overrides ───────────────────────────────────────────────────────────


class TestMergeTechniqueOverrides:
    def test_baseline_disables_all(self):
        overrides = merge_technique_overrides("baseline")
        assert overrides["QUERY_EXPANSION__ENABLED"] == "false"
        assert overrides["COMPRESSION__ENABLED"] == "false"

    def test_multi_query_enables_expansion(self):
        overrides = merge_technique_overrides("multi_query")
        assert overrides["QUERY_EXPANSION__ENABLED"] == "true"

    def test_hyde_override(self):
        assert merge_technique_overrides("hyde")["RETRIEVAL__HYDE__ENABLED"] == "true"

    def test_extra_overrides_merged(self):
        overrides = merge_technique_overrides("baseline", {"FOO": "bar"})
        assert overrides["FOO"] == "bar"

    def test_feedback_loop_isolates_boost_from_pool_size(self):
        overrides = merge_technique_overrides("feedback_loop")
        assert overrides["QUALITY__FEEDBACK_LOOP__EXPAND_CANDIDATE_POOL"] == "false"

    def test_build_feedback_boost_overrides(self):
        off = build_feedback_boost_overrides(boost_enabled=False)
        on = build_feedback_boost_overrides(boost_enabled=True)
        assert off["QUALITY__FEEDBACK_LOOP__ENABLED"] == "false"
        assert on["QUALITY__FEEDBACK_LOOP__ENABLED"] == "true"
        assert off["QUALITY__FEEDBACK_LOOP__EXPAND_CANDIDATE_POOL"] == "false"
        assert on["QUALITY__FEEDBACK_LOOP__EXPAND_CANDIDATE_POOL"] == "false"
        assert off["QUERY_EXPANSION__ENABLED"] == "false"


class TestLoadTechniqueConfigs:
    def test_loads_from_yaml(self, tmp_path: Path):
        cfg = tmp_path / "evals.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "evals": {
                        "technique_benchmark": {
                            "configs": [
                                {
                                    "name": "custom",
                                    "description": "Custom technique",
                                    "overrides": {"COMPRESSION__ENABLED": "true"},
                                }
                            ]
                        }
                    }
                }
            )
        )
        configs = load_technique_configs(cfg)
        assert len(configs) == 1
        assert configs[0].name == "custom"
        assert configs[0].overrides["COMPRESSION__ENABLED"] == "true"

    def test_falls_back_when_yaml_missing(self, tmp_path: Path):
        configs = load_technique_configs(tmp_path / "missing.yaml")
        assert [c.name for c in configs] == list(_DEFAULT_TECHNIQUE_NAMES)

    def test_skips_entries_without_name(self, tmp_path: Path):
        cfg = tmp_path / "evals.yaml"
        cfg.write_text(
            yaml.dump({"evals": {"technique_benchmark": {"configs": [{"description": "no name"}]}}})
        )
        assert [c.name for c in load_technique_configs(cfg)] == list(_DEFAULT_TECHNIQUE_NAMES)


class TestTemporaryConfig:
    def test_applies_and_restores_env(self):
        key = "TECH_BENCH_TEST_KEY"
        os.environ.pop(key, None)
        with temporary_config({key: "on"}):
            assert os.environ[key] == "on"
        assert key not in os.environ

    def test_restores_previous_value(self):
        key = "TECH_BENCH_TEST_KEY"
        os.environ[key] = "original"
        with temporary_config({key: "temporary"}):
            assert os.environ[key] == "temporary"
        assert os.environ[key] == "original"

    def test_reload_settings_module(self):
        reload_settings_module()

    def test_reload_settings_refreshes_retrieval_pipeline_bindings(self):
        import src.rag.pipelines.retrieval_pipeline as retrieval_pipeline

        key = "QUERY_EXPANSION__ENABLED"
        with temporary_config({key: "true"}):
            assert retrieval_pipeline._settings().query_expansion.enabled is True
        with temporary_config({key: "false"}):
            assert retrieval_pipeline._settings().query_expansion.enabled is False

    def test_sequential_overrides_apply_to_pipeline_factory(self):
        """Each technique must see its own env overrides when building the pipeline."""
        from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline

        seen: list[bool] = []

        def _capture_expansion(*_args, **_kwargs):
            import src.rag.pipelines.retrieval_pipeline as rp

            seen.append(rp._settings().query_expansion.enabled)
            raise RuntimeError("stop after settings read")

        with patch.object(RetrievalPipeline, "from_settings", side_effect=_capture_expansion):
            with (
                temporary_config({"QUERY_EXPANSION__ENABLED": "false"}),
                pytest.raises(RuntimeError),
            ):
                build_benchmark_pipeline(self_rag=False)
            with (
                temporary_config({"QUERY_EXPANSION__ENABLED": "true"}),
                pytest.raises(RuntimeError),
            ):
                build_benchmark_pipeline(self_rag=False)

        assert seen == [False, True]


# ── dataclasses ────────────────────────────────────────────────────────────────


class TestTechniqueResult:
    def test_to_dict(self):
        result = TechniqueResult(
            technique="baseline",
            total_samples=1,
            mean_recall_at_5=0.5,
            mean_faithfulness=0.8,
            mean_relevance=0.75,
            mean_latency_ms=100.0,
        )
        data = result.to_dict()
        assert data["technique"] == "baseline"


class TestTechniqueBenchmarkReport:
    def test_save_and_summary(self, tmp_path: Path):
        report = TechniqueBenchmarkReport(
            timestamp="T",
            techniques=["baseline"],
            results=[
                TechniqueResult(
                    technique="baseline",
                    total_samples=2,
                    mean_recall_at_5=0.5,
                    mean_faithfulness=0.8,
                    mean_relevance=0.75,
                    mean_latency_ms=50.0,
                )
            ],
            feedback_comparison=FeedbackComparison(
                recall_boost_off=0.4,
                recall_boost_on=0.6,
                samples=2,
            ),
        )
        out = tmp_path / "report.json"
        report.save(out)
        data = json.loads(out.read_text())
        assert "feedback_comparison" in data
        assert "Technique Benchmark" in report.summary()
        assert "Feedback loop" in report.summary()

    def test_skipped_summary(self):
        report = TechniqueBenchmarkReport(
            timestamp="T",
            techniques=[],
            results=[],
            skipped=True,
            skip_reason="no data",
        )
        assert "skipped" in report.summary().lower()

    def test_print_table_skipped(self, capsys):
        report = TechniqueBenchmarkReport(
            timestamp="T",
            techniques=[],
            results=[],
            skipped=True,
            skip_reason="placeholder",
        )
        report.print_table()
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()

    def test_print_table_with_results(self, capsys):
        report = TechniqueBenchmarkReport(
            timestamp="T",
            techniques=["baseline"],
            results=[
                TechniqueResult(
                    technique="baseline",
                    total_samples=1,
                    mean_recall_at_5=0.5,
                    mean_faithfulness=0.8,
                    mean_relevance=0.75,
                    mean_latency_ms=50.0,
                ),
                TechniqueResult(
                    technique="broken",
                    total_samples=0,
                    mean_recall_at_5=0.0,
                    mean_faithfulness=0.0,
                    mean_relevance=0.0,
                    mean_latency_ms=0.0,
                    error="boom",
                ),
            ],
        )
        report.print_table()
        assert "baseline" in capsys.readouterr().out


# ── pre_seed_feedback ──────────────────────────────────────────────────────────


class TestPreSeedFeedback:
    def test_seeds_unique_chunk_ids(self):
        store = MagicMock()
        count = pre_seed_feedback_scores([_qa(relevant=["c0", "c1", "c0"])], vector_store=store)
        assert count == 2
        assert store.accumulate_feedback_score.call_count == 2

    def test_skips_on_vector_store_error(self):
        from src.core.exceptions import VectorStoreError

        store = MagicMock()
        store.accumulate_feedback_score.side_effect = VectorStoreError("missing")
        count = pre_seed_feedback_scores([_qa(relevant=["missing"])], vector_store=store)
        assert count == 0

    def test_resolves_vector_store_from_qdrant(self):
        store = MagicMock()
        with patch(
            "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings",
            return_value=store,
        ):
            count = pre_seed_feedback_scores([_qa(relevant=["c0"])])
        assert count == 1

    def test_skips_non_list_relevant_chunks(self):
        store = MagicMock()
        pairs: list[dict[str, object]] = [
            {"question": "Q?", "relevant_chunks": "not-a-list"},
        ]
        count = pre_seed_feedback_scores(pairs, vector_store=store)
        assert count == 0
        store.accumulate_feedback_score.assert_not_called()


# ── pipeline adapters ──────────────────────────────────────────────────────────


class TestBuildBenchmarkPipeline:
    def test_chat_pipeline_by_default(self):
        with patch("src.rag.pipelines.chat_pipeline.ChatPipeline.from_settings") as mock:
            mock.return_value = MagicMock()
            pipeline = build_benchmark_pipeline(self_rag=False)
        assert pipeline is mock.return_value

    def test_agent_pipeline_for_self_rag(self):
        with (
            patch("src.rag.pipelines.agent_pipeline.AgentPipeline.from_settings") as mock_agent,
            patch("src.rag.pipelines.chat_pipeline.ChatPipeline.from_settings") as mock_chat,
        ):
            agent = MagicMock()
            mock_agent.return_value = agent
            pipeline = build_benchmark_pipeline(self_rag=True)
        mock_agent.assert_called_once()
        mock_chat.assert_not_called()
        assert pipeline is not agent

    @pytest.mark.asyncio
    async def test_self_rag_benchmark_delegates_to_chat_full(self):
        agent = MagicMock()
        answer = Answer(query_id="q", text="Hi", sources=["c0"])
        agent.chat_full = AsyncMock(
            return_value=MagicMock(
                answer=answer,
                context_texts=["EKS runs on AWS."],
                iterations=1,
                actions=[],
                self_rag_decisions=[],
            )
        )
        with patch(
            "src.rag.pipelines.agent_pipeline.AgentPipeline.from_settings",
            return_value=agent,
        ):
            pipeline = build_benchmark_pipeline(self_rag=True)
        result_answer, contexts = await pipeline.benchmark("question?")
        assert result_answer is answer
        assert contexts == ["EKS runs on AWS."]
        agent.chat_full.assert_awaited_once_with("question?")


# ── TechniqueBenchmark.run ─────────────────────────────────────────────────────


class TestTechniqueBenchmarkRun:
    @pytest.mark.asyncio
    async def test_skips_placeholder_dataset(self):
        report = await _benchmark().run([], ["baseline"])
        assert report.skipped is True
        assert report.results == []

    @pytest.mark.asyncio
    async def test_runs_baseline_technique(self):
        pipeline = _pipeline_mock(sources=["c0", "c1"])
        factory = MagicMock(return_value=pipeline)

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ):
            report = await _benchmark().run(
                [_qa(relevant=["c0"])],
                ["baseline"],
                pipeline_factory=factory,
            )

        assert len(report.results) == 1
        assert report.results[0].technique == "baseline"
        assert report.results[0].mean_recall_at_5 == pytest.approx(1.0)
        assert report.results[0].mean_faithfulness == pytest.approx(0.9)
        assert report.results[0].mean_latency_ms == pytest.approx(42.0)

    @pytest.mark.asyncio
    async def test_pipeline_failure_recorded_as_zero(self):
        pipeline = _pipeline_mock(fail=True)
        factory = MagicMock(return_value=pipeline)

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ):
            report = await _benchmark().run([_qa()], ["baseline"], pipeline_factory=factory)

        assert report.results[0].mean_recall_at_5 == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_technique_factory_exception_sets_error(self):
        factory = MagicMock(side_effect=RuntimeError("factory failed"))

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ):
            report = await _benchmark().run([_qa()], ["hyde"], pipeline_factory=factory)

        assert report.results[0].error == "factory failed"

    @pytest.mark.asyncio
    async def test_feedback_loop_comparison(self):
        pipeline = _pipeline_mock(sources=["c0"])
        factory = MagicMock(return_value=pipeline)

        with (
            patch(
                "src.evals.e2e.technique_benchmark.temporary_config",
                side_effect=lambda *_args, **_kwargs: _noop_context(),
            ) as mock_cfg,
            patch("src.evals.e2e.technique_benchmark.pre_seed_feedback_scores") as mock_seed,
        ):
            report = await _benchmark().run(
                [_qa(relevant=["c0"])],
                ["feedback_loop"],
                pipeline_factory=factory,
            )

        mock_seed.assert_called_once()
        assert report.feedback_comparison is not None
        assert report.results[0].technique == "feedback_loop"
        assert len(mock_cfg.call_args_list) == 2
        for call in mock_cfg.call_args_list:
            applied = call.args[0]
            assert applied["QUALITY__FEEDBACK_LOOP__EXPAND_CANDIDATE_POOL"] == "false"
        assert mock_cfg.call_args_list[0].args[0]["QUALITY__FEEDBACK_LOOP__ENABLED"] == "false"
        assert mock_cfg.call_args_list[1].args[0]["QUALITY__FEEDBACK_LOOP__ENABLED"] == "true"

    @pytest.mark.asyncio
    async def test_feedback_loop_factory_exception_sets_error(self):
        factory = MagicMock(side_effect=RuntimeError("feedback factory failed"))

        with (
            patch(
                "src.evals.e2e.technique_benchmark.temporary_config",
                side_effect=lambda *_args, **_kwargs: _noop_context(),
            ),
            patch("src.evals.e2e.technique_benchmark.pre_seed_feedback_scores"),
        ):
            report = await _benchmark().run(
                [_qa(relevant=["c0"])],
                ["feedback_loop"],
                pipeline_factory=factory,
            )

        assert report.feedback_comparison is None
        assert len(report.results) == 1
        assert report.results[0].technique == "feedback_loop"
        assert report.results[0].error == "feedback factory failed"

    @pytest.mark.asyncio
    async def test_self_rag_eval_uses_pipeline_context(self):
        pipeline = _pipeline_mock(context=["EKS passage."])
        factory = MagicMock(return_value=pipeline)

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ):
            report = await _benchmark().run([_qa()], ["self_rag"], pipeline_factory=factory)

        assert report.results[0].mean_faithfulness == pytest.approx(0.9)
        assert report.results[0].mean_relevance == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_self_rag_uses_agent_factory(self):
        pipeline = _pipeline_mock()
        factory = MagicMock(return_value=pipeline)

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ):
            await _benchmark().run([_qa()], ["self_rag"], pipeline_factory=factory)

        factory.assert_called_with(self_rag=True)

    @pytest.mark.asyncio
    async def test_unknown_technique_uses_default_overrides(self):
        pipeline = _pipeline_mock()
        factory = MagicMock(return_value=pipeline)

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ) as mock_cfg:
            report = await _benchmark().run([_qa()], ["custom_unknown"], pipeline_factory=factory)

        mock_cfg.assert_called_once()
        assert mock_cfg.call_args.args[0]["QUERY_EXPANSION__ENABLED"] == "false"
        assert report.results[0].technique == "custom_unknown"

    @pytest.mark.asyncio
    async def test_answer_without_latency_uses_elapsed(self):
        pipeline = MagicMock()
        answer = MagicMock()
        answer.text = "A"
        answer.sources = ["c0"]
        answer.latency_ms = None
        pipeline.benchmark = AsyncMock(return_value=(answer, ["ctx"]))
        factory = MagicMock(return_value=pipeline)

        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            side_effect=lambda *_args, **_kwargs: _noop_context(),
        ):
            report = await _benchmark().run(
                [_qa(relevant=["c0"])],
                ["baseline"],
                pipeline_factory=factory,
            )

        assert report.results[0].mean_latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_skips_empty_question_in_evaluate(self):
        pipeline = _pipeline_mock()
        factory = MagicMock(return_value=pipeline)

        qa_pairs: list[dict[str, object]] = [
            _qa(),
            {"question": "", "relevant_chunks": ["c0"]},
        ]
        with patch(
            "src.evals.e2e.technique_benchmark.temporary_config",
            return_value=_noop_context(),
        ):
            report = await _benchmark().run(
                qa_pairs,
                ["baseline"],
                pipeline_factory=factory,
            )

        assert report.results[0].total_samples == 1
        pipeline.benchmark.assert_awaited_once()


class TestRunTechniqueMatrix:
    def test_run_technique_matrix_saves_report(self, tmp_path: Path):
        async def _fake_run(*_args, **_kwargs):
            return TechniqueBenchmarkReport(
                timestamp="T",
                techniques=["baseline"],
                results=[
                    TechniqueResult(
                        technique="baseline",
                        total_samples=1,
                        mean_recall_at_5=1.0,
                        mean_faithfulness=0.9,
                        mean_relevance=0.85,
                        mean_latency_ms=10.0,
                    )
                ],
            )

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_fake_run)

        report = run_technique_matrix(["baseline"], [_qa()], output_dir=tmp_path, benchmark=runner)
        assert report.skipped is False
        assert list(tmp_path.glob("technique_benchmark_*.json"))

    def test_skipped_matrix_does_not_save(self, tmp_path: Path):
        async def _fake_run(*_args, **_kwargs):
            return TechniqueBenchmarkReport(
                timestamp="T",
                techniques=[],
                results=[],
                skipped=True,
                skip_reason="placeholder",
            )

        runner = MagicMock()
        runner.run = AsyncMock(side_effect=_fake_run)

        report = run_technique_matrix([], [], output_dir=tmp_path, benchmark=runner)
        assert report.skipped is True
        assert not list(tmp_path.glob("technique_benchmark_*.json"))


class TestTechniqueConfig:
    def test_frozen_dataclass(self):
        cfg = TechniqueConfig(name="baseline", description="d", overrides={})
        assert cfg.name == "baseline"


def _noop_context():
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield

    return _cm()
