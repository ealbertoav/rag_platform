"""T-172 — InfraBenchmark unit tests."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.domain.entities.chunk import Chunk
from src.evals.e2e.infra_benchmark import (
    BaselineComparisonResult,
    BaselineScenarioFailure,
    InfraBenchmark,
    InfraBenchmarkReport,
    InfraBenchmarkThresholds,
    RegressionWarning,
    ScenarioMetrics,
    build_bm25_fixture_chunks,
    build_default_graph_retriever,
    build_default_pipeline,
    compare_to_baseline,
    default_baseline_path,
    load_infra_baseline,
    load_infra_thresholds,
    measure_stream_inter_token_latencies_ms,
    percentile,
    report_to_baseline_payload,
    save_infra_baseline,
)


async def _token_stream(*tokens: str) -> AsyncIterator[str]:
    for token in tokens:
        yield token


def _scenario(
    name: str,
    *,
    p50: float = 10.0,
    p95: float = 20.0,
    samples: int = 5,
    failures: int = 0,
    memory_bytes: int | None = None,
    error: str = "",
    skipped: bool = False,
    skip_reason: str = "",
) -> ScenarioMetrics:
    return ScenarioMetrics(
        name=name,
        p50_ms=p50,
        p95_ms=p95,
        samples=samples,
        failures=failures,
        memory_bytes=memory_bytes,
        error=error,
        skipped=skipped,
        skip_reason=skip_reason,
    )


def _pipeline_mock(
    *,
    stream_tokens: tuple[str, ...] = ("a", "b", "c"),
    chat_full_error: BaseException | None = None,
) -> MagicMock:
    pipeline = MagicMock()

    async def _chat(_question: str) -> AsyncIterator[str]:
        return _token_stream(*stream_tokens)

    pipeline.chat = AsyncMock(side_effect=_chat)

    if chat_full_error is not None:
        pipeline.chat_full = AsyncMock(side_effect=chat_full_error)
    else:
        pipeline.chat_full = AsyncMock(return_value=MagicMock())
    return pipeline


def _graph_llm_sharing_benchmark(
    *,
    graph_search_iterations: int = 1,
) -> tuple[InfraBenchmark, list[dict[str, object]], MagicMock]:
    """Benchmark wired to record which LLM the graph factory receives."""
    shared_llm = MagicMock()
    pipeline = MagicMock()
    pipeline._llm = shared_llm
    retriever = MagicMock()
    retriever.search = AsyncMock(return_value=[])
    calls: list[dict[str, object]] = []

    async def pipeline_factory():
        return pipeline

    def graph_factory(*, llm=None):
        calls.append({"llm": llm})
        return retriever

    benchmark = InfraBenchmark(
        thresholds=InfraBenchmarkThresholds(graph_search_iterations=graph_search_iterations),
        pipeline_factory=pipeline_factory,
        graph_retriever_factory=graph_factory,
        neo4j_available=lambda: True,
    )
    return benchmark, calls, shared_llm


class TestPercentile:
    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_interpolates(self):
        assert percentile([1.0, 3.0], 50) == 2.0

    def test_unsorted_input(self):
        assert percentile([30.0, 10.0, 20.0], 50) == 20.0


class TestLoadInfraThresholds:
    def test_defaults_when_config_missing(self, tmp_path: Path):
        missing = tmp_path / "missing.yaml"
        thresholds = load_infra_thresholds(missing)
        assert thresholds.concurrent_chat_count == 10
        assert thresholds.bm25_fixture_chunks == 100_000

    def test_loads_from_yaml(self, tmp_path: Path):
        path = tmp_path / "evals.yaml"
        path.write_text(
            yaml.dump(
                {
                    "evals": {
                        "infra_benchmark": {
                            "regression_p95_pct": 15,
                            "concurrent_chat_count": 5,
                            "bm25_fixture_chunks": 500,
                            "baseline_path": "data/exports/custom_baseline.json",
                        }
                    }
                }
            )
        )
        thresholds = load_infra_thresholds(path)
        assert thresholds.regression_p95_pct == 15.0
        assert thresholds.concurrent_chat_count == 5
        assert thresholds.bm25_fixture_chunks == 500
        assert thresholds.baseline_path.name == "custom_baseline.json"

    def test_invalid_section_uses_defaults(self, tmp_path: Path):
        path = tmp_path / "evals.yaml"
        path.write_text(yaml.dump({"evals": {"infra_benchmark": "bad"}}))
        thresholds = load_infra_thresholds(path)
        assert thresholds.regression_p95_pct == 10.0

    def test_coerces_string_numbers(self, tmp_path: Path):
        path = tmp_path / "evals.yaml"
        path.write_text(
            yaml.dump(
                {
                    "evals": {
                        "infra_benchmark": {
                            "concurrent_chat_count": "7",
                            "regression_p95_pct": "12.5",
                        }
                    }
                }
            )
        )
        thresholds = load_infra_thresholds(path)
        assert thresholds.concurrent_chat_count == 7
        assert thresholds.regression_p95_pct == 12.5


class TestBaselineHelpers:
    def test_default_baseline_path(self):
        assert default_baseline_path().name == "infra_baseline.json"

    def test_load_missing_baseline(self, tmp_path: Path):
        assert load_infra_baseline(tmp_path / "missing.json") == {}

    def test_load_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not-json")
        assert load_infra_baseline(path) == {}

    def test_load_non_object_json(self, tmp_path: Path):
        path = tmp_path / "list.json"
        path.write_text("[]")
        assert load_infra_baseline(path) == {}

    def test_report_to_baseline_payload_skips_errors(self):
        report = InfraBenchmarkReport(
            timestamp="ts",
            scenarios=[
                _scenario("ok", p50=1, p95=2, memory_bytes=100),
                _scenario("skip", skipped=True, skip_reason="n/a"),
                _scenario("err", error="boom"),
            ],
        )
        payload = report_to_baseline_payload(report)
        assert set(payload["scenarios"].keys()) == {"ok"}  # type: ignore[index]

    def test_report_save_writes_json(self, tmp_path: Path):
        report = InfraBenchmarkReport(
            timestamp="t",
            scenarios=[_scenario("bm25_100k", memory_bytes=512)],
        )
        path = tmp_path / "nested" / "out.json"
        report.save(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["timestamp"] == "t"
        assert loaded["scenarios"][0]["memory_bytes"] == 512

    def test_scenario_to_dict_without_memory(self):
        data = _scenario("streaming_chat").to_dict()
        assert "memory_bytes" not in data

    def test_save_infra_baseline(self, tmp_path: Path):
        report = InfraBenchmarkReport(
            timestamp="20260101T000000",
            scenarios=[_scenario("bm25_100k", p50=3, p95=6, memory_bytes=2048)],
        )
        path = save_infra_baseline(report, tmp_path / "baseline.json")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["scenarios"]["bm25_100k"]["p95_ms"] == 6.0

    def test_save_infra_baseline_merges_existing_scenarios(self, tmp_path: Path):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "description": "existing",
                    "scenarios": {
                        "streaming_chat": {
                            "p50_ms": 1.0,
                            "p95_ms": 2.0,
                            "samples": 10,
                            "failures": 0,
                        },
                        "bm25_100k": {
                            "p50_ms": 3.0,
                            "p95_ms": 4.0,
                            "samples": 20,
                            "failures": 0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        report = InfraBenchmarkReport(
            timestamp="20260101T000000",
            scenarios=[_scenario("bm25_100k", p50=5, p95=8, memory_bytes=100)],
        )
        save_infra_baseline(report, baseline_path)
        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert loaded["description"] == (
            "T-172 infrastructure latency baseline — "
            "compare with scripts/benchmark_infra.py --compare"
        )
        assert loaded["scenarios"]["streaming_chat"]["p95_ms"] == 2.0
        assert loaded["scenarios"]["bm25_100k"]["p95_ms"] == 8.0


class TestCompareToBaseline:
    def test_no_regression(self):
        current = {"streaming_chat": _scenario("streaming_chat", p95=20.0)}
        baseline = {"scenarios": {"streaming_chat": {"p95_ms": 20.0}}}
        result = compare_to_baseline(current, baseline, regression_p95_pct=10.0)
        assert result == BaselineComparisonResult(regressions=[], failures=[])

    def test_detects_regression(self):
        current = {"streaming_chat": _scenario("streaming_chat", p95=25.0)}
        baseline = {"scenarios": {"streaming_chat": {"p95_ms": 20.0}}}
        result = compare_to_baseline(current, baseline, regression_p95_pct=10.0)
        assert len(result.regressions) == 1
        assert result.regressions[0].pct_change == 25.0

    def test_skips_missing_baseline_scenario(self):
        current = {"bm25_100k": _scenario("bm25_100k", p95=99.0)}
        result = compare_to_baseline(current, {"scenarios": {}}, regression_p95_pct=10.0)
        assert result == BaselineComparisonResult(regressions=[], failures=[])

    def test_skips_invalid_baseline(self):
        current = {"bm25_100k": _scenario("bm25_100k", p95=99.0)}
        result = compare_to_baseline(current, {}, regression_p95_pct=10.0)
        assert result == BaselineComparisonResult(regressions=[], failures=[])

    def test_skips_non_dict_scenarios_section(self):
        current = {"bm25_100k": _scenario("bm25_100k", p95=99.0)}
        result = compare_to_baseline(current, {"scenarios": []}, regression_p95_pct=0.0)
        assert result == BaselineComparisonResult(regressions=[], failures=[])

    def test_skips_non_dict_baseline_object(self):
        current = {"bm25_100k": _scenario("bm25_100k", p95=99.0)}
        result = compare_to_baseline(current, "not-a-dict", regression_p95_pct=0.0)
        assert result == BaselineComparisonResult(regressions=[], failures=[])

    def test_skips_invalid_baseline_p95(self):
        current = {"bm25_100k": _scenario("bm25_100k", p95=99.0)}
        baseline = {"scenarios": {"bm25_100k": {"p95_ms": True}}}
        assert compare_to_baseline(current, baseline, regression_p95_pct=0.0).regressions == []
        baseline_zero = {"scenarios": {"bm25_100k": {"p95_ms": 0}}}
        assert compare_to_baseline(current, baseline_zero, regression_p95_pct=0.0).regressions == []

    def test_detects_skipped_and_error_scenarios(self):
        current = {
            "a": _scenario("a", skipped=True, skip_reason="neo4j down"),
            "b": _scenario("b", error="boom"),
        }
        baseline = {"scenarios": {"a": {"p95_ms": 1}, "b": {"p95_ms": 1}}}
        result = compare_to_baseline(current, baseline, regression_p95_pct=0.0)
        assert len(result.failures) == 2
        assert result.failures[0].reason == "neo4j down"
        assert result.failures[1].reason == "boom"

    def test_detects_failure_count_regression(self):
        current = {
            "concurrent_chats": _scenario("concurrent_chats", p95=20.0, failures=2, samples=8),
        }
        baseline = {
            "scenarios": {
                "concurrent_chats": {"p95_ms": 20.0, "failures": 0, "samples": 10},
            }
        }
        result = compare_to_baseline(current, baseline, regression_p95_pct=10.0)
        assert result.regressions == []
        assert len(result.failures) == 1
        assert result.failures[0].reason == "failures increased from 0 to 2"

    def test_allows_failures_matching_baseline(self):
        current = {
            "concurrent_chats": _scenario("concurrent_chats", p95=20.0, failures=1, samples=9),
        }
        baseline = {
            "scenarios": {
                "concurrent_chats": {"p95_ms": 20.0, "failures": 1, "samples": 10},
            }
        }
        result = compare_to_baseline(current, baseline, regression_p95_pct=10.0)
        assert result == BaselineComparisonResult(regressions=[], failures=[])

    def test_failure_count_regression_with_invalid_baseline_failures(self):
        current = {
            "concurrent_chats": _scenario("concurrent_chats", p95=20.0, failures=1, samples=9),
        }
        baseline = {
            "scenarios": {
                "concurrent_chats": {"p95_ms": 20.0, "failures": "bad", "samples": 10},
            }
        }
        result = compare_to_baseline(current, baseline, regression_p95_pct=10.0)
        assert len(result.failures) == 1
        assert result.failures[0].reason == "failures increased from 0 to 1"

    def test_regression_warning_message(self):
        warning = RegressionWarning("s", "p95_ms", 10.0, 15.0, 50.0)
        assert "s.p95_ms" in warning.message()

    def test_baseline_scenario_failure_message(self):
        failure = BaselineScenarioFailure("bm25_100k", "index missing")
        assert "bm25_100k" in failure.message()
        assert "index missing" in failure.message()

    def test_comparison_result_has_issues(self):
        result = BaselineComparisonResult(
            regressions=[RegressionWarning("s", "p95_ms", 1.0, 2.0, 100.0)],
            failures=[],
        )
        assert result.has_issues() is True
        assert BaselineComparisonResult(regressions=[], failures=[]).has_issues() is False


class TestBuildBm25FixtureChunks:
    def test_empty_count(self):
        assert build_bm25_fixture_chunks(0) == []

    def test_needle_chunk_present(self):
        chunks = build_bm25_fixture_chunks(100, needle_index=42)
        assert chunks[42].text.startswith("unique needle xyzzy")


class TestInfraBenchmarkReport:
    def test_scenario_map(self):
        report = InfraBenchmarkReport(
            timestamp="t",
            scenarios=[_scenario("a"), _scenario("b")],
        )
        assert set(report.scenario_map()) == {"a", "b"}

    def test_summary_skipped(self):
        report = InfraBenchmarkReport(timestamp="t", scenarios=[], skipped=True, skip_reason="n/a")
        assert "skipped" in report.summary()

    def test_summary_with_metrics(self):
        report = InfraBenchmarkReport(
            timestamp="t",
            scenarios=[_scenario("bm25_100k", memory_bytes=100, failures=1)],
        )
        summary = report.summary()
        assert "memory=100B" in summary
        assert "failures=1" in summary

    def test_summary_error_and_skip_lines(self):
        report = InfraBenchmarkReport(
            timestamp="t",
            scenarios=[
                _scenario("skip", skipped=True, skip_reason="neo4j"),
                _scenario("err", error="failed"),
            ],
        )
        summary = report.summary()
        assert "skipped" in summary
        assert "error" in summary

    def test_print_table_skipped(self, capsys):
        report = InfraBenchmarkReport(timestamp="t", scenarios=[], skipped=True, skip_reason="n/a")
        report.print_table()
        assert "skipped" in capsys.readouterr().out

    def test_print_table_mixed_status(self, capsys):
        report = InfraBenchmarkReport(
            timestamp="t",
            scenarios=[
                _scenario("ok"),
                _scenario("skip", skipped=True, skip_reason="x"),
                _scenario("err", error="x"),
            ],
        )
        report.print_table()
        out = capsys.readouterr().out
        assert "ok" in out
        assert "SKIP" in out or "skip" in out


class TestMeasureStreamLatencies:
    @pytest.mark.asyncio
    async def test_collects_inter_token_latencies(self):
        latencies = await measure_stream_inter_token_latencies_ms(_token_stream("a", "b", "c"))
        assert len(latencies) == 2

    @pytest.mark.asyncio
    async def test_excludes_time_to_first_token(self):
        latencies = await measure_stream_inter_token_latencies_ms(_token_stream("only"))
        assert latencies == []

    @pytest.mark.asyncio
    async def test_get_pipeline_raises_without_factory(self):
        benchmark = InfraBenchmark(pipeline_factory=None)
        with pytest.raises(RuntimeError, match="no pipeline factory configured"):
            await benchmark._get_pipeline()

    @pytest.mark.asyncio
    async def test_get_graph_retriever_raises_without_factory(self):
        benchmark = InfraBenchmark(graph_retriever_factory=None)
        with pytest.raises(RuntimeError, match="no graph retriever factory configured"):
            benchmark._get_graph_retriever()

    @pytest.mark.asyncio
    async def test_warm_pipeline_cache_noop_without_factory(self):
        benchmark = InfraBenchmark(pipeline_factory=None)
        await benchmark._warm_pipeline_cache()
        assert benchmark._cached_pipeline is None

    @pytest.mark.asyncio
    async def test_warm_pipeline_cache_swallows_pipeline_errors(self):
        async def pipeline_factory():
            raise RuntimeError("pipeline unavailable")

        benchmark = InfraBenchmark(pipeline_factory=pipeline_factory)
        await benchmark._warm_pipeline_cache()
        assert benchmark._cached_pipeline is None


class TestInfraBenchmarkScenarios:
    @pytest.mark.asyncio
    async def test_run_unknown_scenario(self):
        benchmark = InfraBenchmark()
        report = await benchmark.run(["not_a_scenario"])
        assert report.scenarios[0].error.startswith("unknown scenario")

    @pytest.mark.asyncio
    async def test_streaming_chat_success(self):
        pipeline = _pipeline_mock(stream_tokens=("x", "y", "z"))
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(streaming_sample_question="q"),
            pipeline_factory=factory,
        )
        result = await benchmark.run_streaming_chat()
        assert calls["n"] == 1
        assert result.name == "streaming_chat"
        assert result.samples == 2
        assert result.p95_ms >= 0.0

    @pytest.mark.asyncio
    async def test_pipeline_factory_reused_across_chat_scenarios(self):
        pipeline = _pipeline_mock()
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                concurrent_chat_count=2,
                streaming_sample_question="q",
            ),
            pipeline_factory=factory,
        )
        report = await benchmark.run(["streaming_chat", "concurrent_chats"])
        assert calls["n"] == 1
        assert len(report.scenarios) == 2
        assert report.scenarios[0].name == "streaming_chat"
        assert report.scenarios[1].name == "concurrent_chats"

    @pytest.mark.asyncio
    async def test_run_resets_pipeline_cache_between_runs(self):
        pipeline = _pipeline_mock()
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                concurrent_chat_count=1,
                streaming_sample_question="q",
            ),
            pipeline_factory=factory,
        )
        await benchmark.run(["streaming_chat"])
        await benchmark.run(["concurrent_chats"])
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_streaming_chat_no_factory(self):
        result = await InfraBenchmark(pipeline_factory=None).run_streaming_chat()
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_streaming_chat_pipeline_error(self):
        pipeline = MagicMock()
        pipeline.chat = AsyncMock(side_effect=RuntimeError("stream fail"))

        async def factory():
            return pipeline

        result = await InfraBenchmark(pipeline_factory=factory).run_streaming_chat()
        assert result.error == "stream fail"

    @pytest.mark.asyncio
    async def test_streaming_chat_no_tokens(self):
        pipeline = _pipeline_mock(stream_tokens=())

        async def factory():
            return pipeline

        result = await InfraBenchmark(pipeline_factory=factory).run_streaming_chat()
        assert result.error == "stream produced no tokens"

    @pytest.mark.asyncio
    async def test_concurrent_chats_success(self):
        pipeline = _pipeline_mock()
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(concurrent_chat_count=3),
            pipeline_factory=factory,
        )
        result = await benchmark.run_concurrent_chats()
        assert calls["n"] == 1
        assert result.samples == 3
        assert result.failures == 0

    @pytest.mark.asyncio
    async def test_concurrent_chats_partial_failures(self):
        calls = {"n": 0}
        pipeline = _pipeline_mock()

        async def chat_full_side_effect(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise TimeoutError("slow")

        pipeline.chat_full = AsyncMock(side_effect=chat_full_side_effect)

        async def factory():
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(concurrent_chat_count=4),
            pipeline_factory=factory,
        )
        result = await benchmark.run_concurrent_chats()
        assert result.failures == 2
        assert result.samples == 2

    @pytest.mark.asyncio
    async def test_concurrent_chats_all_fail(self):
        pipeline = _pipeline_mock(chat_full_error=RuntimeError("down"))

        async def factory():
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(concurrent_chat_count=2),
            pipeline_factory=factory,
        )
        result = await benchmark.run_concurrent_chats()
        assert result.error == "all concurrent chats failed"

    @pytest.mark.asyncio
    async def test_concurrent_chats_pipeline_init_failure(self):
        async def factory():
            raise RuntimeError("pipeline load failed")

        result = await InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(concurrent_chat_count=2),
            pipeline_factory=factory,
        ).run_concurrent_chats()
        assert result.error == "pipeline load failed"

    @pytest.mark.asyncio
    async def test_concurrent_chats_enforces_timeout(self):
        pipeline = MagicMock()

        async def _slow_chat_full(_question: str) -> MagicMock:
            await asyncio.to_thread(time.sleep, 0.05)
            return MagicMock()

        pipeline.chat_full = AsyncMock(side_effect=_slow_chat_full)

        async def factory():
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                concurrent_chat_count=2,
                concurrent_chat_timeout_s=0.01,
            ),
            pipeline_factory=factory,
        )
        result = await benchmark.run_concurrent_chats()
        assert result.failures == 2
        assert result.samples == 0

    @pytest.mark.asyncio
    async def test_concurrent_chats_no_factory(self):
        result = await InfraBenchmark(pipeline_factory=None).run_concurrent_chats()
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_bm25_search_success(self, tmp_path: Path):
        def _fixture(n: int) -> list[Chunk]:
            return build_bm25_fixture_chunks(n, needle_index=min(50, n - 1))

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                bm25_fixture_chunks=200,
                bm25_search_iterations=3,
            ),
            bm25_fixture_builder=_fixture,
        )
        result = await benchmark.run_bm25_search()
        assert result.name == "bm25_100k"
        assert result.samples == 3
        assert result.memory_bytes is not None

    @pytest.mark.asyncio
    async def test_bm25_search_large_batch_path(self):
        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                bm25_fixture_chunks=5_500,
                bm25_search_iterations=1,
            ),
        )
        result = await benchmark.run_bm25_search()
        assert result.samples == 1
        assert result.memory_bytes is not None

    @pytest.mark.asyncio
    async def test_bm25_search_failure(self):
        def _bad_builder(_n: int) -> list[Chunk]:
            return [Chunk(id="x", document_id="d", text="no needle here")]

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                bm25_fixture_chunks=1,
                bm25_search_iterations=1,
            ),
            bm25_fixture_builder=_bad_builder,
        )
        result = await benchmark.run_bm25_search()
        assert result.error

    @pytest.mark.asyncio
    async def test_graph_retrieval_unavailable(self):
        benchmark = InfraBenchmark(neo4j_available=lambda: False)
        result = await benchmark.run_graph_retrieval()
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_graph_retrieval_no_factory(self):
        benchmark = InfraBenchmark(neo4j_available=lambda: True, graph_retriever_factory=None)
        result = await benchmark.run_graph_retrieval()
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_graph_retrieval_success(self):
        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[])

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(graph_search_iterations=2),
            neo4j_available=lambda: True,
            graph_retriever_factory=lambda: retriever,
        )
        result = await benchmark.run_graph_retrieval()
        assert result.samples == 2

    @pytest.mark.asyncio
    async def test_graph_retrieval_error(self):
        def _factory() -> MagicMock:
            raise RuntimeError("graph down")

        benchmark = InfraBenchmark(
            neo4j_available=lambda: True,
            graph_retriever_factory=_factory,
        )
        result = await benchmark.run_graph_retrieval()
        assert result.error == "graph down"

    @pytest.mark.asyncio
    async def test_graph_retrieval_no_timings(self):
        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[])

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(graph_search_iterations=0),
            neo4j_available=lambda: True,
            graph_retriever_factory=lambda: retriever,
        )
        result = await benchmark.run_graph_retrieval()
        assert result.error == "graph search returned no timings"

    @pytest.mark.asyncio
    async def test_graph_retrieval_reuses_cached_pipeline_llm(self):
        benchmark, calls, shared_llm = _graph_llm_sharing_benchmark()
        await benchmark.run(["streaming_chat", "graph_retrieval"])
        assert calls == [{"llm": shared_llm}]

    @pytest.mark.asyncio
    async def test_graph_retrieval_warms_pipeline_for_shared_llm(self):
        benchmark, calls, shared_llm = _graph_llm_sharing_benchmark()
        result = await benchmark.run_graph_retrieval()
        assert calls == [{"llm": shared_llm}]
        assert result.samples == 1

    @pytest.mark.asyncio
    async def test_graph_retrieval_falls_back_when_factory_rejects_llm_kwarg(self):
        shared_llm = MagicMock()
        pipeline = MagicMock()
        pipeline._llm = shared_llm
        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[])

        async def pipeline_factory():
            return pipeline

        def factory() -> MagicMock:
            return retriever

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(graph_search_iterations=1),
            pipeline_factory=pipeline_factory,
            graph_retriever_factory=factory,
            neo4j_available=lambda: True,
        )
        result = await benchmark.run_graph_retrieval()
        assert result.samples == 1

    @pytest.mark.asyncio
    async def test_graph_retrieval_continues_when_pipeline_warm_fails(self):
        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[])
        calls: list[dict[str, object]] = []

        async def pipeline_factory():
            raise RuntimeError("pipeline unavailable")

        def graph_factory(*, llm=None):
            calls.append({"llm": llm})
            return retriever

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(graph_search_iterations=1),
            pipeline_factory=pipeline_factory,
            graph_retriever_factory=graph_factory,
            neo4j_available=lambda: True,
        )
        result = await benchmark.run_graph_retrieval()
        assert calls == [{"llm": None}]
        assert result.samples == 1

    @pytest.mark.asyncio
    async def test_run_all_default_scenarios(self):
        pipeline = _pipeline_mock()
        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[])

        async def pipeline_factory():
            return pipeline

        benchmark = InfraBenchmark(
            thresholds=InfraBenchmarkThresholds(
                concurrent_chat_count=2,
                bm25_fixture_chunks=100,
                bm25_search_iterations=2,
                graph_search_iterations=1,
            ),
            pipeline_factory=pipeline_factory,
            graph_retriever_factory=lambda: retriever,
            neo4j_available=lambda: True,
        )
        report = await benchmark.run()
        assert len(report.scenarios) == 4


class TestNeo4jReachable:
    @staticmethod
    def _fake_neo4j_module(
        *,
        driver: MagicMock | None = None,
        driver_error: BaseException | None = None,
    ) -> MagicMock:
        mod = MagicMock()
        exceptions = MagicMock()
        exceptions.Neo4jError = type("Neo4jError", (Exception,), {})
        mod.exceptions = exceptions
        if driver_error is not None:
            mod.GraphDatabase.driver.side_effect = driver_error
        elif driver is not None:
            mod.GraphDatabase.driver.return_value = driver
        return mod

    def test_returns_false_when_disabled(self, monkeypatch):
        monkeypatch.setenv("NEO4J__ENABLED", "false")
        from src.evals.e2e import infra_benchmark as mod
        from src.evals.e2e.technique_benchmark import reload_settings_module

        reload_settings_module()
        assert mod._neo4j_reachable() is False

    def test_returns_false_when_settings_flag_disabled(self):
        import sys

        from src.evals.e2e import infra_benchmark as mod

        fake_settings = MagicMock()
        fake_settings.neo4j.enabled = False
        fake_neo4j = self._fake_neo4j_module()
        with (
            patch.dict(
                sys.modules,
                {"neo4j": fake_neo4j, "neo4j.exceptions": fake_neo4j.exceptions},
            ),
            patch("src.core.settings.settings", fake_settings),
        ):
            assert mod._neo4j_reachable() is False

    def test_returns_false_on_connection_error(self, monkeypatch):
        import sys

        monkeypatch.setenv("NEO4J__ENABLED", "true")
        monkeypatch.setenv("NEO4J__PASSWORD", "secret")
        from src.evals.e2e import infra_benchmark as mod
        from src.evals.e2e.technique_benchmark import reload_settings_module

        reload_settings_module()
        fake_neo4j = self._fake_neo4j_module(driver_error=OSError("down"))
        with patch.dict(
            sys.modules,
            {"neo4j": fake_neo4j, "neo4j.exceptions": fake_neo4j.exceptions},
        ):
            assert mod._neo4j_reachable() is False

    def test_returns_true_when_verified(self, monkeypatch):
        import sys

        monkeypatch.setenv("NEO4J__ENABLED", "true")
        monkeypatch.setenv("NEO4J__PASSWORD", "secret")
        from src.evals.e2e import infra_benchmark as mod
        from src.evals.e2e.technique_benchmark import reload_settings_module

        reload_settings_module()
        driver = MagicMock()
        fake_neo4j = self._fake_neo4j_module(driver=driver)
        with patch.dict(
            sys.modules,
            {"neo4j": fake_neo4j, "neo4j.exceptions": fake_neo4j.exceptions},
        ):
            assert mod._neo4j_reachable() is True
        driver.verify_connectivity.assert_called_once()
        driver.close.assert_called_once()

    def test_returns_false_on_import_error(self, monkeypatch):
        import sys

        monkeypatch.setenv("NEO4J__ENABLED", "true")
        monkeypatch.setenv("NEO4J__PASSWORD", "secret")
        from src.evals.e2e import infra_benchmark as mod
        from src.evals.e2e.technique_benchmark import reload_settings_module

        reload_settings_module()
        with patch.dict(sys.modules, {"neo4j": None}):
            assert mod._neo4j_reachable() is False


class TestDefaultFactories:
    @pytest.mark.asyncio
    async def test_build_default_pipeline(self):
        with patch("src.evals.e2e.technique_benchmark.build_benchmark_pipeline") as build:
            build.return_value = MagicMock()
            pipeline = await build_default_pipeline()
            assert pipeline is build.return_value

    def test_build_default_graph_retriever(self):
        llm_path = "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
        idx_path = "src.infrastructure.vectordb.bm25.BM25Index.load_or_create"
        gr_path = "src.rag.retrieval.graph_retriever.GraphRetriever.from_settings"
        with (
            patch(llm_path) as llm,
            patch(idx_path) as idx,
            patch(gr_path) as gr,
        ):
            llm.return_value = MagicMock()
            idx.return_value = MagicMock()
            gr.return_value = MagicMock()
            result = build_default_graph_retriever()
            assert result is gr.return_value
            llm.assert_called_once()

    def test_build_default_graph_retriever_reuses_provided_llm(self):
        llm_path = "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
        idx_path = "src.infrastructure.vectordb.bm25.BM25Index.load_or_create"
        gr_path = "src.rag.retrieval.graph_retriever.GraphRetriever.from_settings"
        shared = MagicMock()
        with (
            patch(llm_path) as llm,
            patch(idx_path) as idx,
            patch(gr_path) as gr,
        ):
            idx.return_value = MagicMock()
            gr.return_value = MagicMock()
            result = build_default_graph_retriever(llm=shared)
            assert result is gr.return_value
            llm.assert_not_called()
            gr.assert_called_once()
            assert gr.call_args.kwargs["llm"] is shared


class TestLlmFromPipeline:
    def test_reads_pipeline_llm(self):
        from src.evals.e2e import infra_benchmark as mod

        shared = MagicMock()
        pipeline = MagicMock()
        pipeline._llm = shared
        assert mod._llm_from_pipeline(pipeline) is shared

    def test_falls_back_to_generation_llm(self):
        from src.evals.e2e import infra_benchmark as mod

        shared = MagicMock()
        pipeline = MagicMock()
        pipeline._llm = None
        pipeline.generation._llm = shared
        assert mod._llm_from_pipeline(pipeline) is shared

    def test_returns_none_when_missing(self):
        from src.evals.e2e import infra_benchmark as mod

        pipeline = MagicMock()
        pipeline._llm = None
        pipeline.generation._llm = None
        assert mod._llm_from_pipeline(pipeline) is None


class TestCoerceHelpers:
    def test_coerce_int_invalid(self):
        from src.evals.e2e import infra_benchmark as mod

        assert mod._coerce_int(True, 3) == 3
        assert mod._coerce_int("bad", 3) == 3
        assert mod._coerce_int([], 3) == 3

    def test_coerce_float_invalid(self):
        from src.evals.e2e import infra_benchmark as mod

        assert mod._coerce_float(False, 1.5) == 1.5
        assert mod._coerce_float("bad", 1.5) == 1.5
        assert mod._coerce_float({}, 1.5) == 1.5

    def test_coerce_int_float_and_string(self):
        from src.evals.e2e import infra_benchmark as mod

        assert mod._coerce_int(4.0, 1) == 4
        assert mod._coerce_float("2.5", 1.0) == 2.5
