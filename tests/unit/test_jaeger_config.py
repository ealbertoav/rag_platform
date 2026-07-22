"""#109 — static config-shape checks for the Jaeger tracing UI.

Asserts docker-compose.yml and configs/otel-collector.yaml are wired
correctly, without needing Docker running. See tests/integration/test_jaeger.py
for the corresponding reachability check.
"""

from __future__ import annotations

import yaml

from src.core.constants import ROOT

_COMPOSE_PATH = ROOT / "docker-compose.yml"
_OTEL_COLLECTOR_CONFIG_PATH = ROOT / "configs" / "otel-collector.yaml"


def _load_compose() -> dict[str, object]:
    return yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))


def _load_otel_collector_config() -> dict[str, object]:
    return yaml.safe_load(_OTEL_COLLECTOR_CONFIG_PATH.read_text(encoding="utf-8"))


class TestDockerComposeJaegerService:
    def test_jaeger_service_defined_with_pinned_v2_image(self):
        """Pinned (not :latest) — this is a fast-moving new major version
        (v1 → v2 migration), and an unpinned tag could silently jump to a
        release with a different health mechanism/port again."""
        compose = _load_compose()
        services = compose["services"]
        assert "jaeger" in services
        image = services["jaeger"]["image"]
        assert image.startswith("cr.jaegertracing.io/jaegertracing/jaeger:")
        assert image != "cr.jaegertracing.io/jaegertracing/jaeger:latest"

    def test_jaeger_ui_port_exposed(self):
        compose = _load_compose()
        ports = compose["services"]["jaeger"]["ports"]
        assert any(str(p).startswith("16686:") for p in ports)

    def test_jaeger_otlp_ports_not_exposed_to_host(self):
        """4317/4318 are already mapped to otel-collector — no host conflict."""
        compose = _load_compose()
        ports = compose["services"]["jaeger"]["ports"]
        assert not any(str(p).startswith(("4317:", "4318:")) for p in ports)

    def test_otel_collector_depends_on_jaeger_healthy(self):
        compose = _load_compose()
        depends_on = compose["services"]["otel-collector"]["depends_on"]
        assert depends_on["jaeger"]["condition"] == "service_healthy"


class TestOtelCollectorJaegerExporter:
    def test_traces_pipeline_exports_to_both_debug_and_jaeger(self):
        config = _load_otel_collector_config()
        exporters = config["service"]["pipelines"]["traces"]["exporters"]
        assert "debug" in exporters
        assert "otlp_grpc/jaeger" in exporters

    def test_jaeger_exporter_targets_internal_docker_network(self):
        config = _load_otel_collector_config()
        jaeger_exporter = config["exporters"]["otlp_grpc/jaeger"]
        assert jaeger_exporter["endpoint"] == "jaeger:4317"

    def test_metrics_and_logs_pipelines_unaffected(self):
        """#92: Prometheus stays the single metrics source — Jaeger is trace-only."""
        config = _load_otel_collector_config()
        pipelines = config["service"]["pipelines"]
        assert pipelines["metrics"]["exporters"] == ["debug"]
        assert pipelines["logs"]["exporters"] == ["debug"]
