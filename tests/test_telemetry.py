"""Traces, metrics and the Langfuse mirror (the real SDK, exporting to memory: no network)."""

from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.pricing import Usage
from app.telemetry.langfuse_sink import LangfuseSink
from app.telemetry.metrics import MetricsRecorder, percentile
from app.telemetry.tracer import Telemetry


@pytest.mark.parametrize("q", [50, 95, 99])
def test_percentiles_match_numpy_linear(q: int) -> None:
    values = list(np.random.default_rng(1).exponential(100, size=257))
    assert percentile(values, q) == pytest.approx(float(np.percentile(values, q, method="linear")))
    assert percentile([], q) is None and percentile([7.0], q) == 7.0


def finished(
    telemetry: Telemetry,
    *,
    cache: str,
    latency_cost: tuple[str, str] = ("0", "0.001"),
    avoided: str = "0",
    fail_first: bool = False,
) -> None:
    trace = telemetry.start_trace("gateway.ask", input_text="q", user_id="u1")
    if cache == "miss":
        if fail_first:
            span = trace.start_span("llm.attempt 1", "generation", model="a:free")
            span.end("error", failure="rate_limited")
        span = trace.start_span("llm.attempt 2", "generation", model="b:free")
        span.generation_result(
            usage=Usage(1000, 100),
            cost_usd=Decimal(latency_cost[0]),
            reference_cost_usd=Decimal(latency_cost[1]),
        )
        span.end()
    trace.cost_avoided_usd = Decimal(avoided)
    trace.set(cache=cache, tier="light")
    telemetry.finish(trace, output_text="answer")


def test_metrics_hit_rate_cost_reduction_and_fallbacks() -> None:
    metrics = MetricsRecorder()
    telemetry = Telemetry(metrics=metrics)
    finished(telemetry, cache="miss", fail_first=True)
    finished(telemetry, cache="miss")
    finished(telemetry, cache="hit", avoided="0.001")
    finished(telemetry, cache="hit", avoided="0.001")
    snap = metrics.snapshot()
    assert snap["requests"] == 4
    assert snap["cache_hit_rate"] == 0.5
    assert snap["fallback_requests"] == 1 and snap["fallback_rate"] == 0.5
    assert snap["cost"]["reference_usd"] == "0.0020000000"
    assert snap["cost"]["avoided_usd"] == "0.0020000000"
    assert snap["cost"]["cost_reduction"] == 0.5  # avoided / (reference + avoided)
    assert snap["tokens"] == {"input": 2000, "output": 200}
    assert snap["latency_ms"]["hit"]["count"] == 2 and snap["latency_ms"]["miss"]["count"] == 2
    assert snap["attempts_by_model"] == {"a:free": 1, "b:free": 2}
    assert snap["failures"] == {"a:free: rate_limited": 1}


def test_errors_are_counted_apart_from_latency() -> None:
    metrics = MetricsRecorder()
    telemetry = Telemetry(metrics=metrics)
    trace = telemetry.start_trace("gateway.ask")
    trace.set(error_type="budget_exhausted")
    telemetry.finish(trace, status="error")
    snap = metrics.snapshot()
    assert snap["outcomes"] == {"error": 1} and snap["errors"] == {"budget_exhausted": 1}
    assert snap["latency_ms"]["all"]["count"] == 0


def test_recent_traces_are_bounded_and_serializable() -> None:
    telemetry = Telemetry(keep=3)
    ids = []
    for _ in range(5):
        trace = telemetry.start_trace("gateway.ask", input_text="x" * 1000)
        with trace.span("cache.lookup", "embedding") as span:
            span.set(outcome="miss")
        telemetry.finish(trace, output_text="y")
        ids.append(trace.trace_id)
    recent = telemetry.recent(10)
    assert [t.trace_id for t in recent] == ids[:1:-1]
    data = recent[0].as_dict()
    assert len(data["input"]) <= 280 and data["spans"][0]["attributes"] == {"outcome": "miss"}
    json.dumps(data)
    assert telemetry.get(ids[0]) is None and telemetry.get(ids[-1]) is not None


def test_a_failing_sink_never_breaks_a_request() -> None:
    class Broken:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            def explode(*args: object) -> None:
                raise RuntimeError("sink down")

            return explode

    telemetry = Telemetry(sinks=[Broken()])  # type: ignore[list-item]
    trace = telemetry.start_trace("gateway.ask")
    with trace.span("x"):
        pass
    telemetry.finish(trace)
    assert trace.finished


_CLIENTS = iter(range(1_000))


def langfuse_sink(capture_content: bool = True) -> tuple[LangfuseSink, InMemorySpanExporter]:
    # The SDK keeps one client per public key in the process: a key per test keeps them apart.
    exporter = InMemorySpanExporter()
    sink = LangfuseSink.create(
        public_key=f"pk-lf-test-{next(_CLIENTS)}",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        environment="test",
        release="1.0.0",
        capture_content=capture_content,
        span_exporter=exporter,
    )
    return sink, exporter


def test_langfuse_receives_nested_observations_with_usage_and_cost() -> None:
    sink, exporter = langfuse_sink()
    telemetry = Telemetry(sinks=[sink])
    trace = telemetry.start_trace(
        "gateway.ask", input_text="How many PTO days?", user_id="u42", session_id="s1"
    )
    with trace.span("retrieval.hybrid", "retriever", top_k=4):
        pass
    span = trace.start_span(
        "llm.attempt 1", "generation", model="heavy/a:free", tier="heavy", attempt=1, max_tokens=900
    )
    span.generation_result(
        usage=Usage(1200, 80, cached_input_tokens=200),
        cost_usd=Decimal("0.0003"),
        reference_cost_usd=Decimal("0.0064"),
    )
    span.end(output_preview="Twenty days [S1].")
    trace.set(cache="miss", tier="heavy")
    telemetry.finish(trace, output_text="Twenty days [S1].")
    sink.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root, retrieval, generation = spans["gateway.ask"], spans["retrieval.hybrid"], spans["llm.attempt 1"]
    assert retrieval.parent.span_id == root.context.span_id == generation.parent.span_id
    assert generation.attributes["langfuse.observation.type"] == "generation"
    assert retrieval.attributes["langfuse.observation.type"] == "retriever"
    assert generation.attributes["langfuse.observation.model.name"] == "heavy/a:free"
    usage = json.loads(generation.attributes["langfuse.observation.usage_details"])
    assert usage == {"input": 1200, "output": 80, "cache_read_input_tokens": 200}
    assert json.loads(generation.attributes["langfuse.observation.cost_details"]) == {"total": 0.0003}
    for s in (root, retrieval, generation):  # trace-level attributes reach every observation
        assert s.attributes["user.id"] == "u42" and s.attributes["session.id"] == "s1"
        assert s.attributes["langfuse.trace.name"] == "gateway.ask"
    assert "Twenty days" in json.dumps(dict(root.attributes))


def test_langfuse_gets_no_text_when_content_capture_is_off() -> None:
    sink, exporter = langfuse_sink(capture_content=False)
    telemetry = Telemetry(sinks=[sink])
    trace = telemetry.start_trace("gateway.ask", input_text="a private question about my salary")
    telemetry.finish(trace, output_text="a private answer")
    sink.flush()
    blob = json.dumps([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert "salary" not in blob and "private answer" not in blob
