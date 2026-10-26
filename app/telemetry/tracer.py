"""Traces and spans: one trace per request, one span per step.

The gateway records every request the same way, whatever happens to it::

    gateway.ask                      (the trace: endpoint, client, outcome, totals)
      guard.defang                   (guardrail: signals found, characters neutralized)
      cache.lookup                   (exact key, then embedding and nearest neighbour)
      retrieval.hybrid               (retriever: the chunks, their dense, BM25 and RRF scores)
      router.complexity              (tier, score, the named signals)
      llm.attempt 1                  (generation: model, tokens, actual and list-price cost)
      llm.attempt 2                  (only when the first one failed: the fallback)
      cache.write

The records live in memory (the last ``TRACES_KEPT``), feed :class:`MetricsRecorder` (hit rate,
p50/p95/p99, cost), and are mirrored live to every sink: Langfuse when its keys are set. A sink
that fails is logged and skipped; telemetry never fails a request.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol

from app.pricing import Usage, usd

logger = logging.getLogger(__name__)

SpanKind = Literal["span", "generation", "retriever", "guardrail", "embedding", "event"]
PREVIEW_CHARS = 280


def preview(text: str | None, limit: int = PREVIEW_CHARS) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class SpanRecord:
    span_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start_ms: float
    duration_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    # Generations only.
    model: str | None = None
    usage: Usage | None = None
    cost_usd: Decimal | None = None
    reference_cost_usd: Decimal | None = None
    ttft_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "start_ms": round(self.start_ms, 2),
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.kind == "generation":
            data["model"] = self.model
            data["usage"] = (
                None
                if self.usage is None
                else {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                    "cached_input_tokens": self.usage.cached_input_tokens,
                    "estimated": self.usage.estimated,
                }
            )
            data["cost_usd"] = usd(self.cost_usd)
            data["reference_cost_usd"] = usd(self.reference_cost_usd)
            data["ttft_ms"] = None if self.ttft_ms is None else round(self.ttft_ms, 2)
        return data


class Sink(Protocol):
    """Receives every trace as it happens. Implementations must not raise."""

    def on_trace_start(self, trace: Trace) -> None: ...

    def on_span_start(self, trace: Trace, span: SpanRecord) -> None: ...

    def on_span_end(self, trace: Trace, span: SpanRecord) -> None: ...

    def on_trace_end(self, trace: Trace) -> None: ...


class SpanHandle:
    """An open span. End it with :meth:`end`, or use it as a context manager."""

    def __init__(self, trace: Trace, record: SpanRecord, started: float) -> None:
        self.trace = trace
        self.record = record
        self._started = started
        self.ended = False

    def set(self, **attributes: Any) -> SpanHandle:
        self.record.attributes.update(attributes)
        return self

    def end(self, status: Literal["ok", "error"] = "ok", **attributes: Any) -> SpanRecord:
        if self.ended:
            return self.record
        self.ended = True
        self.record.attributes.update(attributes)
        self.record.status = status
        self.record.duration_ms = (time.perf_counter() - self._started) * 1000
        self.trace._emit("on_span_end", self.record)
        return self.record

    def generation_result(
        self,
        *,
        usage: Usage,
        cost_usd: Decimal | None,
        reference_cost_usd: Decimal,
        ttft_ms: float | None = None,
    ) -> None:
        self.record.usage = usage
        self.record.cost_usd = cost_usd
        self.record.reference_cost_usd = reference_cost_usd
        self.record.ttft_ms = ttft_ms

    def __enter__(self) -> SpanHandle:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        if exc is not None:
            self.end("error", error=type(exc).__name__)
        else:
            self.end()


class Trace:
    def __init__(
        self,
        name: str,
        *,
        sinks: list[Sink],
        input_text: str | None = None,
        attributes: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.trace_id = secrets.token_hex(16)  # 32 hex characters: Langfuse's and W3C's trace id shape
        self.name = name
        self.started_at = time.time()
        self._t0 = time.perf_counter()
        self.spans: list[SpanRecord] = []
        self.attributes: dict[str, Any] = dict(attributes or {})
        self.input_text = input_text
        self.output_text: str | None = None
        self.user_id = user_id
        self.session_id = session_id
        self.status: Literal["ok", "error"] = "ok"
        self.latency_ms: float | None = None
        self.ttft_ms: float | None = None
        self.cost_avoided_usd = Decimal(0)
        self._sinks = sinks
        self.finished = False

    # ---------------------------------------------------------------- recording
    def _emit(self, method: str, *args: Any) -> None:
        for sink in self._sinks:
            try:
                getattr(sink, method)(self, *args)
            except Exception:  # a sink must never break a request
                logger.exception("Telemetry sink %s.%s failed", type(sink).__name__, method)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def start_span(self, name: str, kind: SpanKind = "span", **attributes: Any) -> SpanHandle:
        record = SpanRecord(
            span_id=secrets.token_hex(8),
            parent_id=None,  # every step is a child of the request
            name=name,
            kind=kind,
            start_ms=self.elapsed_ms(),
            attributes=dict(attributes),
            model=attributes.get("model") if kind == "generation" else None,
        )
        self.spans.append(record)
        handle = SpanHandle(self, record, time.perf_counter())
        self._emit("on_span_start", record)
        return handle

    @contextmanager
    def span(self, name: str, kind: SpanKind = "span", **attributes: Any) -> Iterator[SpanHandle]:
        handle = self.start_span(name, kind, **attributes)
        try:
            yield handle
        except BaseException as exc:
            handle.end("error", error=type(exc).__name__)
            raise
        else:
            handle.end()

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    # ------------------------------------------------------------------ totals
    @property
    def generations(self) -> list[SpanRecord]:
        return [s for s in self.spans if s.kind == "generation"]

    def usage(self) -> Usage:
        records = [g.usage for g in self.generations if g.usage is not None]
        return Usage(
            input_tokens=sum(u.input_tokens for u in records),
            output_tokens=sum(u.output_tokens for u in records),
            cached_input_tokens=sum(u.cached_input_tokens for u in records),
            estimated=any(u.estimated for u in records),
        )

    def cost(self) -> Decimal | None:
        """What the request cost: every attempt counts, failed ones included. None if unknown."""
        costs = [g.cost_usd for g in self.generations if g.usage is not None]
        if any(c is None for c in costs):
            return None
        return sum((c for c in costs if c is not None), Decimal(0))

    def reference_cost(self) -> Decimal:
        return sum(
            (g.reference_cost_usd for g in self.generations if g.reference_cost_usd is not None),
            Decimal(0),
        )

    def as_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        usage = self.usage()
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)),
            "status": self.status,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 2),
            "ttft_ms": None if self.ttft_ms is None else round(self.ttft_ms, 2),
            "attributes": self.attributes,
            "input": preview(self.input_text) if include_content else None,
            "output": preview(self.output_text) if include_content else None,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "estimated": usage.estimated,
            },
            "cost_usd": usd(self.cost()),
            "reference_cost_usd": usd(self.reference_cost()),
            "cost_avoided_usd": usd(self.cost_avoided_usd),
            "spans": [s.as_dict() for s in self.spans],
        }


class Telemetry:
    """Starts traces, keeps the recent ones, and hands finished ones to the metrics."""

    def __init__(self, *, sinks: list[Sink] | None = None, metrics: Any = None, keep: int = 200) -> None:
        self.sinks: list[Sink] = list(sinks or [])
        self.metrics = metrics
        self.keep = keep
        self._recent: OrderedDict[str, Trace] = OrderedDict()

    def start_trace(
        self,
        name: str,
        *,
        input_text: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        **attributes: Any,
    ) -> Trace:
        trace = Trace(
            name,
            sinks=self.sinks,
            input_text=input_text,
            attributes=attributes,
            user_id=user_id,
            session_id=session_id,
        )
        trace._emit("on_trace_start")
        return trace

    def finish(
        self, trace: Trace, *, output_text: str | None = None, status: Literal["ok", "error"] = "ok"
    ) -> None:
        if trace.finished:
            return
        trace.finished = True
        trace.output_text = output_text
        trace.status = status
        trace.latency_ms = trace.elapsed_ms()
        self._recent[trace.trace_id] = trace
        while len(self._recent) > self.keep:
            self._recent.popitem(last=False)
        if self.metrics is not None:
            try:
                self.metrics.record(trace)
            except Exception:  # pragma: no cover - metrics must never break a request
                logger.exception("Recording metrics failed")
        trace._emit("on_trace_end")

    def recent(self, limit: int = 20) -> list[Trace]:
        return list(reversed(self._recent.values()))[:limit]

    def get(self, trace_id: str) -> Trace | None:
        return self._recent.get(trace_id)

    def flush(self) -> None:
        for sink in self.sinks:
            flush = getattr(sink, "flush", None)
            if flush is not None:
                try:
                    flush()
                except Exception:  # pragma: no cover
                    logger.exception("Flushing telemetry sink %s failed", type(sink).__name__)
