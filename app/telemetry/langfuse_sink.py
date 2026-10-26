"""Mirrors every trace to Langfuse, live, through the official SDK (v5, OpenTelemetry based).

Each gateway span becomes a Langfuse observation as it starts, with the matching type: the LLM
attempts are ``generation`` observations carrying the model id, ``usage_details`` (input,
output and cached tokens) and ``cost_details`` (the actual cost in dollars), so Langfuse's own
dashboards show tokens, cost and latency per model; the retrieval step is a ``retriever``, the
defanging a ``guardrail``, the cache lookup an ``embedding``. The trace-level attributes (trace
name, hashed user id, session id, tags) are propagated to every observation, which is what
Langfuse's aggregations by user and by session need.

The SDK batches and exports in a background thread; nothing here waits on the network. With
``LANGFUSE_CAPTURE_CONTENT=false`` only numbers and ids leave the service, never the text of a
question or an answer. Failures are logged once and never reach a request.
"""

from __future__ import annotations

import logging
from typing import Any

from app.pricing import usd
from app.telemetry.tracer import SpanRecord, Trace, preview

logger = logging.getLogger(__name__)

#: Gateway span kind -> Langfuse observation type.
_AS_TYPE = {
    "span": "span",
    "generation": "generation",
    "retriever": "retriever",
    "guardrail": "guardrail",
    "embedding": "embedding",
    "event": "span",
}


class LangfuseSink:
    def __init__(self, client: Any, *, capture_content: bool = True, environment: str | None = None) -> None:
        self.client = client
        self.capture_content = capture_content
        self.environment = environment
        self._roots: dict[str, Any] = {}
        self._observations: dict[str, Any] = {}
        self._warned = False

    @classmethod
    def create(
        cls,
        *,
        public_key: str,
        secret_key: str,
        base_url: str,
        environment: str,
        release: str,
        capture_content: bool,
        span_exporter: Any | None = None,
    ) -> LangfuseSink:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            environment=environment,
            release=release,
            span_exporter=span_exporter,
        )
        return cls(client, capture_content=capture_content, environment=environment)

    # ----------------------------------------------------------------- helpers
    def _propagated(self, trace: Trace) -> Any:
        from langfuse import propagate_attributes

        tags = ["llmops-gateway", trace.name]
        return propagate_attributes(
            trace_name=trace.name,
            user_id=trace.user_id,
            session_id=trace.session_id,
            tags=tags,
        )

    def _content(self, text: str | None) -> str | None:
        return preview(text, 2_000) if self.capture_content else None

    def _guard(self, action: str, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "Langfuse %s failed (%s); further failures are not logged", action, type(exc).__name__
            )

    # ----------------------------------------------------------------- the sink
    def on_trace_start(self, trace: Trace) -> None:
        try:
            with self._propagated(trace):
                root = self.client.start_observation(
                    name=trace.name,
                    as_type="span",
                    input=self._content(trace.input_text),
                    metadata={"gateway_trace_id": trace.trace_id, **_flat(trace.attributes)},
                )
            self._roots[trace.trace_id] = root
        except Exception as exc:
            self._guard("trace start", exc)

    def on_span_start(self, trace: Trace, span: SpanRecord) -> None:
        root = self._roots.get(trace.trace_id)
        if root is None:
            return
        try:
            kwargs: dict[str, Any] = {
                "name": span.name,
                "as_type": _AS_TYPE.get(span.kind, "span"),
                "metadata": _flat(span.attributes),
            }
            if span.kind == "generation":
                kwargs["model"] = span.attributes.get("model")
                kwargs["model_parameters"] = {
                    k: v for k, v in span.attributes.items() if k in {"max_tokens", "tier", "attempt"}
                }
            with self._propagated(trace):
                self._observations[span.span_id] = root.start_observation(**kwargs)
        except Exception as exc:
            self._guard("span start", exc)

    def on_span_end(self, trace: Trace, span: SpanRecord) -> None:
        observation = self._observations.pop(span.span_id, None)
        if observation is None:
            return
        try:
            update: dict[str, Any] = {"metadata": _flat(span.attributes)}
            if span.status == "error":
                update["level"] = "ERROR"
                update["status_message"] = str(
                    span.attributes.get("failure") or span.attributes.get("error") or "error"
                )
            if span.kind == "generation" and span.usage is not None:
                update["usage_details"] = {
                    "input": span.usage.input_tokens,
                    "output": span.usage.output_tokens,
                    "cache_read_input_tokens": span.usage.cached_input_tokens,
                }
                if span.cost_usd is not None:
                    update["cost_details"] = {"total": float(span.cost_usd)}
                if self.capture_content and span.attributes.get("output_preview"):
                    update["output"] = span.attributes["output_preview"]
            observation.update(**update)
            observation.end()
        except Exception as exc:
            self._guard("span end", exc)

    def on_trace_end(self, trace: Trace) -> None:
        root = self._roots.pop(trace.trace_id, None)
        if root is None:
            return
        try:
            usage = trace.usage()
            metadata = {
                **_flat(trace.attributes),
                "latency_ms": None if trace.latency_ms is None else round(trace.latency_ms, 2),
                "ttft_ms": None if trace.ttft_ms is None else round(trace.ttft_ms, 2),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usd(trace.cost()),
                "reference_cost_usd": usd(trace.reference_cost()),
                "cost_avoided_usd": usd(trace.cost_avoided_usd),
            }
            root.update(
                output=self._content(trace.output_text),
                metadata=metadata,
                level="ERROR" if trace.status == "error" else None,
            )
            root.end()
        except Exception as exc:
            self._guard("trace end", exc)

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception as exc:  # pragma: no cover - network at shutdown
            self._guard("flush", exc)

    def shutdown(self) -> None:
        try:
            self.client.shutdown()
        except Exception as exc:  # pragma: no cover
            self._guard("shutdown", exc)


def _flat(attributes: dict[str, Any]) -> dict[str, Any]:
    """Metadata Langfuse can filter on: scalars as they are, anything else as short text."""
    flat: dict[str, Any] = {}
    for key, value in attributes.items():
        if key == "output_preview":
            continue
        if value is None or isinstance(value, bool | int | float | str):
            flat[key] = value
        else:
            flat[key] = preview(str(value), 400)
    return flat
