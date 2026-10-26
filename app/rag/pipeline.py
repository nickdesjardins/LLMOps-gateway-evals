"""A grounded question, end to end, as one traced transaction.

::

    question ──> guard ──> cache ──hit──> stored answer (milliseconds, no model call)
                             │
                            miss
                             v
                         retrieval ──> router ──> cascade ──> validate ──> cache write
                                         (light or heavy tier, failover on 429 and friends)

Both entry points share every step; :meth:`RagPipeline.answer` returns one JSON body and
:meth:`RagPipeline.open_stream` returns a :class:`StreamPlan` whose events are Server-Sent
Events. Anything that can refuse a request (the injection policy, the token ceiling, an empty
model budget, a busy demo) refuses it *before* the stream opens, so a client gets a normal HTTP
error instead of a 200 that turns into an error halfway.

Only a complete, validated answer to an unflagged question is cached. A refusal ("the knowledge
base does not answer this") is a valid answer and is cached too: asking again changes nothing
until the corpus does, and a corpus change empties the namespace.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from app.budget import DemoBudget
from app.cache.semantic import LookupResult, SemanticCache, namespace_for
from app.config import Settings, Tier
from app.errors import GatewayError, PromptInjectionRefused, PromptTooLong, UpstreamUnavailable
from app.guard import GuardReport, defang, estimate_messages_tokens
from app.pricing import usd
from app.providers.base import ChatRequest, Completion
from app.rag.corpus import Corpus
from app.rag.prompts import (
    PROMPT_VERSION,
    REFUSAL_MESSAGE,
    REFUSAL_TOKEN,
    PromptSource,
    build_messages,
    citation_problem,
    cited_labels,
    is_refusal,
)
from app.rag.retriever import HybridRetriever, Retrieved
from app.routing.cascade import (
    BeforeCall,
    CascadeResult,
    MidStreamFailure,
    ModelCascade,
    RequestFactory,
    Validator,
)
from app.routing.complexity import ComplexityRouter, RouteDecision, TaskHint
from app.telemetry.tracer import Telemetry, Trace

logger = logging.getLogger(__name__)

CacheMode = Literal["use", "bypass", "refresh"]
#: A passage is "strong" when its dense score is within STRONG_MARGIN of the best one and above
#: RELEVANCE_FLOOR; the number of documents among strong passages is the router's
#: retrieval-spread feature. The floor matters: a question the corpus does not cover matches
#: many documents weakly, and that spread is irrelevance, not complexity. Measured with
#: bge-small on the golden set: in-scope questions have a best dense score of 0.645 or more,
#: off-topic ones mostly below 0.56.
STRONG_MARGIN = 0.05
RELEVANCE_FLOOR = 0.60


@dataclass(frozen=True)
class AskOptions:
    task: TaskHint = "auto"
    #: use: look up and store. bypass: neither. refresh: skip the lookup, store the new answer.
    cache: CacheMode = "use"
    client: str = "local"
    session_id: str | None = None


@dataclass
class Prepared:
    trace: Trace
    question: str
    defanged: str
    report: GuardReport
    lookup: LookupResult | None
    options: AskOptions
    retrieved: list[Retrieved] = field(default_factory=list)
    route: RouteDecision | None = None
    sources: list[PromptSource] = field(default_factory=list)


def documents_in_play(retrieved: list[Retrieved]) -> int:
    if not retrieved:
        return 0
    best = max(r.dense for r in retrieved)
    cut = max(best - STRONG_MARGIN, RELEVANCE_FLOOR)
    return len({r.chunk.doc_id for r in retrieved if r.dense >= cut})


class RagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        corpus: Corpus,
        retriever: HybridRetriever,
        cache: SemanticCache | None,
        router: ComplexityRouter,
        cascade: ModelCascade,
        telemetry: Telemetry,
        budget: DemoBudget,
    ) -> None:
        self.settings = settings
        self.corpus = corpus
        self.retriever = retriever
        self.cache = cache
        self.router = router
        self.cascade = cascade
        self.telemetry = telemetry
        self.budget = budget
        self.namespace = namespace_for("rag", PROMPT_VERSION, corpus.fingerprint, retriever.embedder.id)

    # ------------------------------------------------------------------ shared
    def _start(self, question: str, options: AskOptions, streamed: bool) -> Trace:
        return self.telemetry.start_trace(
            "gateway.ask",
            input_text=question,
            user_id=options.client,
            session_id=options.session_id,
            endpoint="/api/v1/ask/stream" if streamed else "/api/v1/ask",
            stream=streamed,
            prompt_version=PROMPT_VERSION,
            corpus_version=self.corpus.fingerprint,
        )

    async def _prepare(self, question: str, options: AskOptions, streamed: bool) -> Prepared:
        trace = self._start(question, options, streamed)
        try:
            with trace.span("guard.defang", "guardrail") as span:
                defanged, report = defang(question)
                span.set(**report.as_dict(), policy=self.settings.injection_policy)
            trace.set(injection_detected=report.suspicious, injection_signals=list(report.signals))
            if report.suspicious and self.settings.injection_policy == "block":
                raise PromptInjectionRefused(
                    "The question was refused: it contains text that tries to change the "
                    "assistant's instructions."
                )
            lookup: LookupResult | None = None
            if self.cache is None:
                trace.set(cache="off")
            elif options.cache != "use":
                trace.set(cache="bypass", cache_mode=options.cache)
            else:
                with trace.span("cache.lookup", "embedding", backend=self.cache.store.name) as span:
                    lookup = await self.cache.lookup(self.namespace, defanged)
                    span.set(**lookup.as_dict())
                trace.set(cache=lookup.outcome, cache_lookup_ms=round(lookup.total_ms, 3))
            prepared = Prepared(trace, question, defanged, report, lookup, options)
            if lookup is not None and lookup.outcome == "hit":
                return prepared
            self.budget.check_model_call(options.client)
            with trace.span("retrieval.hybrid", "retriever", top_k=self.settings.retrieval_top_k) as span:
                prepared.retrieved = await self.retriever.search(defanged, self.settings.retrieval_top_k)
                span.set(chunks=[r.as_dict() for r in prepared.retrieved])
            spread = documents_in_play(prepared.retrieved)
            with trace.span("router.complexity") as span:
                prepared.route = self.router.route(defanged, task=options.task, documents_in_play=spread)
                span.set(**prepared.route.as_dict(), documents_in_play=spread)
            trace.set(tier=prepared.route.tier, route_score=round(prepared.route.score, 3))
            prepared.sources = [
                PromptSource(
                    label=f"S{r.rank}", chunk_id=r.chunk.chunk_id, title=r.chunk.title, text=r.chunk.text
                )
                for r in prepared.retrieved
            ]
            messages = build_messages(defanged, prepared.sources)
            if estimate_messages_tokens(messages) > self.settings.max_input_tokens:
                raise PromptTooLong("The question and its passages exceed the input ceiling.")
            return prepared
        except GatewayError as exc:
            self._fail(trace, exc)
            raise

    def _fail(self, trace: Trace, exc: GatewayError) -> None:
        trace.set(error_type=exc.error_type, incident_id=exc.incident_id)
        self.telemetry.finish(trace, status="error")

    def _request_factory(self, prepared: Prepared) -> RequestFactory:
        messages = build_messages(prepared.defanged, prepared.sources)

        def make(tier: Tier) -> ChatRequest:
            return ChatRequest(
                messages=messages,
                max_tokens=self.settings.max_output_tokens(tier),
                tier=tier,
                tag="rag",
            )

        return make

    def _validator(self, prepared: Prepared) -> Validator:
        count = len(prepared.sources)

        def validate(completion: Completion) -> str | None:
            if completion.truncated and not is_refusal(completion.text):
                return "the answer was cut off at the output ceiling"
            return citation_problem(completion.text, count)

        return validate

    def _charge(self, client: str) -> BeforeCall:
        async def before_call(_model: str) -> None:
            await self.budget.charge_model_call(client)

        return before_call

    # ---------------------------------------------------------------- payloads
    def _sources_payload(self, prepared: Prepared) -> list[dict[str, Any]]:
        return [{"label": f"S{r.rank}", **r.as_dict()} for r in prepared.retrieved]

    def _miss_payload(self, prepared: Prepared, result: CascadeResult, problem: str | None) -> dict[str, Any]:
        text = result.completion.text
        refused = is_refusal(text)
        by_label = {s.label: s for s in prepared.sources}
        citations = []
        if not refused:
            for number in cited_labels(text):
                source = by_label.get(f"S{number}")
                if source is not None:
                    citations.append(
                        {"label": source.label, "chunk_id": source.chunk_id, "title": source.title}
                    )
        trace = prepared.trace
        usage = trace.usage()
        return {
            "trace_id": trace.trace_id,
            "question": prepared.question,
            "answer": REFUSAL_MESSAGE if refused else text,
            "refused": refused,
            "citations": citations,
            "sources": self._sources_payload(prepared),
            "cache": (
                prepared.lookup.as_dict() if prepared.lookup else {"outcome": trace.attributes.get("cache")}
            ),
            "route": prepared.route.as_dict() if prepared.route else None,
            "model": result.completion.model,
            "tier": result.tier,
            "attempts": [a.as_dict() for a in result.attempts],
            "fallback": result.fell_back,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "estimated": usage.estimated,
            },
            "cost": {
                "actual_usd": usd(trace.cost()),
                "reference_usd": usd(trace.reference_cost()),
                "avoided_usd": usd(Decimal(0)),
            },
            "validation": problem,
            "injection": prepared.report.as_dict(),
            "prompt_version": PROMPT_VERSION,
            "corpus_version": self.corpus.fingerprint,
        }

    def _hit_payload(self, prepared: Prepared) -> dict[str, Any]:
        assert prepared.lookup is not None and prepared.lookup.entry is not None
        entry = prepared.lookup.entry
        stored = entry.payload
        avoided = Decimal(entry.reference_cost_usd)
        trace = prepared.trace
        trace.cost_avoided_usd = avoided
        trace.set(tier=stored.get("tier"), origin_model=stored.get("model"))
        return {
            **stored,
            "trace_id": trace.trace_id,
            "question": prepared.question,
            "cache": prepared.lookup.as_dict()
            | {"cached_at": entry.created_at, "origin_trace_id": stored.get("trace_id")},
            "route": None,
            "attempts": [],
            "fallback": False,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "estimated": False},
            "cost": {
                "actual_usd": usd(Decimal(0)),
                "reference_usd": usd(Decimal(0)),
                "avoided_usd": usd(avoided),
            },
            "injection": prepared.report.as_dict(),
        }

    async def _store(self, prepared: Prepared, payload: dict[str, Any], completion: Completion) -> None:
        if self.cache is None or prepared.options.cache == "bypass":
            return
        trace = prepared.trace
        reason = None
        if prepared.report.suspicious:
            reason = "the guard flagged the question"
        elif payload.get("validation"):
            reason = "the answer failed validation"
        elif completion.truncated:
            reason = "the answer was cut off"
        if reason:
            trace.set(cache_write="skipped", cache_write_reason=reason)
            return
        stored = {
            k: payload[k]
            for k in (
                "trace_id",
                "answer",
                "refused",
                "citations",
                "sources",
                "model",
                "tier",
                "prompt_version",
                "corpus_version",
            )
        }
        vector = prepared.lookup.vector if prepared.lookup is not None else None
        with trace.span("cache.write", backend=self.cache.store.name) as span:
            ok = await self.cache.store_answer(
                self.namespace,
                prepared.defanged,
                stored,
                reference_cost_usd=trace.reference_cost(),
                actual_cost_usd=trace.cost() or Decimal(0),
                vector=vector,
            )
            span.set(stored=ok)
        trace.set(cache_write="stored" if ok else "refused by the store")

    def _finish(self, prepared: Prepared, payload: dict[str, Any]) -> dict[str, Any]:
        trace = prepared.trace
        trace.set(
            refused=payload.get("refused", False),
            model=payload.get("model"),
            fallback=payload.get("fallback", False),
        )
        self.telemetry.finish(trace, output_text=payload.get("answer"))
        payload["latency_ms"] = round(trace.latency_ms or 0.0, 2)
        payload["ttft_ms"] = None if trace.ttft_ms is None else round(trace.ttft_ms, 2)
        return payload

    # ------------------------------------------------------------------- JSON
    async def answer(self, question: str, options: AskOptions | None = None) -> dict[str, Any]:
        options = options or AskOptions()
        prepared = await self._prepare(question, options, streamed=False)
        if prepared.lookup is not None and prepared.lookup.outcome == "hit":
            return self._finish(prepared, self._hit_payload(prepared))
        assert prepared.route is not None
        try:
            async with self.budget.model_slot():
                result = await self.cascade.complete(
                    self._request_factory(prepared),
                    prepared.trace,
                    tier=prepared.route.tier,
                    validate=self._validator(prepared),
                    before_call=self._charge(options.client),
                )
        except GatewayError as exc:
            self._fail(prepared.trace, exc)
            raise
        payload = self._miss_payload(prepared, result, None)
        await self._store(prepared, payload, result.completion)
        return self._finish(prepared, payload)

    # ------------------------------------------------------------------ stream
    async def open_stream(self, question: str, options: AskOptions | None = None) -> StreamPlan:
        options = options or AskOptions()
        prepared = await self._prepare(question, options, streamed=True)
        if not (prepared.lookup is not None and prepared.lookup.outcome == "hit"):
            self.budget.check_slot()
        return StreamPlan(self, prepared)


class StreamPlan:
    """The events of one streamed answer: ``meta``, ``sources``, ``delta``..., ``done`` or ``error``."""

    def __init__(self, pipeline: RagPipeline, prepared: Prepared) -> None:
        self.pipeline = pipeline
        self.prepared = prepared

    @property
    def trace_id(self) -> str:
        return self.prepared.trace.trace_id

    async def events(self) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        pipeline, prepared = self.pipeline, self.prepared
        trace = prepared.trace
        meta = {
            "trace_id": trace.trace_id,
            "cache": prepared.lookup.as_dict()
            if prepared.lookup
            else {"outcome": trace.attributes.get("cache")},
            "route": prepared.route.as_dict() if prepared.route else None,
        }
        yield "meta", meta
        if prepared.lookup is not None and prepared.lookup.outcome == "hit":
            payload = pipeline._hit_payload(prepared)
            trace.ttft_ms = trace.elapsed_ms()
            yield "delta", {"text": payload["answer"]}
            yield "done", pipeline._finish(prepared, payload)
            return
        assert prepared.route is not None
        yield "sources", {"sources": pipeline._sources_payload(prepared)}
        # The refusal token must never reach the reader as raw text: the first characters are
        # held back until they can no longer be the start of it.
        held = ""
        deciding, refusing = True, False
        result: CascadeResult | None = None
        try:
            async with pipeline.budget.model_slot():
                async for item in pipeline.cascade.stream(
                    pipeline._request_factory(prepared),
                    trace,
                    tier=prepared.route.tier,
                    before_call=pipeline._charge(prepared.options.client),
                ):
                    if isinstance(item, CascadeResult):
                        result = item
                    elif refusing:
                        continue
                    elif not deciding:
                        yield "delta", {"text": item.text}
                    else:
                        held += item.text
                        probe = _probe(held)
                        if probe.startswith(REFUSAL_TOKEN):
                            deciding, refusing = False, True
                        elif not REFUSAL_TOKEN.startswith(probe):
                            deciding = False
                            yield "delta", {"text": held}
        except MidStreamFailure as failure:
            error = UpstreamUnavailable(
                "The model stopped answering mid-way; please retry.", retry_after_s=10
            )
            logger.warning(
                "Stream failed after text was sent: %s (incident %s)", failure.error, error.incident_id
            )
            pipeline._fail(trace, error)
            yield "error", error.body()["error"]
            return
        except GatewayError as exc:
            pipeline._fail(trace, exc)
            yield "error", exc.body()["error"]
            return
        assert result is not None
        problem = pipeline._validator(prepared)(result.completion)
        payload = pipeline._miss_payload(prepared, result, problem)
        if payload["refused"]:
            yield "delta", {"text": REFUSAL_MESSAGE}
        elif deciding and held:
            yield "delta", {"text": held}  # a short answer that was still being held back
        await pipeline._store(prepared, payload, result.completion)
        yield "done", pipeline._finish(prepared, payload)


def _probe(text: str) -> str:
    return text.lstrip().upper().replace(" ", "_")
