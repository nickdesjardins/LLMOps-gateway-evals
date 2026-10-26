"""Multi-model failover: a request is answered by the first model of its tier that can.

The plan for a tier is its own chain, then (with ``CROSS_TIER_FALLBACK``) the other tier's
chain: a light request whose models are all rate limited escalates to the heavy models, and a
heavy request degrades to the light ones rather than failing. Every attempt is a generation
span in the trace, failed ones included, with the reason it failed.

What moves the cascade on, and what stops it:

======================  ==============================================================
429, timeout, 5xx       next model (another model, often another upstream, may have room)
404                     next model (free models are withdrawn without notice)
400                     next model (this model's endpoint refused the request's shape)
invalid output          next model (empty, cut off before any text, failed validation)
refusal                 next model
401 / 403               stop: every model of the provider shares the key
deadline                stop: ``REQUEST_DEADLINE_S`` covers the whole cascade
======================  ==============================================================

Streaming can fail over only until the first token has been sent: after that, the client has
part of one model's answer, and splicing another model's answer onto it would be worse than an
error event.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from app.config import Tier
from app.errors import UpstreamFailed, UpstreamUnavailable
from app.pricing import PriceBook
from app.providers.base import FATAL_KINDS, ChatRequest, Completion, Provider, ProviderError, StreamEvent
from app.telemetry.tracer import SpanHandle, Trace, preview

#: Failure kinds that a retry later will not cure.
_PERMANENT = frozenset({"auth", "not_found", "bad_request"})

RequestFactory = Callable[[Tier], ChatRequest]
Validator = Callable[[Completion], str | None]
BeforeCall = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class Attempt:
    model: str
    tier: Tier
    outcome: str  # "ok" or a failure kind
    detail: str = ""
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "tier": self.tier,
            "outcome": self.outcome,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class CascadeResult:
    completion: Completion
    tier: Tier
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return len(self.attempts) > 1


class MidStreamFailure(Exception):
    """The answering model failed after text had already been streamed."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(str(error))
        self.error = error


class ModelCascade:
    def __init__(
        self,
        provider: Provider,
        chains: dict[Tier, list[str]],
        *,
        prices: PriceBook,
        reference_models: dict[Tier, str],
        cross_tier: bool = True,
        deadline_s: float = 90.0,
        model_timeout_s: float = 45.0,
    ) -> None:
        self.provider = provider
        self.chains = chains
        self.prices = prices
        self.reference_models = reference_models
        self.cross_tier = cross_tier
        self.deadline_s = deadline_s
        self.model_timeout_s = model_timeout_s

    def plan(self, tier: Tier) -> list[tuple[Tier, str]]:
        other: Tier = "heavy" if tier == "light" else "light"
        steps = [(tier, model) for model in self.chains[tier]]
        if self.cross_tier:
            steps += [(other, model) for model in self.chains[other] if (other, model) not in steps]
        return steps

    # ---------------------------------------------------------------- telemetry
    def _open(
        self, trace: Trace, tier: Tier, model: str, number: int, request: ChatRequest, planned: Tier
    ) -> SpanHandle:
        return trace.start_span(
            f"llm.attempt {number}",
            "generation",
            model=model,
            tier=tier,
            attempt=number,
            max_tokens=request.max_tokens,
            provider=self.provider.name,
            cross_tier=tier != planned,
        )

    def _close_ok(self, span: SpanHandle, completion: Completion, tier: Tier) -> None:
        cost = completion.provider_cost_usd
        if cost is None:
            cost = self.prices.cost(completion.model, completion.usage)
        reference = self.prices.reference_cost(self.reference_models[tier], completion.usage)
        span.generation_result(
            usage=completion.usage,
            cost_usd=cost,
            reference_cost_usd=reference,
            ttft_ms=completion.ttft_ms,
        )
        span.end(
            "ok",
            finish_reason=completion.finish_reason,
            output_preview=preview(completion.text, 600),
            cost_source="provider" if completion.provider_cost_usd is not None else "price book",
        )

    def _close_failed(self, span: SpanHandle, error: ProviderError, tier: Tier) -> None:
        if error.usage.total_tokens:
            span.generation_result(
                usage=error.usage,
                cost_usd=self.prices.cost(error.model, error.usage),
                reference_cost_usd=self.prices.reference_cost(self.reference_models[tier], error.usage),
            )
        span.end("error", failure=error.kind, detail=error.detail)

    @staticmethod
    def _give_up(attempts: list[Attempt], reason: str | None = None) -> Exception:
        kinds = {a.outcome for a in attempts}
        if reason == "deadline":
            return UpstreamUnavailable(
                "No model answered within the request deadline; please retry.", retry_after_s=10
            )
        if kinds and kinds <= _PERMANENT:
            return UpstreamFailed("No configured model could take this request.")
        return UpstreamUnavailable(
            "Every model is busy or failing right now; please retry in a moment.", retry_after_s=10
        )

    # ------------------------------------------------------------------ one shot
    async def complete(
        self,
        make_request: RequestFactory,
        trace: Trace,
        *,
        tier: Tier,
        validate: Validator | None = None,
        before_call: BeforeCall | None = None,
    ) -> CascadeResult:
        started = time.perf_counter()
        attempts: list[Attempt] = []
        for number, (step_tier, model) in enumerate(self.plan(tier), start=1):
            remaining = self.deadline_s - (time.perf_counter() - started)
            if remaining <= 1.0:
                raise self._give_up(attempts, "deadline")
            if before_call is not None:
                await before_call(model)
            request = make_request(step_tier)
            span = self._open(trace, step_tier, model, number, request, tier)
            attempt_started = time.perf_counter()
            try:
                async with asyncio.timeout(min(self.model_timeout_s, remaining)):
                    completion = await self.provider.complete(model, request)
                problem = validate(completion) if validate is not None else None
                if problem:
                    raise ProviderError(model, "invalid_output", problem, usage=completion.usage)
            except TimeoutError:
                error = ProviderError(model, "timeout", "no answer before the timeout")
                self._close_failed(span, error, step_tier)
                attempts.append(Attempt(model, step_tier, "timeout", error.detail, _ms(attempt_started)))
                continue
            except ProviderError as error:
                self._close_failed(span, error, step_tier)
                attempts.append(Attempt(model, step_tier, error.kind, error.detail, _ms(attempt_started)))
                if error.kind in FATAL_KINDS:
                    raise UpstreamFailed("The model provider refused the gateway's credentials.") from error
                continue
            self._close_ok(span, completion, step_tier)
            attempts.append(Attempt(model, step_tier, "ok", "", _ms(attempt_started)))
            return CascadeResult(completion=completion, tier=step_tier, attempts=attempts)
        raise self._give_up(attempts)

    # ------------------------------------------------------------------ streaming
    async def stream(
        self,
        make_request: RequestFactory,
        trace: Trace,
        *,
        tier: Tier,
        before_call: BeforeCall | None = None,
    ) -> AsyncIterator[StreamEvent | CascadeResult]:
        """Yields text deltas, then one CascadeResult. Raises MidStreamFailure after text."""
        started = time.perf_counter()
        attempts: list[Attempt] = []
        for number, (step_tier, model) in enumerate(self.plan(tier), start=1):
            remaining = self.deadline_s - (time.perf_counter() - started)
            if remaining <= 1.0:
                raise self._give_up(attempts, "deadline")
            if before_call is not None:
                await before_call(model)
            request = make_request(step_tier)
            span = self._open(trace, step_tier, model, number, request, tier)
            attempt_started = time.perf_counter()
            sent_text = False
            completion: Completion | None = None
            try:
                async for event in self.provider.stream(model, request):
                    if event.kind == "delta":
                        if not sent_text and trace.ttft_ms is None:
                            trace.ttft_ms = trace.elapsed_ms()
                        sent_text = True
                        yield event
                    elif event.completion is not None:
                        completion = event.completion
                if completion is None:
                    raise ProviderError(model, "invalid_output", "the stream ended without a final message")
            except ProviderError as error:
                self._close_failed(span, error, step_tier)
                attempts.append(Attempt(model, step_tier, error.kind, error.detail, _ms(attempt_started)))
                if sent_text:
                    raise MidStreamFailure(error) from error
                if error.kind in FATAL_KINDS:
                    raise UpstreamFailed("The model provider refused the gateway's credentials.") from error
                continue
            self._close_ok(span, completion, step_tier)
            attempts.append(Attempt(model, step_tier, "ok", "", _ms(attempt_started)))
            yield CascadeResult(completion=completion, tier=step_tier, attempts=attempts)
            return
        raise self._give_up(attempts)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
