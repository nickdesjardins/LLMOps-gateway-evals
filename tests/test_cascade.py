"""Multi-model failover: what moves the cascade on, what stops it, and what the trace records."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from app.errors import UpstreamFailed, UpstreamUnavailable
from app.pricing import PriceBook, Usage
from app.providers.base import ChatRequest, Completion, ProviderError
from app.routing.cascade import CascadeResult, MidStreamFailure, ModelCascade
from app.telemetry.tracer import Telemetry
from tests.conftest import ScriptedProvider, rate_limited

CHAINS = {"light": ["l1:free", "l2:free"], "heavy": ["h1:free", "h2:free"]}
REFS = {"light": "claude-haiku-5-5", "heavy": "claude-opus-5-5"}


def make(provider: ScriptedProvider, **kwargs: object) -> ModelCascade:
    return ModelCascade(provider, CHAINS, prices=PriceBook(), reference_models=REFS, **kwargs)  # type: ignore[arg-type]


def factory(tier: str) -> ChatRequest:
    return ChatRequest(
        messages=[{"role": "user", "content": "q"}], max_tokens=100 if tier == "light" else 400, tier=tier
    )  # type: ignore[arg-type]


def trace():  # type: ignore[no-untyped-def]
    return Telemetry().start_trace("test")


def test_the_plan_is_the_tier_then_the_other_tier() -> None:
    cascade = make(ScriptedProvider())
    assert cascade.plan("light") == [
        ("light", "l1:free"),
        ("light", "l2:free"),
        ("heavy", "h1:free"),
        ("heavy", "h2:free"),
    ]
    assert cascade.plan("heavy")[:2] == [("heavy", "h1:free"), ("heavy", "h2:free")]
    assert make(ScriptedProvider(), cross_tier=False).plan("light") == [
        ("light", "l1:free"),
        ("light", "l2:free"),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        rate_limited("l1:free"),
        ProviderError("l1:free", "timeout", "timed out"),
        ProviderError("l1:free", "upstream", "HTTP 502", status=502),
        ProviderError("l1:free", "not_found", "model not found (404)", status=404),
        ProviderError("l1:free", "bad_request", "rejected (400)", status=400),
        ProviderError("l1:free", "refused", "content filter"),
        ProviderError("l1:free", "invalid_output", "empty answer"),
    ],
)
async def test_each_retryable_failure_moves_to_the_next_model(failure: ProviderError) -> None:
    provider = ScriptedProvider({"l1:free": [failure]})
    t = trace()
    result = await make(provider).complete(factory, t, tier="light")
    assert result.completion.model == "l2:free"
    assert [a.outcome for a in result.attempts] == [failure.kind, "ok"]
    assert result.fell_back
    spans = t.generations
    assert [s.status for s in spans] == ["error", "ok"]
    assert spans[0].attributes["failure"] == failure.kind


async def test_a_validation_failure_is_a_failure_too() -> None:
    provider = ScriptedProvider({"l1:free": ["no citation here"], "l2:free": ["cited [S1]."]})
    result = await make(provider).complete(
        factory, trace(), tier="light", validate=lambda c: None if "[S1]" in c.text else "no citation"
    )
    assert result.completion.text == "cited [S1]."
    assert result.attempts[0].outcome == "invalid_output" and result.attempts[0].detail == "no citation"


async def test_light_escalates_to_heavy_with_the_heavy_ceiling() -> None:
    provider = ScriptedProvider({"l1:free": [rate_limited()], "l2:free": [rate_limited()]})
    result = await make(provider).complete(factory, trace(), tier="light")
    assert result.tier == "heavy" and result.completion.model == "h1:free"
    assert provider.calls[-1][1].max_tokens == 400 and provider.calls[0][1].max_tokens == 100


async def test_a_bad_key_stops_the_cascade() -> None:
    provider = ScriptedProvider(
        {"l1:free": [ProviderError("l1:free", "auth", "the key was refused (401)", status=401)]}
    )
    with pytest.raises(UpstreamFailed, match="credentials"):
        await make(provider).complete(factory, trace(), tier="light")
    assert len(provider.calls) == 1


async def test_everything_busy_is_a_503_with_retry_after() -> None:
    provider = ScriptedProvider({m: [rate_limited(m)] for chain in CHAINS.values() for m in chain})
    with pytest.raises(UpstreamUnavailable) as caught:
        await make(provider).complete(factory, trace(), tier="light")
    assert caught.value.status_code == 503 and caught.value.retry_after_s == 10
    assert len(provider.calls) == 4


async def test_everything_gone_is_a_502() -> None:
    gone = {m: [ProviderError(m, "not_found", "404", status=404)] for chain in CHAINS.values() for m in chain}
    with pytest.raises(UpstreamFailed) as caught:
        await make(ScriptedProvider(gone)).complete(factory, trace(), tier="heavy")
    assert caught.value.status_code == 502


async def test_a_slow_model_is_cut_off_by_the_timeout() -> None:
    class Slow(ScriptedProvider):
        async def complete(self, model: str, request: ChatRequest) -> Completion:
            if model == "l1:free":
                await asyncio.sleep(5)
            return await super().complete(model, request)

    result = await make(Slow(), model_timeout_s=0.05).complete(factory, trace(), tier="light")
    assert result.attempts[0].outcome == "timeout" and result.completion.model == "l2:free"


async def test_the_deadline_bounds_the_whole_cascade() -> None:
    with pytest.raises(UpstreamUnavailable, match="deadline"):
        await make(ScriptedProvider(), deadline_s=0.5).complete(factory, trace(), tier="light")


async def test_costs_come_from_the_provider_or_the_price_book() -> None:
    provider = ScriptedProvider()
    provider.usage = Usage(input_tokens=1_000, output_tokens=100)
    t = trace()
    await make(provider).complete(factory, t, tier="heavy")
    span = t.generations[0]
    assert span.cost_usd == 0  # a :free model
    assert span.reference_cost_usd == Decimal("0.006")  # opus: 1000*4 + 100*20, per million
    provider.cost = Decimal("0.0042")
    t2 = trace()
    await make(provider).complete(factory, t2, tier="heavy")
    assert t2.generations[0].cost_usd == Decimal("0.0042")
    assert t2.generations[0].attributes["cost_source"] == "provider"


async def test_before_call_is_charged_for_every_attempt() -> None:
    charged: list[str] = []

    async def charge(model: str) -> None:
        charged.append(model)

    provider = ScriptedProvider({"l1:free": [rate_limited()]})
    await make(provider).complete(factory, trace(), tier="light", before_call=charge)
    assert charged == ["l1:free", "l2:free"]


async def test_streaming_fails_over_before_the_first_token() -> None:
    provider = ScriptedProvider({"h1:free": [rate_limited()], "h2:free": [["Hello ", "world [S1]."]]})
    t = trace()
    items = [item async for item in make(provider).stream(factory, t, tier="heavy")]
    text = "".join(i.text for i in items if not isinstance(i, CascadeResult))
    assert text == "Hello world [S1]."
    result = items[-1]
    assert isinstance(result, CascadeResult) and result.completion.model == "h2:free"
    assert t.ttft_ms is not None


async def test_streaming_cannot_fail_over_after_text_was_sent() -> None:
    provider = ScriptedProvider(
        {"h1:free": [["partial ", ProviderError("h1:free", "upstream", "connection error")]]}
    )
    with pytest.raises(MidStreamFailure):
        async for _ in make(provider).stream(factory, trace(), tier="heavy"):
            pass
    assert len(provider.calls) == 1
