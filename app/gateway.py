"""The OpenAI-compatible reverse proxy: ``POST /v1/chat/completions`` through cache, router, cascade.

Any client that speaks the Chat Completions API (the openai SDK, LangChain, curl) can point its
base URL at the gateway and get, without changing a line, semantic caching, complexity routing,
multi-model failover, token ceilings and a trace per call. The response is a standard
``chat.completion`` (or a stream of ``chat.completion.chunk`` events ending in ``[DONE]``),
with one extra member, ``x_gateway``, and ``X-Gateway-*`` headers that say what happened.

``model`` selects a tier, not a vendor: ``auto`` lets the router decide, ``light`` and ``heavy``
force a tier, and outside demo mode a model id from one of the chains starts the cascade at that
model. Only single-turn requests (one user message, an optional system prompt) are cached: an
answer to a conversation depends on every turn of it. The system prompt is part of the cache
namespace, so two applications with different instructions never share answers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.budget import DemoBudget
from app.cache.semantic import LookupResult, SemanticCache, namespace_for
from app.config import Settings, Tier
from app.errors import (
    GatewayError,
    InvalidRequest,
    PromptInjectionRefused,
    PromptTooLong,
    UpstreamUnavailable,
)
from app.guard import GuardReport, clamp_max_tokens, defang, estimate_messages_tokens
from app.pricing import usd
from app.providers.base import ChatRequest, Completion
from app.routing.cascade import BeforeCall, CascadeResult, MidStreamFailure, ModelCascade, RequestFactory
from app.routing.complexity import ComplexityRouter, RouteDecision, TaskHint
from app.telemetry.tracer import Telemetry, Trace

logger = logging.getLogger(__name__)

TIER_ALIASES: dict[str, Tier | None] = {"auto": None, "light": "light", "heavy": "heavy"}


@dataclass
class ChatPrepared:
    trace: Trace
    messages: list[dict[str, str]]
    report: GuardReport
    max_tokens: int | None
    cacheable: bool
    namespace: str
    cache_mode: str
    client: str
    lookup: LookupResult | None = None
    route: RouteDecision | None = None
    first_model: str | None = None
    created: int = field(default_factory=lambda: int(time.time()))


class ChatGateway:
    def __init__(
        self,
        *,
        settings: Settings,
        cache: SemanticCache | None,
        router: ComplexityRouter,
        cascade: ModelCascade,
        telemetry: Telemetry,
        budget: DemoBudget,
        embedder_id: str,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.router = router
        self.cascade = cascade
        self.telemetry = telemetry
        self.budget = budget
        self.embedder_id = embedder_id

    # --------------------------------------------------------------- preparing
    def _tier_for(self, model: str) -> tuple[Tier | None, str | None]:
        """(forced tier, first model) for the request's ``model`` field."""
        key = model.strip().lower()
        if key in TIER_ALIASES:
            return TIER_ALIASES[key], None
        if self.settings.demo_mode:
            return None, None  # visitors pick tiers, never models
        tiers: tuple[Tier, ...] = ("light", "heavy")
        for tier in tiers:
            if model in self.cascade.chains[tier]:
                return tier, model
        raise InvalidRequest(
            "Unknown model: use auto, light, heavy, or an id from LIGHT_MODELS or HEAVY_MODELS."
        )

    async def prepare(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None,
        task: TaskHint,
        cache_mode: str,
        client: str,
        stream: bool,
    ) -> ChatPrepared:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        trace = self.telemetry.start_trace(
            "gateway.chat",
            input_text=last_user,
            user_id=client,
            endpoint="/v1/chat/completions",
            stream=stream,
            turns=len(messages),
        )
        try:
            if not last_user.strip():
                raise InvalidRequest("The request needs at least one user message with text.")
            forced, first_model = self._tier_for(model)
            with trace.span("guard.defang", "guardrail") as span:
                clean: list[dict[str, str]] = []
                signals: set[str] = set()
                neutralized = removed = 0
                for message in messages:
                    if message["role"] == "user":
                        text, report = defang(message["content"])
                        signals.update(report.signals)
                        neutralized += report.neutralized
                        removed += report.removed_characters
                        clean.append({"role": "user", "content": text})
                    else:
                        clean.append(dict(message))
                report = GuardReport(tuple(sorted(signals)), neutralized, removed)
                span.set(**report.as_dict(), policy=self.settings.injection_policy)
            trace.set(injection_detected=report.suspicious, injection_signals=list(report.signals))
            if report.suspicious and self.settings.injection_policy == "block":
                raise PromptInjectionRefused(
                    "The request was refused: it contains text that tries to change the model's instructions."
                )
            if estimate_messages_tokens(clean) > self.settings.max_input_tokens:
                raise PromptTooLong(
                    f"The prompt is estimated above the {self.settings.max_input_tokens}-token input ceiling."
                )
            users = [m for m in clean if m["role"] == "user"]
            single_turn = len(users) == 1 and all(m["role"] in {"system", "user"} for m in clean)
            system = "\n\n".join(m["content"] for m in clean if m["role"] == "system")
            namespace = namespace_for(
                "chat",
                hashlib.sha256(system.encode("utf-8")).hexdigest(),
                str(forced or "auto"),
                first_model or "",
                self.embedder_id,
            )
            prepared = ChatPrepared(
                trace=trace,
                messages=clean,
                report=report,
                max_tokens=max_tokens,
                cacheable=single_turn,
                namespace=namespace,
                cache_mode=cache_mode,
                client=client,
                first_model=first_model,
            )
            if self.cache is None:
                trace.set(cache="off")
            elif not single_turn:
                trace.set(cache="bypass", cache_bypass_reason="multi-turn")
            elif cache_mode != "use":
                trace.set(cache="bypass", cache_mode=cache_mode)
            else:
                with trace.span("cache.lookup", "embedding", backend=self.cache.store.name) as span:
                    prepared.lookup = await self.cache.lookup(namespace, users[0]["content"])
                    span.set(**prepared.lookup.as_dict())
                trace.set(cache=prepared.lookup.outcome, cache_lookup_ms=round(prepared.lookup.total_ms, 3))
                if prepared.lookup.outcome == "hit":
                    return prepared
            self.budget.check_model_call(client)
            with trace.span("router.complexity") as span:
                prepared.route = self.router.route(
                    users[-1]["content"], task=task, turns=len(clean), forced_tier=forced
                )
                span.set(**prepared.route.as_dict())
            trace.set(tier=prepared.route.tier, route_score=round(prepared.route.score, 3))
            return prepared
        except GatewayError as exc:
            self._fail(trace, exc)
            raise

    def _fail(self, trace: Trace, exc: GatewayError) -> None:
        trace.set(error_type=exc.error_type, incident_id=exc.incident_id)
        self.telemetry.finish(trace, status="error")

    def _cascade_for(self, prepared: ChatPrepared) -> ModelCascade:
        if prepared.first_model is None or prepared.route is None:
            return self.cascade
        tier = prepared.route.tier
        chain = [prepared.first_model] + [m for m in self.cascade.chains[tier] if m != prepared.first_model]
        return ModelCascade(
            self.cascade.provider,
            {**self.cascade.chains, tier: chain},
            prices=self.cascade.prices,
            reference_models=self.cascade.reference_models,
            cross_tier=self.cascade.cross_tier,
            deadline_s=self.cascade.deadline_s,
            model_timeout_s=self.cascade.model_timeout_s,
        )

    def _factory(self, prepared: ChatPrepared) -> RequestFactory:
        def make(tier: Tier) -> ChatRequest:
            ceiling = self.settings.max_output_tokens(tier)
            return ChatRequest(
                messages=prepared.messages,
                max_tokens=clamp_max_tokens(prepared.max_tokens, ceiling),
                tier=tier,
                tag="chat",
            )

        return make

    def _charge(self, client: str) -> BeforeCall:
        async def before_call(_model: str) -> None:
            await self.budget.charge_model_call(client)

        return before_call

    # ---------------------------------------------------------------- payloads
    def _gateway_block(self, prepared: ChatPrepared, result: CascadeResult | None) -> dict[str, Any]:
        trace = prepared.trace
        return {
            "trace_id": trace.trace_id,
            "cache": prepared.lookup.as_dict()
            if prepared.lookup
            else {"outcome": trace.attributes.get("cache")},
            "route": prepared.route.as_dict() if prepared.route else None,
            "tier": result.tier if result else trace.attributes.get("tier"),
            "attempts": [a.as_dict() for a in result.attempts] if result else [],
            "fallback": bool(result and result.fell_back),
            "cost": {
                "actual_usd": usd(trace.cost() if result else Decimal(0)),
                "reference_usd": usd(trace.reference_cost()),
                "avoided_usd": usd(trace.cost_avoided_usd),
            },
            "injection": prepared.report.as_dict(),
            "max_tokens_sent": None,
        }

    def _completion_body(
        self,
        prepared: ChatPrepared,
        text: str,
        model: str,
        finish: str,
        usage: dict[str, int],
        gateway: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{prepared.trace.trace_id}",
            "object": "chat.completion",
            "created": prepared.created,
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            ],
            "usage": usage,
            "x_gateway": gateway,
        }

    def _hit(self, prepared: ChatPrepared) -> tuple[str, str, dict[str, Any]]:
        assert prepared.lookup is not None and prepared.lookup.entry is not None
        entry = prepared.lookup.entry
        prepared.trace.cost_avoided_usd = Decimal(entry.reference_cost_usd)
        prepared.trace.set(tier=entry.payload.get("tier"), origin_model=entry.payload.get("model"))
        gateway = self._gateway_block(prepared, None)
        gateway["cache"] |= {"cached_at": entry.created_at, "origin_trace_id": entry.payload.get("trace_id")}
        gateway["tier"] = entry.payload.get("tier")
        return entry.payload["text"], entry.payload["model"], gateway

    async def _store(self, prepared: ChatPrepared, completion: Completion, tier: Tier) -> None:
        trace = prepared.trace
        if self.cache is None or not prepared.cacheable or prepared.cache_mode == "bypass":
            return
        if prepared.report.suspicious or completion.truncated:
            trace.set(
                cache_write="skipped",
                cache_write_reason="flagged" if prepared.report.suspicious else "cut off",
            )
            return
        user = next(m["content"] for m in prepared.messages if m["role"] == "user")
        vector = prepared.lookup.vector if prepared.lookup is not None else None
        with trace.span("cache.write", backend=self.cache.store.name) as span:
            ok = await self.cache.store_answer(
                prepared.namespace,
                user,
                {
                    "trace_id": trace.trace_id,
                    "text": completion.text,
                    "model": completion.model,
                    "tier": tier,
                },
                reference_cost_usd=trace.reference_cost(),
                actual_cost_usd=trace.cost() or Decimal(0),
                vector=vector,
            )
            span.set(stored=ok)
        trace.set(cache_write="stored" if ok else "refused by the store")

    def _finish(self, prepared: ChatPrepared, text: str, model: str | None) -> None:
        prepared.trace.set(model=model)
        self.telemetry.finish(prepared.trace, output_text=text)

    # -------------------------------------------------------------------- JSON
    async def complete(self, prepared: ChatPrepared) -> dict[str, Any]:
        if prepared.lookup is not None and prepared.lookup.outcome == "hit":
            text, model, gateway = self._hit(prepared)
            self._finish(prepared, text, model)
            gateway["latency_ms"] = round(prepared.trace.latency_ms or 0, 2)
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return self._completion_body(prepared, text, model, "stop", usage, gateway)
        assert prepared.route is not None
        try:
            async with self.budget.model_slot():
                result = await self._cascade_for(prepared).complete(
                    self._factory(prepared),
                    prepared.trace,
                    tier=prepared.route.tier,
                    before_call=self._charge(prepared.client),
                )
        except GatewayError as exc:
            self._fail(prepared.trace, exc)
            raise
        completion = result.completion
        await self._store(prepared, completion, result.tier)
        gateway = self._gateway_block(prepared, result)
        gateway["max_tokens_sent"] = self._factory(prepared)(result.tier).max_tokens
        self._finish(prepared, completion.text, completion.model)
        gateway["latency_ms"] = round(prepared.trace.latency_ms or 0, 2)
        usage = {
            "prompt_tokens": completion.usage.input_tokens,
            "completion_tokens": completion.usage.output_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        finish = "length" if completion.truncated else "stop"
        return self._completion_body(prepared, completion.text, completion.model, finish, usage, gateway)

    # ------------------------------------------------------------------ stream
    def _chunk(
        self,
        prepared: ChatPrepared,
        model: str,
        delta: dict[str, Any],
        finish: str | None = None,
        **extra: Any,
    ) -> str:
        body = {
            "id": f"chatcmpl-{prepared.trace.trace_id}",
            "object": "chat.completion.chunk",
            "created": prepared.created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return f"data: {json.dumps(body, separators=(',', ':'))}\n\n"

    async def stream(self, prepared: ChatPrepared) -> AsyncIterator[str]:
        """``data:`` lines in the Chat Completions streaming format, ending with ``[DONE]``."""
        trace = prepared.trace
        if prepared.lookup is not None and prepared.lookup.outcome == "hit":
            text, model, gateway = self._hit(prepared)
            trace.ttft_ms = trace.elapsed_ms()
            yield self._chunk(prepared, model, {"role": "assistant", "content": text})
            self._finish(prepared, text, model)
            gateway["latency_ms"] = round(trace.latency_ms or 0, 2)
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            yield self._chunk(prepared, model, {}, "stop", usage=usage, x_gateway=gateway)
            yield "data: [DONE]\n\n"
            return
        assert prepared.route is not None
        model_name = prepared.first_model or prepared.route.tier
        result: CascadeResult | None = None
        started = False
        try:
            async with self.budget.model_slot():
                async for item in self._cascade_for(prepared).stream(
                    self._factory(prepared),
                    trace,
                    tier=prepared.route.tier,
                    before_call=self._charge(prepared.client),
                ):
                    if isinstance(item, CascadeResult):
                        result = item
                        continue
                    delta: dict[str, Any] = {"content": item.text}
                    if not started:
                        delta["role"] = "assistant"
                        started = True
                    yield self._chunk(prepared, model_name, delta)
        except MidStreamFailure as failure:
            error = UpstreamUnavailable(
                "The model stopped answering mid-way; please retry.", retry_after_s=10
            )
            logger.warning(
                "Chat stream failed after text was sent: %s (incident %s)", failure.error, error.incident_id
            )
            self._fail(trace, error)
            yield f"data: {json.dumps(error.body())}\n\n"
            yield "data: [DONE]\n\n"
            return
        except GatewayError as exc:
            self._fail(trace, exc)
            yield f"data: {json.dumps(exc.body())}\n\n"
            yield "data: [DONE]\n\n"
            return
        assert result is not None
        completion = result.completion
        await self._store(prepared, completion, result.tier)
        gateway = self._gateway_block(prepared, result)
        gateway["max_tokens_sent"] = self._factory(prepared)(result.tier).max_tokens
        self._finish(prepared, completion.text, completion.model)
        gateway["latency_ms"] = round(trace.latency_ms or 0, 2)
        gateway["ttft_ms"] = None if trace.ttft_ms is None else round(trace.ttft_ms, 2)
        usage = {
            "prompt_tokens": completion.usage.input_tokens,
            "completion_tokens": completion.usage.output_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        finish = "length" if completion.truncated else "stop"
        yield self._chunk(prepared, completion.model, {}, finish, usage=usage, x_gateway=gateway)
        yield "data: [DONE]\n\n"
