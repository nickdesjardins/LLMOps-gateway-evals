"""The real SDK adapters, against a mock HTTP transport: request shapes and error mapping."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import anthropic
import httpx2
import openai
import pytest

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import ChatRequest, ProviderError
from app.providers.offline import OfflineProvider, extractive_answer
from app.providers.openai_compat import OpenAICompatibleProvider
from app.rag.prompts import REFUSAL_TOKEN, PromptSource, build_messages

REQUEST = ChatRequest(
    messages=[{"role": "system", "content": "static prefix"}, {"role": "user", "content": "question"}],
    max_tokens=350,
    tier="light",
)


def completion(
    content: str | None = "Twenty days [S1].",
    finish: str = "stop",
    usage: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": "vendor/model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": usage
        or {"prompt_tokens": 900, "completion_tokens": 12, "total_tokens": 912, "cost": 0.000123},
    }
    body.update(extra)
    return body


def openai_provider(name: str, handler: Any) -> OpenAICompatibleProvider:
    client = openai.AsyncOpenAI(
        api_key="test-only",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAICompatibleProvider(
        name, api_key="test-only", base_url="https://provider.invalid/v1", timeout_s=5, client=client
    )


async def test_openrouter_request_shape_usage_and_cost() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=completion())

    result = await openai_provider("openrouter", handler).complete("vendor/model:free", REQUEST)
    sent = seen[0]
    assert sent["max_tokens"] == 350 and sent["model"] == "vendor/model:free"
    assert sent["reasoning"] == {"enabled": False} and sent["usage"] == {"include": True}
    assert sent["messages"][0] == {"role": "system", "content": "static prefix"}
    assert result.text == "Twenty days [S1]." and result.model == "vendor/model:free"
    assert result.usage.input_tokens == 900 and result.provider_cost_usd == Decimal("0.000123")


async def test_openai_uses_max_completion_tokens_and_no_temperature() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=completion(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 8},
                }
            ),
        )

    result = await openai_provider("openai", handler).complete("small-model", REQUEST)
    assert (
        seen[0]["max_completion_tokens"] == 350
        and "max_tokens" not in seen[0]
        and "temperature" not in seen[0]
    )
    assert "reasoning" not in seen[0]
    assert result.usage.cached_input_tokens == 8 and result.provider_cost_usd is None


async def test_extensions_are_dropped_after_a_400_and_remembered() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning" in body:
            return httpx2.Response(400, json={"error": {"message": "reasoning is not supported"}})
        return httpx2.Response(200, json=completion())

    provider = openai_provider("openrouter", handler)
    await provider.complete("vendor/model:free", REQUEST)
    await provider.complete("vendor/model:free", REQUEST)
    assert ["reasoning" in b for b in bodies] == [True, False, False]


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (429, "rate_limited"),
        (404, "not_found"),
        (401, "auth"),
        (403, "auth"),
        (500, "upstream"),
        (503, "upstream"),
        (422, "bad_request"),
    ],
)
async def test_http_errors_become_failure_kinds(status: int, kind: str) -> None:
    provider = openai_provider(
        "openai", lambda request: httpx2.Response(status, json={"error": {"message": "secret detail"}})
    )
    with pytest.raises(ProviderError) as caught:
        await provider.complete("m", REQUEST)
    assert caught.value.kind == kind and "secret detail" not in caught.value.detail


async def test_openrouter_in_band_errors_and_empty_answers() -> None:
    in_band = openai_provider(
        "openrouter",
        lambda r: httpx2.Response(200, json={"error": {"code": 502, "message": "upstream"}, "choices": []}),
    )
    with pytest.raises(ProviderError) as caught:
        await in_band.complete("m:free", REQUEST)
    assert caught.value.kind == "upstream"
    thinking = openai_provider(
        "openrouter", lambda r: httpx2.Response(200, json=completion(content="", finish="length"))
    )
    with pytest.raises(ProviderError) as caught:
        await thinking.complete("m:free", REQUEST)
    assert caught.value.kind == "invalid_output" and "thinking" in caught.value.detail
    filtered = openai_provider(
        "openrouter", lambda r: httpx2.Response(200, json=completion(finish="content_filter"))
    )
    with pytest.raises(ProviderError) as caught:
        await filtered.complete("m:free", REQUEST)
    assert caught.value.kind == "refused"


async def test_openai_compatible_streaming_with_usage() -> None:
    def chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, Any] | None = None) -> str:
        data = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n"

    stream = (
        chunk({"role": "assistant", "content": ""})
        + chunk({"content": "Twenty "})
        + chunk({"content": "days [S1]."}, "stop")
        + chunk({}, usage={"prompt_tokens": 900, "completion_tokens": 5, "total_tokens": 905})
        + "data: [DONE]\n\n"
    )
    provider = openai_provider(
        "openrouter",
        lambda r: httpx2.Response(200, text=stream, headers={"content-type": "text/event-stream"}),
    )
    events = [e async for e in provider.stream("m:free", REQUEST)]
    assert [e.text for e in events if e.kind == "delta"] == ["Twenty ", "days [S1]."]
    final = events[-1].completion
    assert (
        final is not None
        and final.text == "Twenty days [S1]."
        and final.usage.output_tokens == 5
        and final.ttft_ms is not None
    )


def anthropic_provider(handler: Any) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-test-only",
        base_url="https://anthropic.invalid",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(api_key="sk-ant-test-only", timeout_s=5, client=client)


def message(
    text: str = "Twenty days [S1].", stop: str = "end_turn", model: str = "claude-haiku-5-5"
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 12,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        },
    }


async def test_anthropic_caches_the_static_prefix_and_sets_effort_by_tier() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx2.Response(200, json=message())

    provider = anthropic_provider(handler)
    result = await provider.complete("claude-haiku-5-5", REQUEST)
    url, body = seen[0]
    assert body["system"] == [
        {"type": "text", "text": "static prefix", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "question"}]
    assert (
        body["output_config"] == {"effort": "low"} and "temperature" not in body and "fallbacks" not in body
    )
    assert result.usage.input_tokens == 900 and result.usage.cached_input_tokens == 800  # normalized
    heavy = ChatRequest(messages=REQUEST.messages, max_tokens=900, tier="heavy")
    await provider.complete("claude-opus-5-5", heavy)
    url, body = seen[1]
    assert body["output_config"] == {"effort": "medium"} and body["fallbacks"] == "default"
    assert "beta=true" in url


async def test_anthropic_refusals_and_errors() -> None:
    refused = anthropic_provider(lambda r: httpx2.Response(200, json=message(text="", stop="refusal")))
    with pytest.raises(ProviderError) as caught:
        await refused.complete("claude-haiku-5-5", REQUEST)
    assert caught.value.kind == "refused"
    limited = anthropic_provider(
        lambda r: httpx2.Response(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}
        )
    )
    with pytest.raises(ProviderError) as caught:
        await limited.complete("claude-haiku-5-5", REQUEST)
    assert caught.value.kind == "rate_limited"


def test_the_offline_provider_quotes_the_best_passage_or_refuses() -> None:
    sources = [
        PromptSource(
            "S1",
            "leave-policy#vacation",
            "Leave > Vacation",
            "Full-time employees accrue 20 vacation days per year. More text.",
        ),
        PromptSource("S2", "it#vpn", "IT > VPN", "The VPN is required outside the office."),
    ]
    messages = build_messages("How many vacation days do full-time employees accrue per year?", sources)
    assert extractive_answer(messages[1]["content"]).endswith("[S1]")
    off_topic = build_messages("What is the capital of Australia?", sources)
    assert extractive_answer(off_topic[1]["content"]) == REFUSAL_TOKEN


async def test_offline_streams_and_never_pretends_to_be_a_model() -> None:
    provider = OfflineProvider()
    events = [e async for e in provider.stream("anything", REQUEST)]
    assert events[-1].completion is not None and events[-1].completion.model == "offline/extractive"
    assert events[-1].completion.usage.estimated
