"""The HTTP surface: JSON and SSE bodies, headers, error envelopes, tools, ops routes."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.schemas import AskResponse
from tests.conftest import ScriptedProvider, build_test_app, make_settings, rate_limited

QUESTION = "How many vacation days do full-time employees get each year?"


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def client_for(**settings: Any) -> tuple[httpx.AsyncClient, Any]:
    provider = settings.pop("provider", None)
    app, components = await build_test_app(make_settings(**settings), provider=provider)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ), components


async def test_liveness_and_health(client_and_components: Any) -> None:
    client, _ = client_and_components
    assert (await client.get("/livez")).json() == {"status": "alive"}
    health = (await client.get("/health")).json()
    assert health["status"] == "ok" and health["cache"]["backend"] == "memory"
    assert health["corpus"]["documents"] == 7 and health["provider"]["configured"]
    assert health["cache"]["threshold"] == 0.96


async def test_ask_returns_the_validated_body_and_gateway_headers(client_and_components: Any) -> None:
    client, _ = client_and_components
    response = await client.post("/api/v1/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    assert response.headers["x-gateway-cache"] == "miss"
    assert response.headers["x-gateway-tier"] == body.tier
    assert response.headers["x-gateway-trace-id"] == body.trace_id
    assert response.headers["x-content-type-options"] == "nosniff"
    again = await client.post("/api/v1/ask", json={"question": QUESTION})
    assert again.headers["x-gateway-cache"] == "hit" and again.json()["cache"]["match"] == "exact"


async def test_stream_events_in_order(client_and_components: Any) -> None:
    client, _ = client_and_components
    response = await client.post("/api/v1/ask/stream", json={"question": QUESTION})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
    assert set(names[2:-1]) == {"delta"}
    done = AskResponse.model_validate(events[-1][1])
    assert "".join(d["text"] for n, d in events if n == "delta").strip() == done.answer
    assert done.ttft_ms is not None and done.ttft_ms <= done.latency_ms
    hit = parse_sse((await client.post("/api/v1/ask/stream", json={"question": QUESTION})).text)
    assert [n for n, _ in hit] == ["meta", "delta", "done"] and hit[0][1]["cache"]["outcome"] == "hit"


async def test_a_failure_during_the_stream_is_an_error_event() -> None:
    every = {m: [rate_limited(m)] for m in ("light/a:free", "light/b:free", "heavy/a:free", "heavy/b:free")}
    client, _ = await client_for(provider=ScriptedProvider(every))
    events = parse_sse((await client.post("/api/v1/ask/stream", json={"question": QUESTION})).text)
    assert events[-1][0] == "error" and events[-1][1]["type"] == "upstream_unavailable"
    assert events[-1][1]["incident_id"].startswith("inc_")


async def test_errors_use_one_envelope_and_never_echo_input(client_and_components: Any) -> None:
    client, _ = client_and_components
    secret = "sk-or-v1-should-never-come-back"
    response = await client.post("/api/v1/ask", json={"question": secret, "cache": "sometimes"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request" and "cache" in error["message"]
    assert secret not in response.text
    too_long = await client.post("/api/v1/ask", json={"question": "x" * 600})
    assert too_long.status_code == 400 and "500 characters" in too_long.json()["error"]["message"]


async def test_the_body_limit_comes_before_parsing(client_and_components: Any) -> None:
    client, _ = client_and_components
    response = await client.post(
        "/api/v1/ask", content=b"{" + b" " * 40_000 + b"}", headers={"content-type": "application/json"}
    )
    assert response.status_code == 413 and response.json()["error"]["type"] == "invalid_request"


async def test_upstream_failure_is_a_503_with_retry_after() -> None:
    every = {m: [rate_limited(m)] for m in ("light/a:free", "light/b:free", "heavy/a:free", "heavy/b:free")}
    client, _ = await client_for(provider=ScriptedProvider(every))
    response = await client.post("/api/v1/ask", json={"question": QUESTION})
    assert response.status_code == 503 and response.headers["retry-after"] == "10"
    assert "rate limited" not in response.text  # the provider's own words stay in the log


async def test_without_a_provider_key_cache_hits_still_work() -> None:
    from app.components import build_components
    from app.embeddings import HashEmbedder
    from app.main import create_app

    settings = make_settings(openrouter_api_key=None)
    components = await build_components(settings, embedder=HashEmbedder(), sinks=[])
    assert components.provider_ready is False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(settings, components)), base_url="http://t"
    ) as client:
        response = await client.post("/api/v1/ask", json={"question": QUESTION})
        assert response.status_code == 503 and response.json()["error"]["type"] == "not_configured"
        assert (await client.get("/health")).json()["status"] == "degraded"


async def test_demo_rate_limit_per_visitor() -> None:
    client, _ = await client_for(demo_mode=True, requests_per_minute_per_client=2)
    for _ in range(2):
        assert (await client.post("/api/v1/route", json={"text": "hi"})).status_code == 200
    limited = await client.post("/api/v1/route", json={"text": "hi"})
    assert limited.status_code == 429 and limited.json()["error"]["type"] == "rate_limited"
    assert int(limited.headers["retry-after"]) >= 1


async def test_router_and_similarity_tools(client_and_components: Any) -> None:
    client, _ = client_and_components
    route = (
        await client.post(
            "/api/v1/route", json={"text": "Compare domestic and international travel approvals"}
        )
    ).json()
    assert route["tier"] == "heavy" and any(s["name"] == "comparison" for s in route["signals"])
    same = (
        await client.post("/api/v1/similarity", json={"a": "Same words here?", "b": "same words here"})
    ).json()
    assert same["similarity"] == 1.0 and same["would_hit"] is True and same["threshold"] == 0.96


async def test_metrics_traces_corpus_and_playground_config(client_and_components: Any) -> None:
    client, _ = client_and_components
    answered = (await client.post("/api/v1/ask", json={"question": QUESTION})).json()
    metrics = (await client.get("/api/v1/metrics")).json()
    assert metrics["requests"] == 1 and metrics["cache_entries"] == 1
    traces = (await client.get("/api/v1/traces?limit=5")).json()["traces"]
    assert traces[0]["trace_id"] == answered["trace_id"]
    names = [s["name"] for s in traces[0]["spans"]]
    assert names == [
        "guard.defang",
        "cache.lookup",
        "retrieval.hybrid",
        "router.complexity",
        "llm.attempt 1",
        "cache.write",
    ]
    detail = await client.get(f"/api/v1/traces/{answered['trace_id']}")
    assert detail.status_code == 200
    assert (await client.get("/api/v1/traces/nope")).status_code == 404
    corpus = (await client.get("/api/v1/corpus")).json()
    assert len(corpus["documents"]) == 7
    config = (await client.get("/api/v1/playground/config")).json()
    assert config["threshold"] == 0.96 and config["samples"] and config["system_prompt"].startswith("You are")


async def test_the_playground_is_served(client_and_components: Any) -> None:
    client, _ = client_and_components
    page = await client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert page.headers["cache-control"] == "no-cache"
