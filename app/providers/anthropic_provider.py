"""Claude through the official Anthropic SDK (``LLM_PROVIDER=anthropic``).

* **Prompt caching.** The system prompt is the gateway's static prefix: the same bytes on every
  request, no timestamps, no ids. It is sent as one text block marked
  ``cache_control: {"type": "ephemeral"}``, so once it is long enough to cache (512 tokens on
  the current models) every later request reads it at a tenth of the input price or less. The
  usage counters say whether it happened: ``cache_read_input_tokens`` above zero.
* **Effort, not temperature.** The current models reject sampling parameters; how much they
  think is set with ``output_config.effort``, low for the light tier and medium for the heavy
  one by default. Thinking happens inside ``max_tokens``, so the ceilings still bound the bill.
* **Refusals.** A declined request comes back as HTTP 200 with ``stop_reason: "refusal"``. On
  Claude Opus 5.5 and Claude Sonnet 5.5 the request opts into the server-side fallback
  (``fallbacks: "default"``), which re-runs a declined request on a fallback model inside the
  same call. Whatever still ends in a refusal is a ``refused`` failure, and the gateway's own
  cascade moves on.

The SDK's retries are off: the cascade decides what happens after a failure.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from app.pricing import Usage
from app.providers.base import ChatRequest, Completion, ProviderError, StreamEvent

#: Models that take the server-side refusal fallback. Claude Haiku 5.5 has none.
SERVER_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _split(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """System messages become the system prompt; the rest stay turns, in order."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    return system, turns


def _usage(raw: Any) -> Usage:
    read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
    write = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    uncached = int(getattr(raw, "input_tokens", 0) or 0)
    return Usage(
        input_tokens=uncached + read + write,
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        cached_input_tokens=read,
        cache_write_tokens=write,
    )


def _text(message: Any) -> str:
    return "".join(
        getattr(block, "text", "")
        for block in getattr(message, "content", []) or []
        if getattr(block, "type", None) == "text"
    ).strip()


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        timeout_s: float,
        effort_light: str = "low",
        effort_heavy: str = "medium",
        client: Any | None = None,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._effort = {"light": effort_light, "heavy": effort_heavy}

    def _kwargs(self, model: str, request: ChatRequest) -> dict[str, Any]:
        system, turns = _split(request.messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": turns,
            "output_config": {"effort": self._effort[request.tier]},
        }
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if model in SERVER_FALLBACK_MODELS:
            kwargs["betas"] = [SERVER_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _messages(self, model: str) -> Any:
        return self._client.beta.messages if model in SERVER_FALLBACK_MODELS else self._client.messages

    def _error(self, model: str, exc: Exception) -> ProviderError:
        if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
            return ProviderError(
                model, "auth", f"the key was refused ({exc.status_code})", status=exc.status_code
            )
        if isinstance(exc, anthropic.RateLimitError):
            return ProviderError(model, "rate_limited", "rate limited (429)", status=429)
        if isinstance(exc, anthropic.NotFoundError):
            return ProviderError(model, "not_found", "model not found (404)", status=404)
        if isinstance(exc, anthropic.BadRequestError):
            return ProviderError(model, "bad_request", "request rejected (400)", status=400)
        if isinstance(exc, anthropic.APITimeoutError):
            return ProviderError(model, "timeout", "timed out")
        if isinstance(exc, anthropic.APIStatusError):
            kind = "upstream" if exc.status_code >= 500 or exc.status_code == 408 else "bad_request"
            return ProviderError(model, kind, f"HTTP {exc.status_code}", status=exc.status_code)
        if isinstance(exc, anthropic.APIConnectionError):
            return ProviderError(model, "upstream", "connection error")
        return ProviderError(model, "upstream", type(exc).__name__)

    def _finish(self, model: str, message: Any, started: float, ttft: float | None) -> Completion:
        usage = _usage(getattr(message, "usage", None))
        stop = getattr(message, "stop_reason", None) or "end_turn"
        text = _text(message)
        if stop == "refusal":
            raise ProviderError(model, "refused", "the model declined (stop_reason refusal)", usage=usage)
        if not text:
            detail = "cut off before any text" if stop == "max_tokens" else "empty answer"
            raise ProviderError(model, "invalid_output", detail, usage=usage)
        return Completion(
            model=getattr(message, "model", None) or model,
            text=text,
            finish_reason="length" if stop == "max_tokens" else "stop",
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=ttft,
            extra={"stop_reason": stop},
        )

    async def complete(self, model: str, request: ChatRequest) -> Completion:
        started = time.perf_counter()
        try:
            message = await self._messages(model).create(**self._kwargs(model, request))
        except Exception as exc:
            raise self._error(model, exc) from exc
        return self._finish(model, message, started, None)

    async def stream(self, model: str, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        started = time.perf_counter()
        ttft: float | None = None
        try:
            async with self._messages(model).stream(**self._kwargs(model, request)) as stream:
                async for text in stream.text_stream:
                    if not text:
                        continue
                    if ttft is None:
                        ttft = (time.perf_counter() - started) * 1000
                    yield StreamEvent(kind="delta", text=text)
                message = await stream.get_final_message()
        except ProviderError:
            raise
        except Exception as exc:
            raise self._error(model, exc) from exc
        yield StreamEvent(kind="done", completion=self._finish(model, message, started, ttft))

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:  # pragma: no cover
            pass
