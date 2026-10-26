"""OpenRouter, OpenAI and Gemini through one Chat Completions client (the official openai SDK).

The three differ in small ways that this adapter absorbs:

* **OpenRouter** takes vendor extensions in the body: the unified ``reasoning`` switch (a model
  that thinks before answering spends the output ceiling on thinking) and ``usage.include``,
  which makes the response carry the call's real cost. It also reports upstream failures
  *in-band*, as HTTP 200 with an ``error`` member and no choices, which is the most common way a
  free model fails.
* **OpenAI's** current models take ``max_completion_tokens`` and only their default temperature.
* **Gemini** speaks the same protocol at its OpenAI-compatible endpoint.

The SDK's own retries are off (``max_retries=0``): on a 429 the right move is the next model of
the cascade, not the same model again after a back-off.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from decimal import Decimal, InvalidOperation
from typing import Any

import openai

from app.guard import estimate_messages_tokens, estimate_tokens
from app.pricing import Usage
from app.providers.base import ChatRequest, Completion, ProviderError, StreamEvent

logger = logging.getLogger(__name__)


def _reasoning_body(choice: str) -> dict[str, Any] | None:
    if choice == "default":
        return None
    if choice == "off":
        return {"enabled": False}
    return {"effort": choice}


def _usage_from(raw: Any, request: ChatRequest, text: str) -> tuple[Usage, Decimal | None]:
    """Normalize the SDK's usage object; estimate when the provider sent none."""
    if raw is None:
        return (
            Usage(
                input_tokens=estimate_messages_tokens(request.messages),
                output_tokens=estimate_tokens(text),
                estimated=True,
            ),
            None,
        )
    details = getattr(raw, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    usage = Usage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        cached_input_tokens=cached,
    )
    cost: Decimal | None = None
    extra = getattr(raw, "model_extra", None) or {}
    if "cost" in extra and extra["cost"] is not None:
        try:
            cost = Decimal(str(extra["cost"]))
        except InvalidOperation:
            cost = None
    return usage, cost


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str,
        timeout_s: float,
        reasoning: str = "off",
        app_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.name = name
        headers: dict[str, str] = {}
        if name == "openrouter" and app_name:
            headers["X-Title"] = app_name
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
            default_headers=headers or None,
        )
        self._reasoning = reasoning
        #: Models whose endpoint refused the vendor extensions once: called without them since.
        self._plain: set[str] = set()

    # ---------------------------------------------------------------- request shape
    def _kwargs(self, model: str, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": request.messages}
        if self.name == "openai":
            kwargs["max_completion_tokens"] = request.max_tokens
        else:
            kwargs["max_tokens"] = request.max_tokens
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if request.json_mode and model not in self._plain:
            kwargs["response_format"] = {"type": "json_object"}
        if self.name == "openrouter" and model not in self._plain:
            extra: dict[str, Any] = {"usage": {"include": True}}
            reasoning = _reasoning_body(self._reasoning)
            if reasoning is not None:
                extra["reasoning"] = reasoning
            kwargs["extra_body"] = extra
        return kwargs

    def _error(self, model: str, exc: Exception) -> ProviderError:
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            return ProviderError(
                model, "auth", f"the key was refused ({exc.status_code})", status=exc.status_code
            )
        if isinstance(exc, openai.RateLimitError):
            return ProviderError(model, "rate_limited", "rate limited (429)", status=429)
        if isinstance(exc, openai.NotFoundError):
            return ProviderError(model, "not_found", "model not found (404); withdrawn?", status=404)
        if isinstance(exc, openai.BadRequestError):
            return ProviderError(model, "bad_request", "request rejected (400)", status=400)
        if isinstance(exc, openai.APITimeoutError):
            return ProviderError(model, "timeout", "timed out")
        if isinstance(exc, openai.APIStatusError):
            kind = "upstream" if exc.status_code >= 500 or exc.status_code == 408 else "bad_request"
            return ProviderError(model, kind, f"HTTP {exc.status_code}", status=exc.status_code)
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError(model, "upstream", "connection error")
        return ProviderError(model, "upstream", type(exc).__name__)

    async def _create(self, model: str, request: ChatRequest, *, stream: bool) -> Any:
        kwargs = self._kwargs(model, request, stream=stream)
        try:
            return await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            if "extra_body" not in kwargs and "response_format" not in kwargs:
                raise self._error(model, exc) from exc
            # A 400 on a request that carried optional parameters (OpenRouter's extensions, JSON
            # mode): they are the first suspect, and dropping them costs one more request.
            logger.info("%s refused the optional parameters; calling it without them", model)
            self._plain.add(model)
            kwargs = self._kwargs(model, request, stream=stream)
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as retry_exc:
                raise self._error(model, retry_exc) from retry_exc
        except Exception as exc:
            raise self._error(model, exc) from exc

    @staticmethod
    def _in_band_error(model: str, payload: Any) -> None:
        error = getattr(payload, "error", None)
        if error is None:
            error = (getattr(payload, "model_extra", None) or {}).get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            raise ProviderError(model, "upstream", f"upstream provider error (code {code})")

    # ------------------------------------------------------------------- one shot
    async def complete(self, model: str, request: ChatRequest) -> Completion:
        started = time.perf_counter()
        response = await self._create(model, request, stream=False)
        latency = (time.perf_counter() - started) * 1000
        self._in_band_error(model, response)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProviderError(model, "upstream", "no choices in the response")
        choice = choices[0]
        text = (getattr(choice.message, "content", None) or "").strip()
        finish = getattr(choice, "finish_reason", None) or "stop"
        usage, cost = _usage_from(getattr(response, "usage", None), request, text)
        if finish == "content_filter":
            raise ProviderError(model, "refused", "content filter", usage=usage)
        if not text:
            detail = (
                "cut off before any text (thinking used the budget?)"
                if finish == "length"
                else "empty answer"
            )
            raise ProviderError(model, "invalid_output", detail, usage=usage)
        # Priced and reported under the id that was asked for: OpenRouter answers with the
        # upstream id, which drops the ":free" suffix the price book relies on.
        return Completion(
            model=model,
            text=text,
            finish_reason=finish,
            usage=usage,
            provider_cost_usd=cost,
            latency_ms=latency,
            extra={"served_by": getattr(response, "model", None) or model},
        )

    # ------------------------------------------------------------------ streaming
    async def stream(self, model: str, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        started = time.perf_counter()
        response = await self._create(model, request, stream=True)
        pieces: list[str] = []
        ttft: float | None = None
        finish = "stop"
        raw_usage: Any = None
        try:
            async for chunk in response:
                self._in_band_error(model, chunk)
                if getattr(chunk, "usage", None) is not None:
                    raw_usage = chunk.usage
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice.delta, "content", None) if choice.delta else None
                    if delta:
                        if ttft is None:
                            ttft = (time.perf_counter() - started) * 1000
                        pieces.append(delta)
                        yield StreamEvent(kind="delta", text=delta)
                    if getattr(choice, "finish_reason", None):
                        finish = choice.finish_reason
        except ProviderError:
            raise
        except Exception as exc:
            raise self._error(model, exc) from exc
        text = "".join(pieces).strip()
        usage, cost = _usage_from(raw_usage, request, text)
        if finish == "content_filter":
            raise ProviderError(model, "refused", "content filter", usage=usage)
        if not text:
            raise ProviderError(model, "invalid_output", "empty answer", usage=usage)
        yield StreamEvent(
            kind="done",
            completion=Completion(
                model=model,
                text=text,
                finish_reason=finish,
                usage=usage,
                provider_cost_usd=cost,
                latency_ms=(time.perf_counter() - started) * 1000,
                ttft_ms=ttft,
            ),
        )

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception:  # pragma: no cover
            pass
