"""What every model provider looks like to the cascade.

A provider takes a :class:`ChatRequest` and one model id, and either returns a
:class:`Completion` (or a stream of :class:`StreamEvent` ending with one) or raises a
:class:`ProviderError` whose ``kind`` tells the cascade what to do next. The adapters translate
each SDK's exceptions into these kinds, so no other module ever imports an SDK error class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Protocol

from app.config import Tier
from app.pricing import Usage

FailureKind = Literal[
    "rate_limited",  # 429: the next model may have room
    "timeout",  # no answer in MODEL_TIMEOUT_S
    "upstream",  # 5xx, a dropped connection, OpenRouter's in-band error, no choices
    "not_found",  # 404: the model id is gone (free models are withdrawn often)
    "bad_request",  # 400: this model or its endpoint refused the request's shape
    "invalid_output",  # empty, cut off before any text, or failed the caller's validation
    "refused",  # the model declined (a content filter, Anthropic's refusal stop reason)
    "auth",  # 401/403: the key is wrong; every model of this provider shares it
]

#: Kinds after which trying another model cannot help.
FATAL_KINDS: frozenset[str] = frozenset({"auth"})


class ProviderError(Exception):
    """A call that produced no usable answer. The message is safe to log, never shown to clients."""

    def __init__(
        self,
        model: str,
        kind: FailureKind,
        detail: str,
        *,
        status: int | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(f"{model}: {detail}")
        self.model = model
        self.kind = kind
        self.detail = detail
        self.status = status
        #: Tokens the failed call still consumed (a truncated answer is billed).
        self.usage = usage or Usage()

    @property
    def retryable(self) -> bool:
        return self.kind not in FATAL_KINDS


@dataclass(frozen=True)
class ChatRequest:
    """A provider-neutral request. ``messages`` use the roles system, user and assistant."""

    messages: list[dict[str, str]]
    max_tokens: int
    tier: Tier
    #: Low for grounded answers; providers whose current models reject it leave it out.
    temperature: float | None = 0.1
    #: Names the request in the provider's own logs (OpenRouter's X-Title), never content.
    tag: str = "llmops-gateway"
    #: Ask for a JSON object (the eval judge). Providers that cannot are asked in the prompt only.
    json_mode: bool = False


@dataclass(frozen=True)
class Completion:
    model: str
    text: str
    #: "stop", "length", or the provider's own word.
    finish_reason: str
    usage: Usage
    #: The provider's own figure for this call (OpenRouter's usage.cost), when it reports one.
    provider_cost_usd: Decimal | None = None
    latency_ms: float = 0.0
    #: Time to the first token of text, for streamed calls.
    ttft_ms: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason in {"length", "max_tokens"}


@dataclass(frozen=True)
class StreamEvent:
    """A piece of text as it arrives, or the final completion (``kind == "done"``)."""

    kind: Literal["delta", "done"]
    text: str = ""
    completion: Completion | None = None


class Provider(Protocol):
    #: "openrouter", "openai", "gemini", "anthropic" or "offline".
    name: str

    async def complete(self, model: str, request: ChatRequest) -> Completion: ...

    def stream(self, model: str, request: ChatRequest) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...
