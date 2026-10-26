"""Token prices and the cost of a call, in exact decimal dollars.

Two costs are attributed to every model call:

* **actual**: what the call cost. The provider's own figure when it reports one (OpenRouter
  returns ``usage.cost``), else tokens times the price below. A ``:free`` model costs 0.
* **reference**: what the same tokens would cost at the list price of a paid reference model
  for the tier (Claude Haiku 5.5 for light, Claude Opus 5.5 for heavy, by default). On the free
  stack this is the number that makes the economics visible: what the cache and the router
  saved is measured against it.

Prices are USD per million tokens, Anthropic's first-party list prices as published on
6 October 2026. Prices change: check them before quoting a figure, and add or override models
with ``MODEL_PRICES``. Claude Haiku 5.5's price is for prompts up to 100K tokens, which every
prompt of this gateway is (``MAX_INPUT_TOKENS``).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.config import is_free_model

MILLION = Decimal(1_000_000)
#: Costs are rounded to a millionth of a cent for display; sums are kept exact.
QUANTUM = Decimal("0.0000000001")


@dataclass(frozen=True)
class Price:
    input: Decimal
    output: Decimal
    #: Tokens read from the provider's prompt cache.
    cache_read: Decimal
    #: Tokens written to it: 1.25x the input price for Anthropic's five-minute cache.
    cache_write: Decimal

    @classmethod
    def of(cls, input_: str, output: str, cache_read: str | None = None) -> Price:
        read = Decimal(cache_read) if cache_read is not None else Decimal(input_) / 10
        return cls(Decimal(input_), Decimal(output), read, Decimal(input_) * Decimal("1.25"))


#: USD per million tokens. Cache reads: 0.05x the input price on Opus 5.5, 0.1x elsewhere.
BUILTIN_PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price.of("4.00", "20.00", "0.20"),
    "claude-sonnet-5-5": Price.of("2.00", "10.00", "0.20"),
    "claude-haiku-5-5": Price.of("0.10", "0.50", "0.01"),
    "claude-opus-5": Price.of("5.00", "25.00", "0.50"),
    "claude-sonnet-5": Price.of("2.00", "10.00", "0.20"),
    "offline/extractive": Price.of("0", "0", "0"),
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    #: Input tokens served from the provider's prompt cache. Always part of input_tokens: the
    #: providers normalize Anthropic's separate counters to this shape.
    cached_input_tokens: int = 0
    #: Input tokens written to the provider's prompt cache (Anthropic), also part of input_tokens.
    cache_write_tokens: int = 0
    #: True when the provider reported nothing and the numbers are estimates.
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PriceBook:
    def __init__(self, extra: dict[str, list[str]] | None = None) -> None:
        self.prices = dict(BUILTIN_PRICES)
        for model, values in (extra or {}).items():
            self.prices[model] = Price.of(*values)

    def price(self, model: str) -> Price | None:
        if is_free_model(model):
            return Price.of("0", "0", "0")
        if model in self.prices:
            return self.prices[model]
        # "anthropic/claude-haiku-5-5" (OpenRouter's spelling) prices like "claude-haiku-5-5".
        return self.prices.get(model.rsplit("/", 1)[-1])

    def cost(self, model: str, usage: Usage) -> Decimal | None:
        """Tokens times list price; None when the model has no known price."""
        price = self.price(model)
        if price is None:
            return None
        uncached = max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens)
        total = (
            Decimal(uncached) * price.input
            + Decimal(usage.cached_input_tokens) * price.cache_read
            + Decimal(usage.cache_write_tokens) * price.cache_write
            + Decimal(usage.output_tokens) * price.output
        ) / MILLION
        return total

    def reference_cost(self, reference_model: str, usage: Usage) -> Decimal:
        """The same tokens at the reference model's price, with no prompt-cache discount."""
        price = self.price(reference_model)
        if price is None:
            return Decimal(0)
        return (
            Decimal(usage.input_tokens) * price.input + Decimal(usage.output_tokens) * price.output
        ) / MILLION


def usd(value: Decimal | None) -> str | None:
    """A decimal string with ten places, or None. Strings keep JSON from rounding it."""
    if value is None:
        return None
    return format(value.quantize(QUANTUM, rounding=ROUND_HALF_UP), "f")
