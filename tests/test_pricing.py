"""Token prices and cost arithmetic, in exact decimals."""

from __future__ import annotations

from decimal import Decimal

from app.pricing import BUILTIN_PRICES, PriceBook, Usage, usd


def test_claude_list_prices() -> None:
    assert BUILTIN_PRICES["claude-opus-5-5"].input == Decimal("4.00")
    assert BUILTIN_PRICES["claude-opus-5-5"].output == Decimal("20.00")
    assert BUILTIN_PRICES["claude-opus-5-5"].cache_read == Decimal("0.20")
    assert BUILTIN_PRICES["claude-haiku-5-5"].input == Decimal("0.10")
    assert BUILTIN_PRICES["claude-haiku-5-5"].cache_write == Decimal("0.125")


def test_cost_is_tokens_times_price_per_million() -> None:
    book = PriceBook()
    usage = Usage(input_tokens=1_000, output_tokens=200)
    assert book.cost("claude-opus-5-5", usage) == Decimal("0.008")  # 1000*4/1e6 + 200*20/1e6
    assert book.cost("claude-haiku-5-5", usage) == Decimal("0.0002")


def test_prompt_cache_reads_and_writes_are_priced_separately() -> None:
    book = PriceBook()
    usage = Usage(input_tokens=1_000, output_tokens=0, cached_input_tokens=600, cache_write_tokens=100)
    # 300 uncached at 4.00, 600 read at 0.20, 100 written at 5.00 (1.25x), per million
    assert book.cost("claude-opus-5-5", usage) == Decimal("0.00182")


def test_free_models_cost_nothing_and_unknown_models_are_unknown() -> None:
    book = PriceBook()
    assert book.cost("vendor/model:free", Usage(10_000, 10_000)) == 0
    assert book.cost("mystery-model", Usage(1, 1)) is None


def test_reference_cost_ignores_prompt_cache_discounts() -> None:
    book = PriceBook()
    usage = Usage(input_tokens=1_000, output_tokens=200, cached_input_tokens=1_000)
    assert book.reference_cost("claude-opus-5-5", usage) == Decimal("0.008")
    assert book.reference_cost("unknown", usage) == 0


def test_extra_prices_and_vendor_prefixed_ids() -> None:
    book = PriceBook({"gpt-x": ["0.15", "0.60"]})
    assert book.cost("gpt-x", Usage(1_000_000, 1_000_000)) == Decimal("0.75")
    assert book.cost("anthropic/claude-haiku-5-5", Usage(1_000_000, 0)) == Decimal("0.10")


def test_usd_formatting_is_fixed_point() -> None:
    assert usd(Decimal(0)) == "0.0000000000"
    assert usd(Decimal("0.00000000004")) == "0.0000000000"
    assert usd(Decimal("0.00000000005")) == "0.0000000001"
    assert usd(None) is None
