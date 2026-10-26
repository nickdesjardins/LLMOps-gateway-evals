"""The public demo's budget."""

from __future__ import annotations

import asyncio

import pytest

from app.budget import DemoBudget, client_key
from app.errors import BudgetExhausted, Busy, RateLimited
from tests.conftest import make_settings


class Clock:
    def __init__(self, now: float = 1_760_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def budget(clock: Clock, **settings: object) -> DemoBudget:
    return DemoBudget(make_settings(demo_mode=True, **settings), clock=clock)


def test_nothing_is_limited_outside_demo_mode() -> None:
    b = DemoBudget(make_settings(), clock=Clock())
    for _ in range(1_000):
        b.admit_request("x")
    assert b.snapshot() == {"enabled": False}


def test_requests_per_minute_per_visitor() -> None:
    clock = Clock()
    b = budget(clock, requests_per_minute_per_client=3)
    for _ in range(3):
        b.admit_request("a")
    with pytest.raises(RateLimited) as caught:
        b.admit_request("a")
    assert 1 <= caught.value.retry_after_s <= 61
    b.admit_request("b")  # another visitor is unaffected
    clock.now += 61
    b.admit_request("a")


async def test_model_calls_per_visitor_per_day_and_the_utc_day_rollover() -> None:
    clock = Clock()
    b = budget(clock, model_calls_per_day_per_client=2)
    await b.charge_model_call("a")
    await b.charge_model_call("a")
    with pytest.raises(BudgetExhausted, match="cached answers still work"):
        await b.charge_model_call("a")
    with pytest.raises(BudgetExhausted):
        b.check_model_call("a")  # the pre-check refuses without charging
    clock.now += 86_400
    await b.charge_model_call("a")


async def test_global_limits() -> None:
    clock = Clock()
    b = budget(clock, model_calls_per_minute=2, model_calls_per_day=3)
    await b.charge_model_call("a")
    await b.charge_model_call("b")
    with pytest.raises(RateLimited, match="busy"):
        await b.charge_model_call("c")
    clock.now += 61
    await b.charge_model_call("c")
    clock.now += 61
    with pytest.raises(BudgetExhausted, match="midnight UTC"):
        await b.charge_model_call("d")
    assert b.snapshot()["model_calls_today"] == 3


async def test_inflight_slots() -> None:
    b = budget(Clock(), max_inflight=1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with b.model_slot():
            entered.set()
            await release.wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    with pytest.raises(Busy):
        async with b.model_slot():
            pass
    with pytest.raises(Busy):
        b.check_slot()
    release.set()
    await task
    async with b.model_slot():
        pass


def test_client_keys_are_salted_hashes_and_ipv6_is_grouped_by_64() -> None:
    a = client_key("2001:db8:1:2:aaaa::1")
    b = client_key("2001:db8:1:2:bbbb::9")
    c = client_key("2001:db8:1:3::1")
    assert a == b != c
    assert client_key("203.0.113.7") != client_key("203.0.113.8")
    assert "203" not in client_key("203.0.113.7") and len(client_key(None)) == 16
