"""The public demo's budget: who may spend the free model allowance, and how fast.

Two kinds of limit, because a cache hit and a model call do not cost the same:

* **Requests**, per visitor per minute, for every route that does work (a cache hit costs a few
  milliseconds of CPU, and the similarity tool an embedding). Cheap, so the limit is generous.
* **Model calls**, charged only when the cascade is about to call a model: per visitor per UTC
  day, and for everyone together per minute and per UTC day. A failed attempt that fell back
  to another model is a second call and is charged as one. When the model budget is spent, cache
  hits keep working; only misses are refused.

At most ``MAX_INFLIGHT`` requests call models at once; the next gets 503 with ``Retry-After``.
Outside demo mode none of this applies: the operator's own key, the operator's own limits.

Visitor ids are hashed with a salt drawn at start-up and never logged or stored as addresses.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from app.config import Settings
from app.errors import BudgetExhausted, Busy, RateLimited

_SALT = secrets.token_bytes(16)


def client_key(address: str | None) -> str:
    """A short, salted hash of a client address; IPv6 visitors are grouped by their /64."""
    text = (address or "unknown").strip()
    try:
        ip = ipaddress.ip_address(text)
        if ip.version == 6:
            text = str(ipaddress.ip_network(f"{ip}/64", strict=False).network_address) + "/64"
        else:
            text = str(ip)
    except ValueError:
        text = text[:64]
    return hashlib.blake2b(text.encode("utf-8"), key=_SALT, digest_size=8).hexdigest()


def _utc_day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


class _Window:
    """Events per key in the last ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def count(self, key: str, now: float) -> int:
        events = self.events[key]
        while events and events[0] <= now - self.seconds:
            events.popleft()
        return len(events)

    def add(self, key: str, now: float) -> None:
        self.events[key].append(now)

    def retry_after(self, key: str, now: float) -> int:
        events = self.events[key]
        return max(1, int(events[0] + self.seconds - now) + 1) if events else 1


class DemoBudget:
    def __init__(self, settings: Settings, clock: Callable[[], float] = time.time) -> None:
        self.enabled = settings.demo_mode
        self.settings = settings
        self._clock = clock
        self._requests = _Window(60)
        self._calls_minute = _Window(60)
        self._day = _utc_day(clock())
        self._calls_today: dict[str, int] = defaultdict(int)
        self._calls_today_total = 0
        self._inflight = asyncio.Semaphore(settings.max_inflight)
        self._inflight_now = 0

    def _roll_day(self, now: float) -> None:
        day = _utc_day(now)
        if day != self._day:
            self._day = day
            self._calls_today.clear()
            self._calls_today_total = 0

    def admit_request(self, client: str) -> None:
        """Every request that does work. Raises RateLimited."""
        if not self.enabled:
            return
        now = self._clock()
        if self._requests.count(client, now) >= self.settings.requests_per_minute_per_client:
            raise RateLimited(
                "Too many requests from this address; please wait a minute.",
                retry_after_s=self._requests.retry_after(client, now),
            )
        self._requests.add(client, now)

    def check_model_call(self, client: str) -> None:
        """Would a model call be admitted now? Raises like :meth:`charge_model_call`, charges nothing.

        Run before a stream opens, so a spent budget is an HTTP 429, not an error event."""
        if not self.enabled:
            return
        self._refuse_if_spent(client, self._clock())

    def check_slot(self) -> None:
        if self.enabled and self._inflight.locked():
            raise Busy(
                "The demo is answering other questions; please retry in a few seconds.", retry_after_s=5
            )

    def _refuse_if_spent(self, client: str, now: float) -> None:
        self._roll_day(now)
        if self._calls_today[client] >= self.settings.model_calls_per_day_per_client:
            raise BudgetExhausted(
                "This address has used its model calls for today; cached answers still work.",
                retry_after_s=self._seconds_to_midnight(now),
            )
        if self._calls_today_total >= self.settings.model_calls_per_day:
            raise BudgetExhausted(
                "The demo's model calls for today are used up; cached answers still work. "
                "It resumes at midnight UTC.",
                retry_after_s=self._seconds_to_midnight(now),
            )
        if self._calls_minute.count("all", now) >= self.settings.model_calls_per_minute:
            raise RateLimited(
                "The demo is busy right now; please retry in a few seconds.",
                retry_after_s=self._calls_minute.retry_after("all", now),
            )

    async def charge_model_call(self, client: str) -> None:
        """Called just before each model call. Raises BudgetExhausted or RateLimited."""
        if not self.enabled:
            return
        now = self._clock()
        self._refuse_if_spent(client, now)
        self._calls_minute.add("all", now)
        self._calls_today[client] += 1
        self._calls_today_total += 1

    @asynccontextmanager
    async def model_slot(self) -> Any:
        """At most MAX_INFLIGHT requests calling models at once; the next one is told to retry."""
        if not self.enabled:
            yield
            return
        self.check_slot()
        async with self._inflight:
            self._inflight_now += 1
            try:
                yield
            finally:
                self._inflight_now -= 1

    @staticmethod
    def _seconds_to_midnight(now: float) -> int:
        return int(86_400 - (now % 86_400)) + 1

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        now = self._clock()
        self._roll_day(now)
        return {
            "enabled": True,
            "model_calls_today": self._calls_today_total,
            "model_calls_per_day": self.settings.model_calls_per_day,
            "model_calls_last_minute": self._calls_minute.count("all", now),
            "model_calls_per_minute": self.settings.model_calls_per_minute,
            "per_client_per_day": self.settings.model_calls_per_day_per_client,
            "requests_per_minute_per_client": self.settings.requests_per_minute_per_client,
            "inflight": self._inflight_now,
            "max_inflight": self.settings.max_inflight,
        }
