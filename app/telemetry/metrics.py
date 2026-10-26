"""Aggregates over finished traces: hit rate, latency percentiles, tokens, cost, fallbacks.

Percentiles are exact over a sliding window (the last ``WINDOW`` requests of each series), by
linear interpolation between the two nearest ranks, the method numpy calls ``linear``. A p99
over fewer than a hundred requests is mostly the slowest request, and the snapshot says how
many requests each figure is computed from so a reader can judge that.

Three cost totals, all in exact decimal dollars:

* ``actual``: what the model calls cost (0 on the free tier);
* ``reference``: the same tokens at the list price of the tier's reference model;
* ``avoided``: for every cache hit, the reference cost of the miss that produced the entry.

Without the cache the bill would have been ``reference + avoided``; with it, ``reference``.
``cost_reduction = avoided / (reference + avoided)`` is the share the cache removed.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from decimal import Decimal
from typing import Any

from app.pricing import usd

WINDOW = 5_000


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks; None for an empty series."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


class _Series:
    def __init__(self) -> None:
        self.values: deque[float] = deque(maxlen=WINDOW)

    def add(self, value: float | None) -> None:
        if value is not None:
            self.values.append(float(value))

    def summary(self) -> dict[str, Any]:
        data = list(self.values)
        return {
            "count": len(data),
            "p50": _round(percentile(data, 50)),
            "p95": _round(percentile(data, 95)),
            "p99": _round(percentile(data, 99)),
            "max": _round(max(data)) if data else None,
        }


class MetricsRecorder:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.requests = 0
        self.outcomes: Counter[str] = Counter()
        self.tiers: Counter[str] = Counter()
        self.endpoints: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.attempts: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.fallback_requests = 0
        self.injections_flagged = 0
        self.refusals = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_actual = Decimal(0)
        self.cost_unknown_requests = 0
        self.cost_reference = Decimal(0)
        self.cost_avoided = Decimal(0)
        self.latency = {"all": _Series(), "hit": _Series(), "miss": _Series()}
        self.ttft = _Series()
        self.lookup = _Series()

    def record(self, trace: Any) -> None:
        attributes = trace.attributes
        outcome = attributes.get("cache", "none")
        if trace.status == "error":
            outcome = "error"
            self.errors[str(attributes.get("error_type", "internal_error"))] += 1
        self.requests += 1
        self.outcomes[outcome] += 1
        self.endpoints[trace.name] += 1
        if attributes.get("tier"):
            self.tiers[str(attributes["tier"])] += 1
        if attributes.get("injection_detected"):
            self.injections_flagged += 1
        if attributes.get("refused"):
            self.refusals += 1
        generations = trace.generations
        failed = [g for g in generations if g.status == "error"]
        for g in generations:
            self.attempts[g.model or "unknown"] += 1
        for g in failed:
            self.failures[f"{g.model}: {g.attributes.get('failure', 'error')}"] += 1
        if failed and any(g.status == "ok" for g in generations):
            self.fallback_requests += 1
        usage = trace.usage()
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        cost = trace.cost()
        if cost is None:
            self.cost_unknown_requests += 1
        else:
            self.cost_actual += cost
        self.cost_reference += trace.reference_cost()
        self.cost_avoided += trace.cost_avoided_usd
        if trace.status == "ok":
            self.latency["all"].add(trace.latency_ms)
            if outcome in self.latency:
                self.latency[outcome].add(trace.latency_ms)
            if outcome == "miss":
                self.ttft.add(trace.ttft_ms)
        self.lookup.add(attributes.get("cache_lookup_ms"))

    def snapshot(self) -> dict[str, Any]:
        hits, misses = self.outcomes.get("hit", 0), self.outcomes.get("miss", 0)
        looked_up = hits + misses
        # Only a miss calls a model, so only a miss can fall back.
        model_requests = misses
        total_reference = self.cost_reference + self.cost_avoided
        return {
            "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started_at)),
            "requests": self.requests,
            "outcomes": dict(self.outcomes),
            "cache_hit_rate": None if not looked_up else round(hits / looked_up, 4),
            "tiers": dict(self.tiers),
            "endpoints": dict(self.endpoints),
            "errors": dict(self.errors),
            "fallback_rate": None
            if not model_requests
            else round(self.fallback_requests / model_requests, 4),
            "fallback_requests": self.fallback_requests,
            "attempts_by_model": dict(self.attempts),
            "failures": dict(self.failures),
            "injections_flagged": self.injections_flagged,
            "refusals": self.refusals,
            "tokens": {"input": self.input_tokens, "output": self.output_tokens},
            "cost": {
                "actual_usd": usd(self.cost_actual),
                "reference_usd": usd(self.cost_reference),
                "avoided_usd": usd(self.cost_avoided),
                "without_cache_usd": usd(total_reference),
                "cost_reduction": None
                if total_reference == 0
                else round(float(self.cost_avoided / total_reference), 4),
                "requests_with_unknown_cost": self.cost_unknown_requests,
            },
            "latency_ms": {name: series.summary() for name, series in self.latency.items()},
            "ttft_ms_miss": self.ttft.summary(),
            "cache_lookup_ms": self.lookup.summary(),
        }
