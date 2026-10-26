"""The complexity router: which tier answers a request that the cache could not.

It is deliberately not a model. A router that called a model to decide would add a round trip
and a bill to every miss, which is what it exists to avoid. It scores the request from features
it can read in microseconds, and every point of the score has a name, so a trace says *why* a
request went where it went:

* **Light signals** lower the score: the request classifies, extracts, converts to a format,
  or asks one short factual question.
* **Heavy signals** raise it: comparing, analysing, explaining why, planning, a conditional
  scenario, several questions at once, a long request.
* **Retrieval spread** (grounded questions only): when the strongest passages come from two or
  three different documents, the answer has to be synthesized across them.

A score at or above ``ROUTER_HEAVY_THRESHOLD`` (0.5) goes to the heavy tier. A client may name
the task (``task: "classify"``), which decides the tier outright; ``model: "light"`` or
``"heavy"`` on the chat route does the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.config import Tier
from app.guard import estimate_tokens

TaskHint = Literal["auto", "classify", "extract", "answer", "synthesize"]

BASE_SCORE = 0.30

_LIGHT: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "classification",
        -0.20,
        re.compile(
            r"^\s*(?:please\s+)?(?:classify|categori[sz]e|label|tag|triage|route)\b|"
            r"\bwhich (?:category|team|queue|priority|department)\b"
        ),
    ),
    (
        "extraction",
        -0.20,
        re.compile(r"^\s*(?:please\s+)?(?:extract|parse|pull out)\b|\b(?:extract|parse)\b.{0,60}\bfrom\b"),
    ),
    ("structured_output", -0.10, re.compile(r"\b(?:json|csv|yaml|key[- ]value pairs?)\b")),
    ("yes_no", -0.10, re.compile(r"\b(?:yes or no|true or false)\b")),
    (
        "short_lookup",
        -0.10,
        re.compile(
            r"^\s*(?:what|how many|how much|how long|when|who|where|which|is|are|can|do|does)\b[^?]{0,80}\??\s*$"
        ),
    ),
)

_HEAVY: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "comparison",
        0.25,
        re.compile(r"\b(?:compare|comparison|contrast|versus|vs\.?|difference between|differ(?:s|ent)?)\b"),
    ),
    (
        "analysis",
        0.20,
        re.compile(r"\b(?:analy[sz]e|evaluate|assess|implications?|trade-?offs?|pros and cons|weigh)\b"),
    ),
    ("explanation", 0.15, re.compile(r"\b(?:why|explain|justify|reasoning)\b")),
    (
        "planning",
        0.20,
        re.compile(
            r"\b(?:plan|step[- ]by[- ]step|walk me through|checklist|draft|write (?:a|an|me)|"
            r"recommend|how (?:do|should) (?:i|we) handle)\b|"
            r"\bwhat (?:should|must|do|does|can) [a-z ]{1,30}?(?:have to |need to )?do\b"
        ),
    ),
    (
        "conditional",
        0.15,
        re.compile(r"\b(?:if|unless|suppose|scenario|what happens|what would happen|in case)\b"),
    ),
    ("aggregation", 0.15, re.compile(r"\b(?:summari[sz]e|overview|all the|every|each of)\b")),
)


@dataclass(frozen=True)
class Signal:
    name: str
    weight: float

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "weight": round(self.weight, 2)}


@dataclass(frozen=True)
class RouteDecision:
    tier: Tier
    score: float
    threshold: float
    signals: tuple[Signal, ...] = field(default_factory=tuple)
    #: "router", "task hint" or "client".
    decided_by: str = "router"

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "score": round(self.score, 3),
            "base": BASE_SCORE,
            "threshold": self.threshold,
            "decided_by": self.decided_by,
            "signals": [s.as_dict() for s in self.signals],
        }


class ComplexityRouter:
    def __init__(self, heavy_threshold: float = 0.5) -> None:
        self.threshold = heavy_threshold

    def score(
        self,
        text: str,
        *,
        documents_in_play: int = 0,
        turns: int = 1,
    ) -> tuple[float, list[Signal]]:
        lowered = text.lower().strip()
        signals: list[Signal] = []
        for name, weight, pattern in (*_LIGHT, *_HEAVY):
            if pattern.search(lowered):
                signals.append(Signal(name, weight))
        questions = lowered.count("?")
        if questions >= 2:
            signals.append(Signal("several_questions", 0.20))
        tokens = estimate_tokens(text)
        if tokens > 120:
            signals.append(Signal("long_request", 0.20))
        elif tokens > 45:
            signals.append(Signal("long_request", 0.10))
        if documents_in_play >= 3:
            signals.append(Signal("spans_three_documents", 0.30))
        elif documents_in_play == 2:
            signals.append(Signal("spans_two_documents", 0.15))
        if turns > 4:
            signals.append(Signal("long_conversation", 0.10))
        score = BASE_SCORE + sum(s.weight for s in signals)
        return max(0.0, min(1.0, score)), signals

    def route(
        self,
        text: str,
        *,
        task: TaskHint = "auto",
        documents_in_play: int = 0,
        turns: int = 1,
        forced_tier: Tier | None = None,
    ) -> RouteDecision:
        score, signals = self.score(text, documents_in_play=documents_in_play, turns=turns)
        if forced_tier is not None:
            return RouteDecision(forced_tier, score, self.threshold, tuple(signals), "client")
        if task in {"classify", "extract"}:
            return RouteDecision("light", score, self.threshold, tuple(signals), "task hint")
        if task == "synthesize":
            return RouteDecision("heavy", score, self.threshold, tuple(signals), "task hint")
        tier: Tier = "heavy" if score >= self.threshold else "light"
        return RouteDecision(tier, score, self.threshold, tuple(signals))
