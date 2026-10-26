"""One test of free models: send the cases, grade every answer, say which roles each can fill.

A test sends each model the four cases of :mod:`app.models.cases`, one model at a time, through
the gateway's own provider adapter (no cascade: a result here is what the gateway would get from
that model alone), and grades what comes back. Request starts are kept
``seconds_between_starts`` apart, so a test never crowds out a visitor.

A call that OpenRouter turned away (a rate limit, a timeout, an upstream error) leaves the model
*not judged*: that says something about the moment, nothing about the model. A 404, a 400 or a
403 means the model cannot be called from this account today: the model *failed*, and its test
stops there so no call is spent on it (OpenRouter answers 403 for a model it keeps for
"agentic harnesses", or one whose provider's terms rule this account out). A wrong or unusable answer also fails the case.

Usable as a command, to measure every free model OpenRouter lists (about four free requests per
model) and save every answer::

    python -m app.models.testing --all --out benchmarks/2026-10-26/models.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import cases
from app.models.cases import Role
from app.providers.base import ChatRequest, Provider, ProviderError

STAMP = "%Y-%m-%dT%H:%M:%SZ"
#: A light model must answer quickly: its usual answer may take at most this long.
LIGHT_MAX_MEDIAN_S = 12.0
#: Failure kinds after which the model is not worth another call today.
PERMANENT = frozenset({"not_found", "bad_request", "forbidden", "auth"})
#: Failure kinds that judge the moment, not the model.
TRANSIENT = frozenset({"rate_limited", "timeout", "upstream"})


@dataclass
class CaseResult:
    case_id: str
    #: "right", "wrong", "no_answer" (transient), "unreachable" (404/400/403, or 401: the key)
    outcome: str
    seconds: float | None = None
    why: str = ""
    failure: str | None = None
    answer: str = ""
    #: The provider's own words on a failed call: for logs and the Model check summary only,
    #: never in :meth:`as_dict` (served by ``/api/v1/models``, published in ``chain.json``).
    provider_said: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("provider_said")
        return data


@dataclass
class ModelReport:
    model: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def unreachable(self) -> bool:
        return any(r.outcome == "unreachable" for r in self.results)

    @property
    def judged(self) -> bool:
        """Every case got an answer (right or wrong), or the model cannot be called at all."""
        return self.unreachable or all(r.outcome in {"right", "wrong"} for r in self.results)

    def right(self, case_id: str) -> bool:
        return any(r.case_id == case_id and r.outcome == "right" for r in self.results)

    @property
    def median_s(self) -> float | None:
        times = [r.seconds for r in self.results if r.seconds is not None and r.outcome in {"right", "wrong"}]
        return round(statistics.median(times), 2) if times else None

    def roles(self) -> list[Role]:
        """The roles this model passed every needed case for."""
        qualified: list[Role] = []
        for role, needs in cases.NEEDS.items():
            if all(self.right(case_id) for case_id in needs):
                if role == "light" and (self.median_s is None or self.median_s > LIGHT_MAX_MEDIAN_S):
                    continue
                qualified.append(role)
        return qualified

    @property
    def verdict(self) -> str:
        if self.unreachable:
            return "unreachable"
        if self.roles():
            return "passed"
        if not self.judged:
            return "not_judged"
        return "failed"

    def as_dict(self, *, answers: bool = False) -> dict[str, Any]:
        rows = []
        for r in self.results:
            row = r.as_dict()
            if not answers:
                row.pop("answer")
            rows.append(row)
        return {
            "model": self.model,
            "verdict": self.verdict,
            "roles": self.roles(),
            "median_s": self.median_s,
            "results": rows,
        }


BeforeCall = Callable[[str], Awaitable[None]]


def _ceilings() -> dict[str, int]:
    """The tiers' real output ceilings (the defaults, so every test is the same test)."""
    from app.config import Settings

    settings = Settings(_env_file=None)
    return {"light": settings.max_output_tokens_light, "heavy": settings.max_output_tokens_heavy}


async def try_model(
    provider: Provider,
    model: str,
    *,
    pace: Callable[[], Awaitable[None]],
    before_call: BeforeCall | None = None,
    case_ids: Sequence[str] | None = None,
    ceilings: dict[str, int] | None = None,
) -> ModelReport:
    report = ModelReport(model)
    ceilings = ceilings or _ceilings()
    for case in cases.CASES:
        if case_ids is not None and case.case_id not in case_ids:
            continue
        await pace()
        if before_call is not None:
            await before_call(model)
        request = ChatRequest(
            messages=case.messages(),
            max_tokens=2_048 if case.case_id == "judge" else ceilings[case.tier],
            tier=case.tier,
            temperature=0.0,
            tag="model-test",
            json_mode=case.json_mode,
        )
        started = time.perf_counter()
        try:
            completion = await provider.complete(model, request)
        except ProviderError as exc:
            seconds = round(time.perf_counter() - started, 2)
            if exc.kind in PERMANENT:
                report.results.append(
                    CaseResult(
                        case.case_id,
                        "unreachable",
                        seconds,
                        exc.detail,
                        exc.kind,
                        provider_said=exc.provider_message or "",
                    )
                )
                break  # this model cannot be called from this account: stop spending on it
            outcome = "no_answer" if exc.kind in TRANSIENT else "wrong"
            report.results.append(CaseResult(case.case_id, outcome, seconds, exc.detail, exc.kind))
            continue
        seconds = round(time.perf_counter() - started, 2)
        right, why = case.grade(completion.text, completion.truncated)
        report.results.append(
            CaseResult(
                case.case_id, "right" if right else "wrong", seconds, why, None, completion.text[:1_200]
            )
        )
    return report


class Pacer:
    """At least ``seconds`` between two request starts."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def __call__(self) -> None:
        async with self._lock:
            wait = self._last + self.seconds - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


async def run_test(
    provider: Provider,
    models: Sequence[str],
    *,
    seconds_between_starts: float = 6.0,
    before_call: BeforeCall | None = None,
) -> list[ModelReport]:
    pace = Pacer(seconds_between_starts)
    reports = []
    for model in models:
        if not model.endswith(":free"):
            raise ValueError(f"refusing to test a model that is not free: {model}")
        reports.append(await try_model(provider, model, pace=pace, before_call=before_call))
    return reports


def now_stamp() -> str:
    return datetime.now(UTC).strftime(STAMP)


async def _main(args: argparse.Namespace) -> int:
    from app.config import Settings
    from app.models.catalogue import free_models
    from app.providers import build_provider

    settings = Settings()
    if settings.openrouter_api_key is None:
        print("OPENROUTER_API_KEY is not set (.env.local).", file=sys.stderr)
        return 2
    provider = build_provider(Settings(llm_provider="openrouter", light_models="", heavy_models=""))
    models = (
        await asyncio.to_thread(free_models, settings.openrouter_base_url)
        if args.all
        else args.models.split(",")
    )
    started = now_stamp()
    reports = await run_test(provider, models, seconds_between_starts=args.pace_s)
    await provider.aclose()
    record = {
        "measured_at": started,
        "cases": [c.case_id for c in cases.CASES],
        "light_max_median_s": LIGHT_MAX_MEDIAN_S,
        "models": [r.as_dict(answers=True) for r in reports],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for r in reports:
        marks = " ".join(f"{x.case_id}:{x.outcome}" for x in r.results)
        print(
            f"{r.model:52s} {r.verdict:12s} roles={','.join(r.roles()) or '-':18s} median={r.median_s}s  {marks}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test free models with this gateway's own cases.")
    parser.add_argument("--all", action="store_true", help="every free model OpenRouter lists")
    parser.add_argument("--models", default="", help="comma-separated ids instead")
    parser.add_argument("--pace-s", type=float, default=4.0)
    parser.add_argument("--out", type=Path)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
