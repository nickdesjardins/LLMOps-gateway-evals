"""The published chains: what the scheduled test confirmed, in one small file every service reads.

A service that nobody uses tests nothing, and on a free host it forgets what it knew at every
sleep. So a scheduled job (``.github/workflows/model-check.yml``) runs this module every day,
tests the free models when a test is due, and commits the chains it confirmed to ``chain.json``
on the branch ``tested-models``. Every service in automatic mode reads that file at start-up and
after a model failed (:data:`app.config.DEFAULT_PUBLISHED_URL`): one request, no key, no free
request spent.

The file is a claim like any other, so :func:`parse` checks it before it is used: at most
64,000 bytes, version 1, a test date not in the future, one to three different ``:free`` ids per
role, and a fingerprint of everything the test is made of (the cases, the prompt, the rules,
the defaults). A fork that changes a prompt therefore ignores this project's file by itself.

A test is due (:func:`due`) when nothing usable is published, when the last test is six and a
half days old, when the last one confirmed nothing and 20 hours have passed, when OpenRouter's
list of free models changed, or when the run was started with ``--force``.

Run by the workflow (``--previous`` is the file currently published, if any)::

    python -m app.models.published --previous previous/chain.json --out out/chain.json

It exits 0 when all is well (with or without a new file), and 1 with a line beginning
"Failing this run on purpose:" when the owner has something to do.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import (
    DEFAULT_OPENROUTER_HEAVY,
    DEFAULT_OPENROUTER_JUDGE,
    DEFAULT_OPENROUTER_LIGHT,
    parse_model_chain,
)
from app.models import cases, rules
from app.models.cases import ROLES, Role
from app.models.catalogue import free_models, is_free_id
from app.models.testing import STAMP, ModelReport, run_test

VERSION = 1
MAX_BYTES = 64_000
DUE_AFTER = timedelta(hours=156)
RETRY_AFTER = timedelta(hours=20)
STALE_AFTER = timedelta(days=14)
#: Models a scheduled test sends the cases to, at most (4 cases each).
SCHEDULED_MODELS = 24

DEFAULT_CHAINS: dict[Role, list[str]] = {
    "light": parse_model_chain(DEFAULT_OPENROUTER_LIGHT),
    "heavy": parse_model_chain(DEFAULT_OPENROUTER_HEAVY),
    "judge": parse_model_chain(DEFAULT_OPENROUTER_JUDGE),
}


class Unusable(Exception):
    """The published file cannot be used; the message says why, for the log."""


@dataclass(frozen=True)
class Published:
    tested: datetime
    chains: dict[Role, list[str]]
    free_list: list[str]
    confirmed: bool
    record: dict[str, Any]


def fingerprint() -> str:
    """A hash of what a test result depends on. A change to any of it makes old files stale."""
    material = [
        cases.fingerprint_material(),
        [rules.PLACES, sorted((k, list(v)) for k, v in rules.PREFERENCE.items())],
        {role: chain for role, chain in DEFAULT_CHAINS.items()},
        VERSION,
    ]
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _moment(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, STAMP).replace(tzinfo=UTC)
    except ValueError:
        return None


def parse(text: str, *, expected: str | None = None, now: datetime | None = None) -> Published:
    now = now or datetime.now(UTC)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise Unusable("the file is larger than 64,000 bytes")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise Unusable("the file is not JSON") from exc
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise Unusable("the file is not a version 1 chain file")
    tested = _moment(data.get("tested"))
    if tested is None or tested > now + timedelta(minutes=5):
        raise Unusable("the test date is missing or in the future")
    if data.get("fingerprint") != (expected or fingerprint()):
        raise Unusable("it was made by another version of the test")
    chains_raw = data.get("chains")
    if not isinstance(chains_raw, dict):
        raise Unusable("the file has no chains")
    chains: dict[Role, list[str]] = {}
    for role in ROLES:
        chain = chains_raw.get(role)
        if (
            not isinstance(chain, list)
            or not 1 <= len(chain) <= rules.PLACES
            or len(set(chain)) != len(chain)
            or not all(is_free_id(model) for model in chain)
        ):
            raise Unusable(f"the {role} chain is not one to three different free model ids")
        chains[role] = list(chain)
    free_list = [m for m in data.get("free_list", []) if is_free_id(m)]
    return Published(tested, chains, free_list, bool(data.get("confirmed", True)), data)


def fetch(url: str, timeout: float = 8.0) -> str:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - a fixed https host
        return response.read(MAX_BYTES + 1).decode("utf-8")


def due(previous: Published | None, free_list: list[str], now: datetime, force: bool = False) -> str | None:
    """Why a test is due now, or None."""
    if force:
        return "the run was started with --force"
    if previous is None:
        return "nothing usable is published"
    age = now - previous.tested
    if not previous.confirmed and age >= RETRY_AFTER:
        return "the last test confirmed nothing, 20 hours ago or more"
    if age >= DUE_AFTER:
        return "the last test is six and a half days old"
    if sorted(free_list) != sorted(previous.free_list):
        return "OpenRouter's list of free models changed"
    for role, chain in previous.chains.items():
        missing = [m for m in chain if m not in free_list]
        if missing:
            return f"a model of the {role} chain is no longer listed: {', '.join(missing)}"
    return None


def build(
    chains: dict[Role, list[str]],
    reports: list[ModelReport],
    free_list: list[str],
    tested: datetime,
    *,
    confirmed: bool,
    reason: str,
    placings: dict[Role, dict[str, str]],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "tested": tested.strftime(STAMP),
        "fingerprint": fingerprint(),
        "confirmed": confirmed,
        "reason": reason,
        "chains": chains,
        "placings": placings,
        "free_list": free_list,
        "models": [r.as_dict() for r in reports],
    }


def _summary(line: str) -> None:
    print(line)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _fail(why: str) -> int:
    _summary(f"Failing this run on purpose: {why}")
    return 1


async def _main(args: argparse.Namespace) -> int:
    from app.config import Settings
    from app.models.catalogue import allowance
    from app.providers import build_provider

    now = datetime.now(UTC)
    previous: Published | None = None
    if args.previous and args.previous.is_file():
        try:
            previous = parse(args.previous.read_text(encoding="utf-8"), now=now)
        except Unusable as exc:
            print(f"The published file is not usable ({exc}): a test is due.")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    free_list = free_models(base)
    current = previous.chains if previous else DEFAULT_CHAINS
    reason = due(previous, free_list, now, args.force)
    if reason is None:
        _summary(
            f"No test is due. Chains in use (tested {previous.tested:%Y-%m-%d}): "
            + "; ".join(  # type: ignore[union-attr]
                f"{role} {', '.join(chain)}" for role, chain in current.items()
            )
        )
        if previous and now - previous.tested > STALE_AFTER:
            return _fail("the published chains are over 14 days old")
        return 0
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        unlisted = [m for chain in current.values() for m in chain if m not in free_list]
        print(
            f"::warning::A test is due ({reason}) but there is no OPENROUTER_API_KEY secret: nothing tested."
        )
        if unlisted and previous is None:
            return _fail(f"no key, and default models are no longer listed: {', '.join(unlisted)}")
        return 0
    try:
        account = allowance(base, key)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return _fail(
                "OpenRouter refused the key (401 on GET /api/v1/key): replace the OPENROUTER_API_KEY secret"
            )
        return _fail(f"OpenRouter's record of the key could not be read (HTTP {exc.code})")
    if account.get("is_management_key"):
        return _fail("OPENROUTER_API_KEY is a management key, which cannot call a model: make a regular key")
    left = account.get("remaining_today")
    to_test = list(
        dict.fromkeys([m for chain in current.values() for m in chain if m in free_list] + free_list)
    )
    to_test = to_test[:SCHEDULED_MODELS]
    if isinstance(left, int) and left < len(to_test) * len(cases.CASES) + 50:
        return _fail(
            f"only {left} free requests are left today; a test needs about {len(to_test) * len(cases.CASES)}"
        )
    print(f"A test is due: {reason}. Testing {len(to_test)} models.")
    provider = build_provider(Settings(_env_file=None, llm_provider="openrouter", openrouter_api_key=key))
    reports = await run_test(provider, to_test, seconds_between_starts=args.pace_s)
    await provider.aclose()
    refused = rules.key_refused(reports)
    if refused:
        return _fail(refused)
    by_model = {r.model: r for r in reports}
    unlisted = {m for chain in current.values() for m in chain if m not in free_list}
    chains: dict[Role, list[str]] = {}
    placings: dict[Role, dict[str, str]] = {}
    for role in ROLES:
        chains[role], placings[role] = rules.repair(current[role], by_model, role=role, gone=unlisted)
    confirmed = all(any(role in by_model[m].roles() for m in chains[role] if m in by_model) for role in ROLES)
    record = build(chains, reports, free_list, now, confirmed=confirmed, reason=reason, placings=placings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for role in ROLES:
        _summary(
            f"{role}: {', '.join(chains[role])} ({', '.join(f'{m} {p}' for m, p in placings[role].items())})"
        )
    for report in reports:
        line = f"- {report.model}: {report.verdict}, roles {', '.join(report.roles()) or 'none'}, median {report.median_s} s"
        closed = next((r for r in report.results if r.outcome == "unreachable"), None)
        if closed:
            said = f"; OpenRouter said: {closed.provider_said}" if closed.provider_said else ""
            line += f" ({closed.failure}: {closed.why}{said})"
        _summary(line)
    failing = {role: rules.members_failing(current[role], by_model, role) for role in ROLES}
    if not confirmed:
        return _fail("a role has no model that passed its cases; the chains are published unconfirmed")
    if any(failing.values()):
        named = "; ".join(f"{role}: {', '.join(models)}" for role, models in failing.items() if models)
        return _fail(f"models of the chains failed their cases and only a person may reorder them ({named})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the free models when due and write the chains to publish."
    )
    parser.add_argument("--previous", type=Path, help="the chain.json currently published")
    parser.add_argument("--out", type=Path, default=Path("out/chain.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pace-s", type=float, default=rules.SECONDS_BETWEEN_STARTS)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
