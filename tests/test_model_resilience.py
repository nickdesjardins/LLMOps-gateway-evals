"""Keep honest model evidence across outages, and refresh it during healthy traffic."""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app import providers
from app.models import catalogue, published, rules
from app.models.chooser import PUBLISHED_REFRESH_S, ModelChooser
from app.models.testing import CaseResult, ModelReport
from app.providers.base import ProviderError
from app.routing.cascade import Attempt
from tests.test_models import Script, answering, automatic, publishable


@pytest.mark.parametrize(
    "pricing",
    [{}, {"prompt": "0"}, {"prompt": "NaN", "completion": "0"}, {"prompt": "0", "completion": None}],
)
def test_unknown_prices_are_not_free(pricing: dict) -> None:
    listing = {"data": [{"id": "a/model:free", "pricing": pricing}]}
    assert catalogue.free_models("https://example.invalid", fetch=lambda url: listing) == []


def test_weekly_candidate_batches_rotate_beyond_the_cap_without_skipping_incumbents() -> None:
    chains = {role: [f"current/{role}:free"] for role in ("light", "heavy", "judge")}
    incumbents = [chain[0] for chain in chains.values()]
    newcomers = [f"candidate/model-{index:02}:free" for index in range(55)]
    listed = incumbents + newcomers
    previous = None
    seen = set()
    for _ in range(3):
        batch, cursor = published.select_models(chains, listed, previous)
        assert len(batch) <= published.SCHEDULED_MODELS and batch[:3] == incumbents
        seen.update(batch)
        previous = published.parse(
            json.dumps(
                publishable(
                    tested=datetime.now(UTC).strftime(published.STAMP),
                    chains=chains,
                    free_list=listed,
                    candidate_cursor=cursor,
                )
            )
        )
    assert seen == set(listed)


def scheduled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, age: int = 7, previous: bool = True):
    models = ["a/model:free"]
    record = publishable(
        tested=(datetime.now(UTC) - timedelta(days=age)).strftime(published.STAMP),
        chains={role: models for role in ("light", "heavy", "judge")},
        free_list=models,
    )
    path = tmp_path / "previous.json"
    if previous:
        path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-only")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setattr(published, "free_models", lambda base: models)
    monkeypatch.setattr(catalogue, "allowance", lambda base, key: {"remaining_today": 1000})
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script())
    args = argparse.Namespace(previous=path, out=tmp_path / "chain.json", force=False, pace_s=0)
    return args, record


@pytest.mark.parametrize(("age", "previous", "expected"), [(7, True, 0), (15, True, 1), (0, False, 1)])
async def test_missing_key_uses_real_evidence_age(monkeypatch, tmp_path, age, previous, expected) -> None:
    args, _ = scheduled(monkeypatch, tmp_path, age=age, previous=previous)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert await published._main(args) == expected
    assert not args.out.exists()


@pytest.mark.parametrize("problem", ["catalogue", "quota", "account"])
@pytest.mark.parametrize(("age", "expected"), [(7, 0), (15, 1)])
async def test_temporary_checks_keep_fresh_chains_but_alert_when_stale(
    monkeypatch, tmp_path, problem, age, expected
) -> None:
    args, _ = scheduled(monkeypatch, tmp_path, age=age)

    def unavailable(*args):
        raise urllib.error.HTTPError("https://example.invalid", 503, "busy", {}, None)

    if problem == "catalogue":
        monkeypatch.setattr(published, "free_models", unavailable)
    elif problem == "account":
        monkeypatch.setattr(catalogue, "allowance", unavailable)
    else:
        monkeypatch.setattr(catalogue, "allowance", lambda base, key: {"remaining_today": 1})
    assert await published._main(args) == expected
    assert not args.out.exists()


async def test_inconclusive_model_test_preserves_success_date_and_retries_after_cooldown(
    monkeypatch, tmp_path
) -> None:
    args, old = scheduled(monkeypatch, tmp_path)
    busy = ProviderError("a/model:free", "rate_limited", "429", status=429)
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script({"a/model:free": busy}))
    assert await published._main(args) == 0
    kept = published.parse(args.out.read_text())
    assert kept.confirmed and kept.record["tested"] == old["tested"]
    assert kept.record["attempt"]["status"] == "inconclusive"
    now = datetime.now(UTC)
    assert published.due(kept, kept.free_list, now) is None
    assert "last attempt" in published.due(kept, kept.free_list, now + timedelta(hours=21))


async def test_real_quality_failure_preserves_last_good_date_but_still_fails(monkeypatch, tmp_path) -> None:
    args, old = scheduled(monkeypatch, tmp_path)
    monkeypatch.setattr(
        providers,
        "build_provider",
        lambda settings: Script({"a/model:free": answering({"synthesis": "wrong"})}),
    )
    assert await published._main(args) == 1
    kept = published.parse(args.out.read_text())
    assert kept.record["tested"] == old["tested"] and kept.record["attempt"]["status"] == "failed"


@pytest.mark.parametrize("problem", ["key", "catalogue", "quota"])
async def test_a_skipped_test_cannot_hide_an_unresolved_quality_failure(
    monkeypatch, tmp_path, problem
) -> None:
    args, record = scheduled(monkeypatch, tmp_path)
    record["attempt"] = {
        "at": (datetime.now(UTC) - timedelta(days=1)).strftime(published.STAMP),
        "status": "failed",
    }
    record["quality_failures"] = {"heavy": ["a/model:free"]}
    args.previous.write_text(json.dumps(record))
    if problem == "key":
        monkeypatch.delenv("OPENROUTER_API_KEY")
    elif problem == "quota":
        monkeypatch.setattr(catalogue, "allowance", lambda base, key: {"remaining_today": 1})
    else:

        def unavailable(base):
            raise TimeoutError("catalogue")

        monkeypatch.setattr(published, "free_models", unavailable)
    assert await published._main(args) == 1


async def test_only_a_successful_retest_clears_a_recorded_role_failure(monkeypatch, tmp_path) -> None:
    args, record = scheduled(monkeypatch, tmp_path)
    record["attempt"] = {
        "at": (datetime.now(UTC) - timedelta(days=1)).strftime(published.STAMP),
        "status": "failed",
    }
    record["quality_failures"] = {"heavy": ["a/model:free"]}
    args.previous.write_text(json.dumps(record))
    busy = ProviderError("a/model:free", "timeout", "temporary timeout")
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script({"a/model:free": busy}))
    assert await published._main(args) == 1
    kept = published.parse(args.out.read_text())
    assert kept.record["attempt"]["status"] == "failed"
    assert kept.record["quality_failures"]["heavy"] == ["a/model:free"]
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script())
    args.previous.write_text(args.out.read_text())
    args.force = True
    assert await published._main(args) == 0
    assert not published.parse(args.out.read_text()).record.get("quality_failures")


async def test_tentative_removal_cannot_clear_a_failure_in_the_retained_publication(
    monkeypatch, tmp_path
) -> None:
    args, record = scheduled(monkeypatch, tmp_path)
    primary, backup = "a/model:free", "b/model:free"
    record["chains"] = {"light": [primary], "heavy": [primary, backup], "judge": [primary]}
    record["free_list"] = [primary, backup]
    record["quality_failures"] = {"heavy": [backup]}
    record["attempt"] = {"at": record["tested"], "status": "failed"}
    args.previous.write_text(json.dumps(record))
    monkeypatch.setattr(published, "free_models", lambda base: [primary, backup])

    def judge_busy(model, request):
        if "verdicts" in request.messages[-1]["content"]:
            raise ProviderError(model, "timeout", "judge temporarily unavailable")
        return answering()(model, request)

    gone = ProviderError(backup, "not_found", "404", status=404)
    monkeypatch.setattr(
        providers, "build_provider", lambda settings: Script({primary: judge_busy, backup: gone})
    )
    assert await published._main(args) == 1
    kept = published.parse(args.out.read_text())
    assert kept.chains == record["chains"] and kept.record["tested"] == record["tested"]
    assert kept.record["quality_failures"]["heavy"] == [backup]

    # A later successful primary cannot erase the bad backup's history during its timeout.
    args.previous.write_text(args.out.read_text())
    args.force = True
    busy = ProviderError(backup, "timeout", "temporary timeout")
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script({backup: busy}))
    assert await published._main(args) == 1
    assert published.parse(args.out.read_text()).record["quality_failures"]["heavy"] == [backup]

    # Once a confirmed repair really removes that backup, its failure no longer applies.
    args.previous.write_text(args.out.read_text())
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script({backup: gone}))
    assert await published._main(args) == 0
    repaired = published.parse(args.out.read_text())
    assert backup not in repaired.chains["heavy"] and not repaired.record.get("quality_failures")


@pytest.mark.parametrize("detailed", [True, False])
async def test_role_retest_clears_its_failure_while_another_role_is_inconclusive(
    monkeypatch, tmp_path, detailed
) -> None:
    args, record = scheduled(monkeypatch, tmp_path)
    record["attempt"] = {"at": record["tested"], "status": "failed"}
    if detailed:
        record["quality_failures"] = {"heavy": ["a/model:free"]}
    args.previous.write_text(json.dumps(record))

    def judge_busy(model, request):
        if "verdicts" in request.messages[-1]["content"]:
            raise ProviderError(model, "timeout", "judge temporarily unavailable")
        return answering()(model, request)

    monkeypatch.setattr(providers, "build_provider", lambda settings: Script({"a/model:free": judge_busy}))
    assert await published._main(args) == (0 if detailed else 1)
    kept = published.parse(args.out.read_text())
    assert kept.chains == record["chains"] and kept.record["tested"] == record["tested"]
    assert not any(kept.record["quality_failures"].values())
    assert kept.record["attempt"]["status"] == ("inconclusive" if detailed else "failed")


async def test_all_listed_models_becoming_unreachable_is_not_a_temporary_success(
    monkeypatch, tmp_path
) -> None:
    args, _ = scheduled(monkeypatch, tmp_path)
    gone = ProviderError("a/model:free", "not_found", "404", status=404)
    monkeypatch.setattr(providers, "build_provider", lambda settings: Script({"a/model:free": gone}))
    assert await published._main(args) == 1


def test_another_roles_rate_limit_cannot_hide_a_known_bad_answer() -> None:
    report = ModelReport(
        "a/model:free", [CaseResult("grounded", "wrong", 1.0), CaseResult("judge", "no_answer", 0.1)]
    )
    assert rules.members_failing([report.model], {report.model: report}, "light") == [report.model]
    assert rules.members_failing([report.model], {report.model: report}, "judge") == []


async def test_test_deadline_closes_provider_and_keeps_fresh_evidence(monkeypatch, tmp_path) -> None:
    args, _ = scheduled(monkeypatch, tmp_path)
    closed = []

    class ClosingScript(Script):
        async def aclose(self):
            closed.append(True)

    async def slow(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(providers, "build_provider", lambda settings: ClosingScript())
    monkeypatch.setattr(published, "run_test", slow)
    monkeypatch.setattr(published, "TEST_TIMEOUT_S", 0.01)
    assert await published._main(args) == 0
    assert closed == [True] and not args.out.exists()


async def test_healthy_traffic_adopts_hourly_updates_once_without_model_calls(tmp_path) -> None:
    now = [0.0]
    fetched = []
    data = publishable(tested=(datetime.now(UTC) - timedelta(hours=2)).strftime(published.STAMP))

    def fetch():
        fetched.append(True)
        return json.dumps(data)

    provider = Script()
    chooser = ModelChooser(automatic(tmp_path), provider, fetch_published=fetch, clock=lambda: now[0])
    await chooser.start()
    data["tested"] = datetime.now(UTC).strftime(published.STAMP)
    data["chains"]["light"] = ["new/model:free"]
    now[0] = PUBLISHED_REFRESH_S
    for _ in range(10):
        chooser.note([Attempt("a/x:free", "light", "ok")])
    await chooser.wait()
    assert len(fetched) == 2 and provider.calls == []
    assert chooser.chains["light"] == ["new/model:free"]
    chooser.note([Attempt("new/model:free", "light", "ok")])
    await chooser.wait()
    assert len(fetched) == 2


async def test_unconfirmed_publication_cannot_replace_working_runtime_chains(tmp_path) -> None:
    data = publishable(tested=datetime.now(UTC).strftime(published.STAMP), confirmed=False)
    settings = automatic(tmp_path)
    chooser = ModelChooser(settings, Script(), fetch_published=lambda: json.dumps(data))
    await chooser.start()
    assert chooser.chains["light"] == settings.chain("light")
