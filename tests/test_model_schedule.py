"""Scheduled checks react to relevant changes and retain real quality failures."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app import providers
from app.models import catalogue, published
from tests.test_models import NOW, Script, answering, publishable


def full_record() -> published.Published:
    chain = ["a/x:free", "b/y:free", "c/z:free"]
    return published.parse(
        json.dumps(publishable(chains={role: chain for role in ("light", "heavy", "judge")})), now=NOW
    )


@pytest.mark.parametrize(
    "listing",
    [
        ["a/x:free", "b/y:free", "c/z:free", "new/model:free"],
        ["c/z:free", "a/x:free", "b/y:free"],
    ],
)
def test_full_chains_ignore_unrelated_catalogue_changes(listing: list[str]) -> None:
    assert published.due(full_record(), listing, NOW) is None


def test_removing_an_unused_model_does_not_spend_more_requests() -> None:
    previous = published.parse(
        json.dumps(publishable(free_list=["a/x:free", "b/y:free", "c/z:free", "old/model:free"])), now=NOW
    )
    assert published.due(previous, ["a/x:free", "b/y:free", "c/z:free"], NOW) is None


def test_a_withdrawn_chain_member_triggers_immediately() -> None:
    reason = published.due(full_record(), ["b/y:free", "c/z:free"], NOW)
    assert reason and "no longer listed: a/x:free" in reason


def test_new_candidates_only_trigger_when_a_role_has_space() -> None:
    previous = published.parse(json.dumps(publishable()), now=NOW)
    assert published.due(previous, previous.free_list, NOW) is None
    reason = published.due(previous, [*previous.free_list, "new/model:free"], NOW)
    assert reason and "open place" in reason


def test_full_chains_still_get_weekly_and_forced_checks() -> None:
    previous = full_record()
    assert published.due(previous, previous.free_list, NOW, force=True)
    assert published.due(previous, previous.free_list, NOW + timedelta(days=7))


def test_failure_annotation_and_job_output_escape_external_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    assert published._fail("100% failed\r\nsecond line") == 1
    assert "::error title=Model check failed::100%25 failed%0D%0Asecond line" in capsys.readouterr().out
    assert output.read_text() == "failure_reason=100%25 failed%0D%0Asecond line\n"


@pytest.mark.parametrize("bad_backup", [False, True])
async def test_scheduled_check_publishes_but_still_fails_bad_retained_models(
    bad_backup: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = ["good/primary:free", "bad/backup:free"]
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            publishable(
                tested=datetime.now(UTC).strftime(published.STAMP),
                chains={role: models for role in ("light", "heavy", "judge")},
                free_list=models,
            )
        ),
        encoding="utf-8",
    )
    provider = Script(
        {models[1]: answering({"synthesis": "Call the supplier back [S1]."})} if bad_backup else {}
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-only")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(published, "free_models", lambda base: models)
    monkeypatch.setattr(catalogue, "allowance", lambda base, key: {"remaining_today": 1000})
    monkeypatch.setattr(providers, "build_provider", lambda settings: provider)
    out = tmp_path / "chain.json"
    args = argparse.Namespace(previous=previous, out=out, force=True, pace_s=0)

    assert await published._main(args) == int(bad_backup)
    record = published.parse(out.read_text(encoding="utf-8"))
    assert record.confirmed
    assert record.chains["heavy"] == models  # passing backup order is never silently ranked
    log = capsys.readouterr().out
    if bad_backup:
        assert "heavy: bad/backup:free" in log
        assert "failed cases: synthesis: did not name the 5-business-day hold" in log
        assert "::error title=Model check failed::" in log
    else:
        assert "Failing this run" not in log


async def test_an_unconfirmed_role_still_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = ["bad/only:free"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-only")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(published, "DEFAULT_CHAINS", {role: models for role in ("light", "heavy", "judge")})
    monkeypatch.setattr(published, "free_models", lambda base: models)
    monkeypatch.setattr(catalogue, "allowance", lambda base, key: {"remaining_today": 1000})
    monkeypatch.setattr(
        providers, "build_provider", lambda settings: Script({models[0]: answering({"synthesis": "wrong"})})
    )
    out = tmp_path / "chain.json"
    assert await published._main(argparse.Namespace(previous=None, out=out, force=True, pace_s=0)) == 1
    assert not published.parse(out.read_text(encoding="utf-8")).confirmed
