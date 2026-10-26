"""Keeping the free chains working: the cases, the rules, the published file, the chooser."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.models import cases, published, rules
from app.models.catalogue import free_models, is_free_id
from app.models.chooser import ModelChooser
from app.models.testing import CaseResult, ModelReport, Pacer, run_test, try_model
from app.providers.base import ChatRequest, ProviderError
from app.rag.prompts import REFUSAL_TOKEN
from app.routing.cascade import Attempt
from tests.conftest import ScriptedProvider, make_settings

RIGHT = {
    "grounded": "Full-time employees accrue 20 vacation days per year [S1].",
    "refusal": REFUSAL_TOKEN,
    "synthesis": "Call the supplier back on a phone number already on file [S1]. Two people in accounts payable approve the change, and the first payment is held for 5 business days [S1]. The person who received it reports it with the Report Phish button [S2].",
    "judge": '{"verdicts": [{"verdict": "yes", "reason": "stated"}, {"verdict": "no", "reason": "contradicted"}]}',
}


def answering(case_answers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    """A scripted answer that recognizes each case by its prompt."""
    answers = {**RIGHT, **(case_answers or {})}

    def reply(model: str, request: ChatRequest) -> str:
        text = request.messages[-1]["content"]
        if "verdicts" in text:
            return answers["judge"]
        if "stock price" in text:
            return answers["refusal"]
        if "bank details" in text:
            return answers["synthesis"]
        return answers["grounded"]

    return reply


class Script(ScriptedProvider):
    """Per-model behaviour: a callable, an exception, or nothing (right answers)."""

    def __init__(self, behaviours: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.behaviours = behaviours or {}

    def _next(self, model: str, request: ChatRequest) -> Any:
        self.calls.append((model, request))
        behaviour = self.behaviours.get(model, answering())
        if isinstance(behaviour, BaseException):
            raise behaviour
        return behaviour(model, request)


def gone(model: str) -> ProviderError:
    return ProviderError(model, "not_found", "model not found (404)", status=404)


def harness_only(model: str) -> ProviderError:
    """OpenRouter's 403 for a free model kept for agentic harnesses (inkling, 26 October 2026)."""
    return ProviderError(
        model,
        "forbidden",
        "refused for this account (403)",
        status=403,
        provider_message=f"{model} is only available on agentic harnesses.",
    )


# ------------------------------------------------------------------------ the cases


@pytest.mark.parametrize(
    ("case_id", "text", "right"),
    [
        ("grounded", RIGHT["grounded"], True),
        ("grounded", "Twenty days a year.", False),  # no citation
        ("grounded", "They get 25 days [S1].", False),  # not 20
        ("grounded", "20 days [S7].", False),  # a passage that was not given
        ("refusal", REFUSAL_TOKEN, True),
        ("refusal", "The stock price is 42 dollars [S1].", False),
        ("synthesis", RIGHT["synthesis"], True),
        ("synthesis", "Call the supplier back on a number on file [S1].", False),  # no hold
        (
            "synthesis",
            "Call the supplier back on a number on file and hold 5 business days [S1]. Use Report Phish [S1].",
            False,
        ),  # one passage
        ("judge", RIGHT["judge"], True),
        ("judge", '```json\n{"verdicts": [{"verdict": "Yes"}, {"verdict": "No"},]}\n```', True),
        ("judge", '{"verdicts": [{"verdict": "yes"}, {"verdict": "yes"}]}', False),
        ("judge", "yes and no", False),
    ],
)
def test_case_grading(case_id: str, text: str, right: bool) -> None:
    assert cases.by_id(case_id).grade(text, False)[0] is right


def test_a_cut_off_answer_fails() -> None:
    assert cases.by_id("grounded").grade(RIGHT["grounded"], True) == (False, "cut off at the ceiling")


def test_case_messages_are_stable_bytes() -> None:
    assert cases.by_id("grounded").messages() == cases.by_id("grounded").messages()
    assert "_c0ffee42>" in cases.by_id("grounded").messages()[1]["content"]


# ---------------------------------------------------------------------- the tester


async def test_a_model_that_answers_right_qualifies_for_every_role() -> None:
    report = await try_model(Script(), "good/model:free", pace=Pacer(0))
    assert [r.outcome for r in report.results] == ["right"] * 4
    assert report.roles() == ["light", "heavy", "judge"] and report.verdict == "passed"


async def test_a_404_stops_the_test_of_that_model() -> None:
    provider = Script({"gone/model:free": gone("gone/model:free")})
    report = await try_model(provider, "gone/model:free", pace=Pacer(0))
    assert report.unreachable and report.verdict == "unreachable" and len(provider.calls) == 1


async def test_a_403_makes_that_model_unreachable_and_says_why() -> None:
    provider = Script({"closed/model:free": harness_only("closed/model:free")})
    report = await try_model(provider, "closed/model:free", pace=Pacer(0))
    assert report.verdict == "unreachable" and len(provider.calls) == 1
    result = report.results[0]
    assert result.failure == "forbidden" and result.why == "refused for this account (403)"
    assert result.provider_said == "closed/model:free is only available on agentic harnesses."
    # served by /api/v1/models and published in chain.json: the provider's words stay out
    assert "provider_said" not in result.as_dict() and "agentic" not in json.dumps(report.as_dict())


def closed_report(model: str, kind: str = "forbidden") -> ModelReport:
    return ModelReport(model, [CaseResult("grounded", "unreachable", 0.1, f"{kind} said so", kind)])


def test_one_model_s_403_is_not_a_refused_key() -> None:
    # the Model check of 26 October 2026 failed here: two of 17 models answered 403
    reports = [report("good/a:free"), closed_report("thinkingmachines/inkling:free"), report("good/b:free")]
    assert rules.key_refused(reports) is None


def test_a_401_or_a_403_from_every_model_is_a_refused_key() -> None:
    assert rules.key_refused([report("good/a:free"), closed_report("x/y:free", "auth")]) == (
        "OpenRouter refused the key (auth said so)"
    )
    every = [closed_report("x/a:free"), closed_report("x/b:free")]
    assert rules.key_refused(every) == (
        "OpenRouter refused every model with a 403, so the key itself is closed (forbidden said so)"
    )
    assert rules.key_refused([]) is None


async def test_a_rate_limit_leaves_the_model_not_judged() -> None:
    def busy(model: str, request: ChatRequest) -> str:
        raise ProviderError(model, "rate_limited", "429", status=429)

    report = await try_model(Script({"busy/model:free": busy}), "busy/model:free", pace=Pacer(0))
    assert report.verdict == "not_judged" and report.roles() == []


async def test_roles_follow_the_cases_passed() -> None:
    weak = answering({"synthesis": "Call the supplier back [S1].", "judge": "no idea"})
    report = await try_model(Script({"weak/model:free": weak}), "weak/model:free", pace=Pacer(0))
    assert report.roles() == ["light"]


async def test_only_free_models_are_tested() -> None:
    with pytest.raises(ValueError, match="not free"):
        await run_test(Script(), ["vendor/paid-model"], seconds_between_starts=0)


def test_a_slow_model_is_not_light() -> None:
    report = ModelReport(
        "slow/model:free",
        [CaseResult(c, "right", 30.0) for c in ("grounded", "refusal", "synthesis", "judge")],
    )
    assert report.roles() == ["heavy", "judge"]


# ------------------------------------------------------------------------ the rules


def report(
    model: str,
    roles: tuple[str, ...] = ("light", "heavy", "judge"),
    seconds: float = 2.0,
    unreachable: bool = False,
) -> ModelReport:
    if unreachable:
        return ModelReport(model, [CaseResult("grounded", "unreachable", 0.1, "404", "not_found")])
    results = []
    for case_id in ("grounded", "refusal", "synthesis", "judge"):
        needed_by = [role for role, needs in cases.NEEDS.items() if case_id in needs]
        ok = any(role in roles for role in needed_by)
        results.append(CaseResult(case_id, "right" if ok else "wrong", seconds))
    return ModelReport(model, results)


def test_repair_keeps_members_and_order_and_fills_behind() -> None:
    reports = {
        r.model: r
        for r in [
            report("a:free"),
            report("b:free", roles=()),
            report("c:free", seconds=1),
            report("d:free", seconds=3),
        ]
    }
    chain, placings = rules.repair(["a:free", "b:free"], reports, role="light")
    assert chain == ["a:free", "b:free", "c:free"]  # b failed its cases but keeps its place
    assert placings == {"a:free": "kept", "b:free": "kept", "c:free": "added"}
    assert rules.members_failing(["a:free", "b:free"], reports, "light") == ["b:free"]


def test_repair_drops_what_cannot_be_called() -> None:
    reports = {"a:free": report("a:free", unreachable=True), "n:free": report("n:free")}
    chain, placings = rules.repair(["a:free", "b:free"], reports, role="heavy", gone=["b:free"])
    assert chain == ["n:free"] and placings["a:free"] == placings["b:free"] == "dropped"


def test_repair_never_empties_a_chain() -> None:
    chain, _ = rules.repair(["a:free"], {"a:free": report("a:free", unreachable=True)}, role="light")
    assert chain == ["a:free"]


def test_preference_orders_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(rules.PREFERENCE, "heavy", ("vendor/big",))
    reports = [report("vendor/fast:free", seconds=1), report("vendor/big:free", seconds=9)]
    assert [r.model for r in rules.order_candidates(reports, "heavy")] == [
        "vendor/big:free",
        "vendor/fast:free",
    ]


# ------------------------------------------------------------------ published file


def publishable(**overrides: Any) -> dict[str, Any]:
    data = {
        "version": 1,
        "tested": "2026-10-26T06:30:00Z",
        "fingerprint": published.fingerprint(),
        "confirmed": True,
        "chains": {"light": ["a/x:free"], "heavy": ["b/y:free", "c/z:free"], "judge": ["b/y:free"]},
        "free_list": ["a/x:free", "b/y:free", "c/z:free"],
    }
    data.update(overrides)
    return data


NOW = datetime(2026, 10, 27, tzinfo=UTC)


def test_a_good_file_parses() -> None:
    found = published.parse(json.dumps(publishable()), now=NOW)
    assert found.chains["heavy"] == ["b/y:free", "c/z:free"] and found.tested.day == 26


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"version": 2}, "version 1"),
        ({"fingerprint": "0" * 16}, "another version"),
        ({"tested": "2027-01-01T00:00:00Z"}, "future"),
        (
            {
                "chains": {
                    "light": ["anthropic/claude-opus-5-5"],
                    "heavy": ["b/y:free"],
                    "judge": ["b/y:free"],
                }
            },
            "free model ids",
        ),
        ({"chains": {"light": ["a/x:free"] * 2, "heavy": ["b/y:free"], "judge": ["b/y:free"]}}, "different"),
        ({"chains": {"light": ["a/x:free"], "heavy": []}}, "heavy"),
    ],
)
def test_a_bad_file_is_refused(change: dict[str, Any], why: str) -> None:
    with pytest.raises(published.Unusable, match=why):
        published.parse(json.dumps(publishable(**change)), now=NOW)
    with pytest.raises(published.Unusable, match="64,000"):
        published.parse("x" * 70_000, now=NOW)


def test_when_a_test_is_due() -> None:
    found = published.parse(json.dumps(publishable()), now=NOW)
    free = ["a/x:free", "b/y:free", "c/z:free"]
    assert published.due(None, free, NOW) == "nothing usable is published"
    assert published.due(found, free, NOW) is None
    assert published.due(found, free, NOW, force=True)
    assert "six and a half days" in published.due(found, free, NOW + timedelta(days=7))  # type: ignore[operator]
    assert "changed" in published.due(found, [*free, "new/model:free"], NOW)  # type: ignore[operator]
    unconfirmed = published.parse(json.dumps(publishable(confirmed=False)), now=NOW)
    assert published.due(unconfirmed, free, NOW) is None  # 17.5 hours: not yet
    assert "confirmed nothing" in published.due(unconfirmed, free, NOW + timedelta(hours=3))  # type: ignore[operator]


def test_the_catalogue_keeps_free_text_models_only() -> None:
    listing = {
        "data": [
            {"id": "a/x:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "b/y", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {
                "id": "c/z:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["image"]},
            },
            {"id": "d/w:free", "pricing": {"prompt": "0.5", "completion": "0"}},
        ]
    }
    assert free_models("https://openrouter.ai/api/v1", fetch=lambda url: listing) == ["a/x:free"]
    assert (
        is_free_id("vendor/model-1.5:free") and not is_free_id("vendor/model") and not is_free_id("../x:free")
    )


def test_the_measured_defaults_are_free_and_well_formed() -> None:
    for role, chain in published.DEFAULT_CHAINS.items():
        assert 1 <= len(chain) <= rules.PLACES, role
        assert all(is_free_id(m) for m in chain), role


# --------------------------------------------------------------------- the chooser


def automatic(tmp_path: Any, **overrides: Any) -> Any:
    return make_settings(
        light_models="",
        heavy_models="",
        model_cache_path=str(tmp_path / "model_cache.json"),
        model_published_url=None,
        **overrides,
    )


async def test_a_failing_listed_model_is_probed_benched_and_replaced(tmp_path: Any) -> None:
    settings = automatic(tmp_path)
    light = settings.chain("light")
    broken = light[0]
    provider = Script({broken: gone(broken)})
    listed = [*light, *settings.chain("heavy"), "new/candidate:free"]
    chooser = ModelChooser(settings, provider, list_free=lambda: listed)
    assert chooser.mode == "automatic"
    chooser.note([Attempt(broken, "light", "not_found", "404")])
    await chooser.wait()
    assert broken in chooser.benched and broken not in chooser.chains["light"]
    assert "new/candidate:free" in chooser.chains["light"]  # tested, passed, added behind
    assert chooser.last_test is not None and chooser.origin["source"] == "repaired here"
    saved = json.loads((tmp_path / "model_cache.json").read_text(encoding="utf-8"))
    assert broken in saved["benched"]


async def test_a_403_on_a_real_request_is_probed_and_benched_like_a_404(tmp_path: Any) -> None:
    settings = automatic(tmp_path)
    closed = settings.chain("light")[0]
    provider = Script({closed: harness_only(closed)})
    listed = [*settings.chain("light"), *settings.chain("heavy")]
    chooser = ModelChooser(settings, provider, list_free=lambda: listed)
    chooser.note([Attempt(closed, "light", "forbidden", "refused for this account (403)")])
    await chooser.wait()
    assert closed in chooser.benched and closed not in chooser.chains["light"]


async def test_a_403_from_every_model_is_the_key_and_benches_nothing(tmp_path: Any) -> None:
    # a spent key budget: OpenRouter answers 403 to every model, so no model is to blame
    settings = automatic(tmp_path)
    light, heavy = settings.chain("light"), settings.chain("heavy")
    everyone = [*light, *heavy, "new/candidate:free"]
    provider = Script({m: harness_only(m) for m in everyone})
    chooser = ModelChooser(settings, provider, list_free=lambda: everyone)
    chooser.note([Attempt(m, "light", "forbidden", "403") for m in [*light, *heavy]])
    await chooser.wait()
    assert provider.calls == [] and chooser.benched == {}  # not even looked into
    benched = await chooser.look(light)  # two probes, both refused: still the key
    assert benched == [] and chooser.benched == {} and chooser.chains["light"] == light


async def test_benching_never_empties_a_chain(tmp_path: Any) -> None:
    settings = automatic(tmp_path)
    light = settings.chain("light")
    chooser = ModelChooser(settings, Script(), list_free=lambda: [])
    chooser.chains["light"][:] = light[:1]
    assert await chooser.look(light[:1]) == light[:1]  # unlisted: benched, but kept (rule 4)
    assert chooser.chains["light"] == light[:1]


async def test_candidates_that_all_refuse_are_not_benched(tmp_path: Any) -> None:
    settings = automatic(tmp_path)
    candidates = ["new/a:free", "new/b:free", "new/c:free"]
    listed = [*settings.chain("light"), *settings.chain("heavy"), *candidates]
    provider = Script({m: harness_only(m) for m in candidates})
    chooser = ModelChooser(settings, provider, list_free=lambda: listed)
    assert len(chooser.chains["light"]) < rules.PLACES  # a free place to fill
    assert await chooser.repair(reason="test") is None
    assert not any(m in chooser.benched for m in candidates)


def test_the_key_record_says_when_it_is_a_management_key() -> None:
    from app.models.catalogue import allowance

    record = allowance(
        "https://x.invalid/api/v1", "k", fetch=lambda url, headers: {"data": {"is_management_key": True}}
    )
    assert record["is_management_key"] is True


async def test_a_rate_limit_is_not_looked_into(tmp_path: Any) -> None:
    provider = Script()
    chooser = ModelChooser(automatic(tmp_path), provider, list_free=lambda: [])
    chooser.note([Attempt("a:free", "light", "rate_limited", "429")])
    await chooser.wait()
    assert provider.calls == [] and chooser.benched == {}


async def test_a_model_that_answers_the_probe_stays(tmp_path: Any) -> None:
    settings = automatic(tmp_path)
    model = settings.chain("light")[0]
    chooser = ModelChooser(settings, Script(), list_free=lambda: settings.chain("light"))
    assert await chooser.look([model]) == []
    assert model in chooser.chains["light"]


async def test_an_unlisted_model_is_benched_without_a_probe(tmp_path: Any) -> None:
    settings = automatic(tmp_path, model_repair=False)
    model = settings.chain("heavy")[0]
    provider = Script()
    chooser = ModelChooser(settings, provider, list_free=lambda: [])
    assert await chooser.look([model]) == [model]
    assert provider.calls == [] and model not in chooser.chains["heavy"]


async def test_looks_are_throttled_per_model(tmp_path: Any) -> None:
    calls: list[int] = []
    clock = iter([0.0, 10.0, 70.0])
    chooser = ModelChooser(
        automatic(tmp_path, model_repair=False),
        Script(),  # the probe answers: nothing is benched, so the next note is looked at again
        list_free=lambda: (calls.append(1), ["x/y:free"])[1],
        clock=lambda: next(clock),
    )
    for _ in range(3):
        chooser.note([Attempt("x/y:free", "light", "not_found", "404")])
        await chooser.wait()
    assert len(calls) == 2  # at 0 s and 70 s, not at 10 s


async def test_pinned_chains_are_never_touched(tmp_path: Any) -> None:
    settings = make_settings(model_cache_path=str(tmp_path / "c.json"), model_published_url=None)
    chooser = ModelChooser(settings, Script(), list_free=lambda: [])
    assert chooser.mode == "pinned"
    chooser.note([Attempt("light/a:free", "light", "not_found", "404")])
    await chooser.wait()
    assert chooser.chains["light"] == ["light/a:free", "light/b:free"]


async def test_newer_published_chains_are_adopted_at_start_up(tmp_path: Any) -> None:
    data = publishable(tested=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    settings = automatic(tmp_path)
    chooser = ModelChooser(settings, Script(), fetch_published=lambda: json.dumps(data))
    await chooser.start()
    assert chooser.chains == {"light": ["a/x:free"], "heavy": ["b/y:free", "c/z:free"]}
    assert chooser.judge == ["b/y:free"] and chooser.origin["source"] == "published"


async def test_older_or_unusable_published_chains_are_ignored(tmp_path: Any) -> None:
    settings = automatic(tmp_path)
    old = publishable(tested="2026-01-01T00:00:00Z")
    chooser = ModelChooser(settings, Script(), fetch_published=lambda: json.dumps(old))
    await chooser.start()
    assert chooser.chains["light"] == settings.chain("light")

    def boom() -> str:
        raise OSError("no network")

    chooser = ModelChooser(settings, Script(), fetch_published=boom)
    await chooser.start()
    assert chooser.origin["source"] == "measured default"


async def test_the_cascade_reports_every_request_to_the_chooser(tmp_path: Any) -> None:
    from tests.conftest import build_test_app

    settings = automatic(tmp_path, model_repair=False)
    broken = settings.chain("light")[0]
    _, components = await build_test_app(settings, provider=Script({broken: gone(broken)}))
    components.chooser._list_free = lambda: []  # OpenRouter no longer lists anything
    result = await components.pipeline.answer("How many vacation days do full-time employees get each year?")
    assert result["attempts"][0]["outcome"] == "not_found" and result["fallback"]
    await components.chooser.wait()
    assert broken in components.chooser.benched
    second = await components.pipeline.answer("How many paid sick days do employees get?")
    assert all(a["model"] != broken for a in second["attempts"])
