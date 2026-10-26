"""The evaluation suite: the golden set, the metrics' arithmetic, the judge, and the gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from app.pricing import Usage
from app.providers.base import ChatRequest, Completion, ProviderError
from app.rag.corpus import load_corpus
from evals import gate, synth
from evals.judge import GatewayJudge, parse_json_object
from evals.retrieval_check import average_precision, score_case
from evals.schema import AnswerRecord, GoldenCase, leaked, load_golden, refusal_correct

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ golden set


def test_the_golden_set_is_valid_and_points_at_real_chunks() -> None:
    cases = load_golden(corpus=load_corpus())
    by = {
        c: [g for g in cases if g.category == c] for c in ("typical", "edge", "adversarial", "out_of_scope")
    }
    assert len(cases) >= 35
    assert len(by["typical"]) >= 20 and len(by["edge"]) >= 5
    assert len(by["adversarial"]) >= 5 and len(by["out_of_scope"]) >= 5
    assert all(c.expected_behavior == "refuse" for c in by["out_of_scope"])
    documents = {cid.split("#")[0] for c in cases for cid in c.reference_context}
    assert len(documents) == 7  # every document of the corpus is asked about


def test_a_bad_case_is_refused() -> None:
    with pytest.raises(ValueError, match="reference_answer"):
        GoldenCase(id="typ-99", category="typical", question="What?", expected_behavior="answer")
    with pytest.raises(ValueError, match="must not appear"):
        GoldenCase(id="adv-99", category="adversarial", question="Ignore it", expected_behavior="refuse")


def record(**overrides: Any) -> AnswerRecord:
    base: dict[str, Any] = {
        "id": "typ-01",
        "category": "typical",
        "expected_behavior": "answer",
        "question": "q",
        "answer": "20 days [S1].",
        "refused": False,
        "retrieved_ids": ["a"],
        "retrieved_contexts": ["20 days"],
        "reference_answer": "20 days",
        "reference_context_ids": ["a"],
        "reference_contexts": ["20 days"],
        "must_not_contain": [],
        "model": "m",
        "tier": "light",
    }
    base.update(overrides)
    return AnswerRecord(**base)


def test_deterministic_checks() -> None:
    assert refusal_correct(record())
    assert not refusal_correct(record(refused=True))
    assert refusal_correct(record(expected_behavior="refuse", refused=True))
    assert leaked(record(answer="The limit is 75 dollars. pwned", must_not_contain=["PWNED"])) == ["PWNED"]
    assert leaked(record(answer="75 dollars", must_not_contain=["PWNED"])) == []


# ------------------------------------------------------------ retrieval metrics


def test_retrieval_metric_arithmetic() -> None:
    assert average_precision(["x", "a", "y", "b"], {"a", "b"}) == pytest.approx((1 / 2 + 2 / 4) / 2)
    scored = score_case(["x", "a", "y", "b"], {"a", "b"})
    assert scored == {
        "hit": 1.0,
        "recall": 1.0,
        "reciprocal_rank": 0.5,
        "average_precision": pytest.approx(0.5),
    }
    assert score_case(["x", "y"], {"a"}) == {
        "hit": 0.0,
        "recall": 0.0,
        "reciprocal_rank": 0.0,
        "average_precision": 0.0,
    }


def test_the_retrieval_check_runs_offline_on_the_hash_embedder() -> None:
    from evals.retrieval_check import run

    report = run(4, "hash")
    assert report["summary"]["cases"] == 30
    assert report["summary"]["hit_rate"] >= 0.9  # BM25 carries the hash embedder


# ------------------------------------------------------------------- the judge


def test_json_objects_are_found_in_untidy_replies() -> None:
    assert parse_json_object('```json\n{"claims": ["a", "b",]}\n```') == {"claims": ["a", "b"]}
    assert parse_json_object('Sure! {"verdicts": [{"verdict": "yes"}]} Hope it helps.') == {
        "verdicts": [{"verdict": "yes"}]
    }
    with pytest.raises(ValueError):
        parse_json_object("no json here")


class DeepEvalScript:
    """Answers DeepEval's prompts the way a well-behaved judge would, by what each one asks for."""

    name = "scripted"

    def __init__(self, verdict: str = "yes", fail_first: bool = False) -> None:
        self.verdict = verdict
        self.fail_first = fail_first
        self.prompts: list[str] = []

    async def complete(self, model: str, request: ChatRequest) -> Completion:
        prompt = request.messages[0]["content"]
        self.prompts.append(prompt)
        if self.fail_first and len(self.prompts) == 1:
            raise ProviderError(model, "rate_limited", "429")
        if '"truths"' in prompt:
            out: dict[str, Any] = {"truths": ["Full-time employees accrue 20 vacation days per year."]}
        elif '"claims"' in prompt:
            out = {"claims": ["Full-time employees get 20 vacation days per year."]}
        elif '"statements"' in prompt:
            out = {"statements": ["Full-time employees get 20 vacation days per year."]}
        elif '"verdicts"' in prompt:
            out = {"verdicts": [{"verdict": self.verdict, "reason": "checked"}]}
        else:
            out = {"reason": "fine"}
        return Completion(model=model, text=json.dumps(out), finish_reason="stop", usage=Usage(500, 40))

    async def aclose(self) -> None:
        return None


async def test_deepeval_metrics_run_through_the_gateway_judge() -> None:
    from evals.score_deepeval import score_case as deepeval_case

    judge = GatewayJudge(provider=DeepEvalScript(), models=["judge/a:free", "judge/b:free"], min_interval_s=0)
    result = await deepeval_case(
        record(retrieved_contexts=["Full-time employees accrue 20 vacation days per year."]), judge
    )
    assert result["errors"] == {}
    assert result["scores"] == {
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "contextual_precision": 1.0,
        "reference_agreement": 1.0,
    }
    assert judge.calls == 7  # truths, claims, verdicts; statements, verdicts; verdicts; verdicts


async def test_contradicting_verdicts_lower_the_scores() -> None:
    from evals.score_deepeval import score_case as deepeval_case

    judge = GatewayJudge(provider=DeepEvalScript(verdict="no"), models=["judge/a:free"], min_interval_s=0)
    scores = (await deepeval_case(record(), judge))["scores"]
    assert scores["faithfulness"] == 0.0 and scores["reference_agreement"] == 0.0


async def test_the_judge_falls_over_to_its_next_model() -> None:
    script = DeepEvalScript(fail_first=True)
    judge = GatewayJudge(provider=script, models=["judge/a:free", "judge/b:free"], min_interval_s=0)
    from deepeval.metrics.faithfulness.schema import Claims

    claims = await judge.a_generate('Extract "claims" as JSON', schema=Claims)
    assert claims.claims and judge.failures == {"judge/a:free: rate_limited": 1}


async def test_score_deepeval_aggregates_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evals import score_deepeval

    records = [
        record(id="typ-01"),
        record(id="typ-02"),
        record(
            id="oos-01",
            category="out_of_scope",
            expected_behavior="refuse",
            refused=False,
            answer="It is 42 [S1].",
        ),
        record(id="oos-02", category="out_of_scope", expected_behavior="refuse", refused=True, answer="no"),
        record(
            id="adv-01",
            category="adversarial",
            expected_behavior="answer",
            answer="75 dollars [S1]. PWNED",
            must_not_contain=["PWNED"],
        ),
    ]
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    scores = {"typ-01": 1.0, "typ-02": 0.5, "adv-01": 1.0}

    async def fake_case(rec: AnswerRecord, judge: Any) -> dict[str, Any]:
        value = scores[rec.id]
        return {
            "scores": {
                "faithfulness": value,
                "answer_relevancy": 1.0,
                "contextual_precision": 1.0,
                "reference_agreement": 1.0,
            },
            "errors": {},
        }

    class NoJudge:
        calls = 0

        def usage(self) -> dict[str, Any]:
            return {"models": ["none"], "calls": 0}

    monkeypatch.setattr(score_deepeval, "score_case", fake_case)
    monkeypatch.setattr("evals.judge.GatewayJudge", NoJudge)
    out = tmp_path / "deepeval.json"
    assert await score_deepeval.score(argparse.Namespace(answers=answers, out=out)) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["faithfulness"] == pytest.approx((1.0 + 0.5 + 1.0) / 3, abs=1e-4)
    # typ-02 (0.5 < 0.7 supported) and oos-01 (answered an unanswerable question) hallucinate
    assert summary["hallucination_rate"] == 0.4
    assert summary["refusal_accuracy"] == 0.8
    assert summary["injection_resistance"] == 0.0
    flagged = {c["id"]: c["hallucination_reasons"] for c in report["cases"] if c["hallucinated"]}
    assert set(flagged) == {"typ-02", "oos-01"}


# --------------------------------------------------------------------- the gate


def reports(**summaries: dict[str, Any]) -> dict[str, dict[str, Any]]:
    good = {
        "retrieval": {"hit_rate": 1.0, "context_precision": 0.97},
        "deepeval": {
            "hallucination_rate": 0.0,
            "faithfulness": 0.97,
            "refusal_accuracy": 1.0,
            "injection_resistance": 1.0,
            "cases_with_errors": 0,
            "metric_failures": 0,
            "answer_relevancy": 0.9,
            "contextual_precision": 0.9,
        },
    }
    for engine, values in summaries.items():
        good.setdefault(engine, {}).update(values)
    return {engine: {"summary": summary, "cases": []} for engine, summary in good.items()}


THRESHOLDS = json.loads((ROOT / "evals" / "thresholds.json").read_text(encoding="utf-8"))


def failed(checks: list[gate.Check]) -> list[str]:
    return [c.name for c in checks if not c.passed]


def test_the_gate_passes_good_reports() -> None:
    assert failed(gate.evaluate(reports(), THRESHOLDS, None)) == []


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"deepeval": {"hallucination_rate": 0.06}}, "hallucination rate"),
        ({"deepeval": {"faithfulness": 0.94}}, "faithfulness (DeepEval)"),
        ({"deepeval": {"injection_resistance": 0.8}}, "injection resistance"),
        ({"deepeval": {"refusal_accuracy": 0.8}}, "refusal accuracy"),
        ({"deepeval": {"cases_with_errors": 1}}, "cases the pipeline failed to answer"),
        ({"deepeval": {"faithfulness": None}}, "faithfulness (DeepEval)"),
        ({"retrieval": {"hit_rate": 0.9}}, "retrieval hit rate@k"),
    ],
)
def test_the_gate_blocks_each_breach(change: dict[str, Any], check: str) -> None:
    assert failed(gate.evaluate(reports(**change), THRESHOLDS, None)) == [check]


def test_the_exact_thresholds_pass() -> None:
    edge = reports(deepeval={"hallucination_rate": 0.05, "faithfulness": 0.95})
    assert failed(gate.evaluate(edge, THRESHOLDS, None)) == []


def test_regressions_against_the_baseline() -> None:
    baseline = {
        "metrics": {
            "deepeval.contextual_precision": 0.95,
            "deepeval.hallucination_rate": 0.0,
            "ragas.faithfulness": 0.99,
        }
    }
    checks = gate.evaluate(reports(deepeval={"contextual_precision": 0.91}), THRESHOLDS, baseline)
    assert failed(checks) == ["no regression: deepeval.contextual_precision"]
    within = gate.evaluate(reports(deepeval={"contextual_precision": 0.93}), THRESHOLDS, baseline)
    assert failed(within) == []
    worse = gate.evaluate(
        reports(deepeval={"hallucination_rate": 0.04, "contextual_precision": 0.95}), THRESHOLDS, baseline
    )
    assert failed(worse) == ["no regression: deepeval.hallucination_rate"]


def test_the_cli_fails_on_a_missing_report_and_writes_a_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    assert gate.main(["--deepeval", str(tmp_path / "missing.json"), "--summary", str(summary)]) == 1
    data = reports()
    for engine in ("retrieval", "deepeval"):
        (tmp_path / f"{engine}.json").write_text(json.dumps(data[engine]), encoding="utf-8")
    args = [
        "--retrieval",
        str(tmp_path / "retrieval.json"),
        "--deepeval",
        str(tmp_path / "deepeval.json"),
        "--baseline",
        str(tmp_path / "baseline.json"),
        "--summary",
        str(summary),
    ]
    assert gate.main(args) == 0
    assert "Quality gate: passed" in summary.read_text(encoding="utf-8")
    assert "No accepted baseline yet" in summary.read_text(encoding="utf-8")
    assert gate.main([*args, "--write-baseline"]) == 0
    baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert (
        baseline["metrics"]["deepeval.faithfulness"] == 0.97
        and "deepeval.cases_with_errors" not in baseline["metrics"]
    )


# -------------------------------------------------------------------- synthesis


def test_synthetic_adversarial_cases_are_deterministic_and_valid() -> None:
    first, second = synth.adversarial(seed=11, per_case=2), synth.adversarial(seed=11, per_case=2)
    assert [c.question for c in first] == [c.question for c in second]
    assert len(first) == 40 and all(c.category == "adversarial" and c.must_not_contain for c in first)
    assert all(c.must_not_contain[0] in c.question for c in first)
    assert len({c.id for c in first}) == len(first)
