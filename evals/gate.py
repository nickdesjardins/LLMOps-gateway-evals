"""The quality gate: read the reports, apply the thresholds, pass or fail the build.

Two kinds of rule, both in ``evals/thresholds.json``:

* **Floors** that hold whatever happened before: the hallucination rate may not exceed 5%,
  faithfulness may not fall below 0.95, every adversarial case must resist its injection,
  retrieval must find the expected passage for at least 95% of the questions.
* **Regressions** against the accepted baseline (``evals/baseline.json``): a tracked metric
  may not drop more than the tolerance (0.03) below the value that was last accepted on purpose.
  Without a baseline the regression rules are skipped, and the summary says so.

Missing evidence is a failure, never a pass: a report that should be there and is not, a case
that errored, a judge that could not score more than a couple of metrics. The script uses the
standard library only, so it runs in any job of the workflow.

    python -m evals.gate --retrieval evals/out/retrieval.json --deepeval evals/out/deepeval.json
    python -m evals.gate ... --write-baseline   # accept the current scores as the new baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
THRESHOLDS = HERE / "thresholds.json"
BASELINE = HERE / "baseline.json"


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    rule: str
    passed: bool
    note: str = ""


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value:.3f}"


def current_metrics(reports: dict[str, dict[str, Any] | None]) -> dict[str, float]:
    """``engine.metric`` -> value, for every numeric summary value of the reports present."""
    flat: dict[str, float] = {}
    for engine, report in reports.items():
        if report is None:
            continue
        for key, value in (report.get("summary") or {}).items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                flat[f"{engine}.{key}"] = float(value)
    return flat


def evaluate(
    reports: dict[str, dict[str, Any] | None],
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> list[Check]:
    checks: list[Check] = []

    def floor(engine: str, metric: str, minimum: float, label: str) -> None:
        report = reports.get(engine)
        value = None if report is None else report["summary"].get(metric)
        passed = value is not None and value >= minimum
        checks.append(
            Check(label, value, f">= {minimum}", passed, "" if value is not None else "not measured")
        )

    def ceiling(engine: str, metric: str, maximum: float, label: str) -> None:
        report = reports.get(engine)
        value = None if report is None else report["summary"].get(metric)
        passed = value is not None and value <= maximum
        checks.append(
            Check(label, value, f"<= {maximum}", passed, "" if value is not None else "not measured")
        )

    if "retrieval" in reports:
        rules = thresholds["retrieval"]
        floor("retrieval", "hit_rate", rules["hit_rate_min"], "retrieval hit rate@k")
        floor(
            "retrieval", "context_precision", rules["context_precision_min"], "retrieval context precision@k"
        )
    if "deepeval" in reports:
        rules = thresholds["deepeval"]
        ceiling("deepeval", "hallucination_rate", rules["hallucination_rate_max"], "hallucination rate")
        floor("deepeval", "faithfulness", rules["faithfulness_min"], "faithfulness (DeepEval)")
        floor("deepeval", "refusal_accuracy", rules["refusal_accuracy_min"], "refusal accuracy")
        floor("deepeval", "injection_resistance", rules["injection_resistance_min"], "injection resistance")
        ceiling(
            "deepeval",
            "cases_with_errors",
            rules["max_cases_with_errors"],
            "cases the pipeline failed to answer",
        )
        ceiling(
            "deepeval", "metric_failures", rules["max_metric_failures"], "metrics the judge failed to score"
        )

    regression = thresholds["regression"]
    tolerance = float(regression["tolerance"])
    accepted = (baseline or {}).get("metrics", {})
    now = current_metrics(reports)
    for name, direction in regression["metrics"].items():
        if name.split(".")[0] not in reports or name not in accepted:
            continue
        value, before = now.get(name), float(accepted[name])
        if direction == "higher":
            passed = value is not None and value >= before - tolerance
            rule = f">= {before:.3f} - {tolerance}"
        else:
            passed = value is not None and value <= before + tolerance
            rule = f"<= {before:.3f} + {tolerance}"
        checks.append(
            Check(f"no regression: {name}", value, rule, passed, "baseline" if passed else "declined")
        )
    return checks


def markdown(
    checks: list[Check], baseline: dict[str, Any] | None, reports: dict[str, dict[str, Any] | None]
) -> str:
    passed = all(c.passed for c in checks)
    lines = [
        f"## Quality gate: {'passed' if passed else 'FAILED'}",
        "",
        "| Check | Value | Rule | Result |",
        "|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.name} | {_fmt(c.value)} | {c.rule} | {'pass' if c.passed else 'FAIL'}{(' (' + c.note + ')') if c.note and not c.passed else ''} |"
        )
    if baseline is None:
        lines += ["", "No accepted baseline yet (evals/baseline.json): regression checks were skipped."]
    else:
        lines += ["", f"Baseline accepted {baseline.get('accepted_at', 'at an unknown date')}."]
    deepeval = reports.get("deepeval")
    if deepeval:
        bad = [
            c
            for c in deepeval["cases"]
            if c.get("hallucinated") or c.get("leaked") or c.get("refusal_correct") is False or c.get("error")
        ]
        if bad:
            lines += ["", "| Case | Category | Problem |", "|---|---|---|"]
            for c in bad:
                problems = list(c.get("hallucination_reasons") or [])
                if c.get("leaked"):
                    problems.append(f"leaked {', '.join(c['leaked'])}")
                if c.get("refusal_correct") is False:
                    problems.append(
                        "declined an answerable question"
                        if c["expected_behavior"] == "answer"
                        else "answered instead of declining"
                    )
                if c.get("error"):
                    problems.append(f"error: {c['error']}")
                lines.append(f"| {c['id']} | {c['category']} | {'; '.join(dict.fromkeys(problems))} |")
        judge = deepeval.get("judge") or {}
        lines += ["", f"Judge: {', '.join(judge.get('models', []))}, {judge.get('calls', 0)} calls."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the quality gate to the evaluation reports.")
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--deepeval", type=Path)
    parser.add_argument("--ragas", type=Path)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None,
    )
    parser.add_argument("--write-baseline", action="store_true", help="accept the current scores")
    args = parser.parse_args(argv)

    if not (args.retrieval or args.deepeval or args.ragas):
        parser.error("name at least one report")
    try:
        reports = {
            name: _load(path)
            for name, path in (
                ("retrieval", args.retrieval),
                ("deepeval", args.deepeval),
                ("ragas", args.ragas),
            )
            if path is not None
        }
    except FileNotFoundError as exc:
        print(f"Quality gate FAILED: the report {exc.args[0]} was not produced.", file=sys.stderr)
        return 1
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.is_file() else None

    if args.write_baseline:
        import time

        data = {
            "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                k: round(v, 4)
                for k, v in current_metrics(reports).items()
                if k in thresholds["regression"]["metrics"]
            },
        }
        args.baseline.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(f"Baseline written to {args.baseline}: {data['metrics']}")
        return 0

    checks = evaluate(reports, thresholds, baseline)
    text = markdown(checks, baseline, reports)
    print(text)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if checks and all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
