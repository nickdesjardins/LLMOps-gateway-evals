"""Light eligibility uses the latency of the requests that role actually serves."""

import pytest

from app.models import rules
from app.models.testing import CaseResult, ModelReport


@pytest.mark.parametrize(
    ("light_seconds", "other_seconds", "qualifies"),
    [
        (1.0, 30.0, True),  # Slow synthesis and judging cannot disqualify fast light answers.
        (20.0, 1.0, False),  # Fast unrelated answers cannot conceal slow light answers.
        (12.0, 30.0, True),  # The documented limit is inclusive.
        (None, 1.0, False),  # Other roles' measurements cannot stand in for missing light timings.
    ],
)
def test_light_latency_is_independent_of_other_roles(
    light_seconds: float | None, other_seconds: float, qualifies: bool
) -> None:
    report = ModelReport(
        "example/model:free",
        [
            CaseResult("grounded", "right", light_seconds),
            CaseResult("refusal", "right", light_seconds),
            CaseResult("synthesis", "right", other_seconds),
            CaseResult("judge", "right", other_seconds),
        ],
    )

    assert ("light" in report.roles()) is qualifies
    assert "heavy" in report.roles() and "judge" in report.roles()
    assert rules.members_failing([report.model], {report.model: report}, "light") == (
        [] if qualifies else [report.model]
    )


def test_overall_latency_remains_available_in_the_report() -> None:
    report = ModelReport(
        "example/model:free",
        [
            CaseResult("grounded", "right", 1.0),
            CaseResult("refusal", "right", 1.0),
            CaseResult("synthesis", "right", 30.0),
            CaseResult("judge", "right", 30.0),
        ],
    )

    assert report.as_dict()["median_s"] == 15.5
    assert "light" in report.as_dict()["roles"]
