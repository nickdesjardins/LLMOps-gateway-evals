"""The cases a free model must pass to answer for this gateway, per role.

Being listed as free proves nothing about a model. The only evidence that counts is this
gateway's own work, sent through the same prompt, the same ceilings and the same validation a
real request uses (:mod:`app.rag.prompts`, :func:`app.rag.prompts.citation_problem`):

==============  =========================================================  =================
case            what is sent                                               right when
==============  =========================================================  =================
``grounded``    the RAG prompt, two passages, a question the first one     cites a passage it
                answers ("how many vacation days...")                     was given, says 20,
                                                                           is not cut off at
                                                                           the light ceiling
``refusal``     the RAG prompt, two passages that do not answer the        exactly the refusal
                question ("what is the stock price?")                      token
``synthesis``   the RAG prompt, three passages, a two-part question       cites both passages
                (a supplier's new bank details: what accounts payable      it needs, names the
                does, and how the suspicious email is reported)            callback, the 5-day
                                                                           hold and the Report
                                                                           Phish button, within
                                                                           the heavy ceiling
``judge``       a DeepEval-style verdict task: two claims, one context     JSON with one
                                                                           verdict per claim,
                                                                           yes then no
==============  =========================================================  =================

Which cases a role needs: **light** answers ``grounded`` and ``refusal`` (and must be quick);
**heavy** answers those and ``synthesis``; the eval **judge** answers ``judge``. A model is tested
on every case once and can then serve any role it qualifies for.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.rag.prompts import (
    REFUSAL_TOKEN,
    SYSTEM_PROMPT,
    PromptSource,
    build_messages,
    citation_problem,
    cited_labels,
    is_refusal,
)

Role = Literal["light", "heavy", "judge"]
ROLES: tuple[Role, ...] = ("light", "heavy", "judge")

#: Fixed nonce: the cases are the same bytes every time, so a fingerprint can name them.
NONCE = "c0ffee42"

_VACATION = PromptSource(
    "S1",
    "leave-policy#vacation",
    "Time Off and Leave Policy > Vacation",
    "Full-time employees accrue 1.67 vacation days per month of service, which is 20 days per year. "
    "From the fifth anniversary of their start date, full-time employees accrue 25 days per year.",
)
_VPN = PromptSource(
    "S2",
    "it-support#vpn-and-remote-access",
    "IT Service Desk Guide > VPN and remote access",
    "The VPN is required to reach internal systems from outside the office, and it uses multi-factor "
    "authentication. Public Wi-Fi may only be used with the VPN connected.",
)
_BANKING = PromptSource(
    "S1",
    "vendor-onboarding#banking-details-and-verification",
    "Supplier Onboarding Procedure > Banking details and verification",
    "Accounts payable enters a supplier's bank details only after calling the supplier back on a phone "
    "number already on file, never a number given in the same email or letter. A request to change bank "
    "details follows the same callback, needs the approval of two people in accounts payable, and the first "
    "payment to the new account is held for 5 business days.",
)
_PHISHING = PromptSource(
    "S2",
    "information-security#reporting-phishing",
    "Information Security Policy > Reporting phishing",
    "Report a suspicious email with the Report Phish button in Outlook, or forward it as an attachment to "
    "security@cedarridge.example. Do not click its links or open its attachments.",
)
_TERMS = PromptSource(
    "S3",
    "vendor-onboarding#payment-terms",
    "Supplier Onboarding Procedure > Payment terms",
    "The standard payment terms are net 45 days. An early payment discount such as 2 percent for payment "
    "within 10 days is accepted when a supplier offers it.",
)

JUDGE_PROMPT = """You are grading whether claims are supported by a context.

Context:
"Full-time employees accrue 20 vacation days per year. From the fifth anniversary they accrue 25 days per year."

Claims:
1. "Full-time employees get 20 vacation days per year."
2. "Full-time employees get 30 vacation days per year from their first day."

For each claim, in order, give a verdict: "yes" if the context supports it, "no" if the context contradicts it.

Return only a JSON object of this form, with exactly one verdict per claim:
{"verdicts": [{"verdict": "yes", "reason": "..."}, {"verdict": "no", "reason": "..."}]}"""


@dataclass(frozen=True)
class Case:
    case_id: str
    #: The ceiling the case is sent with: the tier's real one, so a model that cannot answer
    #: inside it fails here exactly as it would on a real request.
    tier: Literal["light", "heavy"]
    messages: Callable[[], list[dict[str, str]]]
    #: (right, why it is wrong). Never raises on what a model wrote.
    grade: Callable[[str, bool], tuple[bool, str]]
    json_mode: bool = False


def _grade_grounded(text: str, truncated: bool) -> tuple[bool, str]:
    if truncated:
        return False, "cut off at the ceiling"
    if is_refusal(text):
        return False, "refused an answerable question"
    problem = citation_problem(text, 2)
    if problem:
        return False, problem
    if not re.search(r"\b20\b", text):
        return False, "did not say 20 days"
    return True, ""


def _grade_refusal(text: str, truncated: bool) -> tuple[bool, str]:
    if is_refusal(text):
        return True, ""
    return False, "answered a question the passages do not answer"


def _grade_synthesis(text: str, truncated: bool) -> tuple[bool, str]:
    if truncated:
        return False, "cut off at the ceiling"
    if is_refusal(text):
        return False, "refused an answerable question"
    problem = citation_problem(text, 3)
    if problem:
        return False, problem
    lowered = text.lower()
    if not re.search(r"call(ing|ed)? (the supplier )?back|callback|phone number (already )?on file", lowered):
        return False, "did not name the callback"
    if not re.search(r"\b5\b|five", lowered) or "business day" not in lowered:
        return False, "did not name the 5-business-day hold"
    if not re.search(r"report phish|security@cedarridge", lowered):
        return False, "did not say how to report the email"
    labels = set(cited_labels(text))
    if not {1, 2} <= labels:
        return False, "did not cite both passages the answer needs"
    return True, ""


def _grade_judge(text: str, truncated: bool) -> tuple[bool, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return False, "no JSON object"
    try:
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", match.group(0)))
    except ValueError:
        return False, "invalid JSON"
    verdicts = data.get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list) or len(verdicts) != 2:
        return False, "not one verdict per claim"
    words = [str(v.get("verdict", "")).strip().lower() if isinstance(v, dict) else "" for v in verdicts]
    if words != ["yes", "no"]:
        return False, f"verdicts {words}, expected ['yes', 'no']"
    return True, ""


CASES: tuple[Case, ...] = (
    Case(
        "grounded",
        "light",
        lambda: build_messages(
            "How many vacation days do full-time employees get each year?", [_VACATION, _VPN], NONCE
        ),
        _grade_grounded,
    ),
    Case(
        "refusal",
        "light",
        lambda: build_messages("What is the company's stock price today?", [_VACATION, _VPN], NONCE),
        _grade_refusal,
    ),
    Case(
        "synthesis",
        "heavy",
        lambda: build_messages(
            "A supplier emailed us new bank details for their next payment. What should accounts payable do, "
            "and how should the person who received the email report it?",
            [_BANKING, _PHISHING, _TERMS],
            NONCE,
        ),
        _grade_synthesis,
    ),
    Case(
        "judge",
        "heavy",
        lambda: [{"role": "user", "content": JUDGE_PROMPT}],
        _grade_judge,
        json_mode=True,
    ),
)

#: The cases each role needs passed.
NEEDS: dict[Role, tuple[str, ...]] = {
    "light": ("grounded", "refusal"),
    "heavy": ("grounded", "refusal", "synthesis"),
    "judge": ("judge",),
}


def by_id(case_id: str) -> Case:
    return next(case for case in CASES if case.case_id == case_id)


def fingerprint_material() -> list[object]:
    """Everything a test result depends on: a change here makes an old published chain stale."""
    return [
        SYSTEM_PROMPT,
        REFUSAL_TOKEN,
        JUDGE_PROMPT,
        [(c.case_id, c.tier, c.json_mode, c.messages()) for c in CASES],
        {role: list(needs) for role, needs in NEEDS.items()},
    ]
