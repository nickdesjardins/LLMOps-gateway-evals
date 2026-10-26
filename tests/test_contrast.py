"""The contrast guard, on the pairs measured in site/recorded/cache-threshold.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cache.contrast import contrast_conflict
from app.cache.semantic import normalize_question

INTENTS = Path(__file__).resolve().parent.parent / "evals" / "cache_intents.json"


@pytest.mark.parametrize(
    ("a", "b", "why"),
    [
        (
            "what is the daily meal limit when travelling domestically",
            "what is the daily meal limit when travelling internationally",
            "opposite",
        ),
        (
            "how fast must a p1 ticket get a first response",
            "how fast must a p2 ticket get a first response",
            "codes",
        ),
        (
            "who approves an expense report above 10,000 dollars",
            "who approves an expense report under 2,500 dollars",
            "numbers",
        ),
        (
            "who approves an expense report above 2,500 dollars",
            "who approves an expense report below 2,500 dollars",
            "opposite",
        ),
        ("can full-time staff do this", "can part-time staff do this", "opposite"),
    ],
)
def test_contradictions_are_caught(a: str, b: str, why: str) -> None:
    reason = contrast_conflict(a, b)
    assert reason is not None and why in reason


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "what approval do i need before an international trip",
            "what approval is needed for travel outside the country",
        ),
        ("how long must my password be", "what is the minimum password length"),
        (
            "who approves an expense report above 10,000 dollars",
            "who must approve an expense report over 10,000 dollars",
        ),
        (
            "how many vacation days do full-time employees get each year",
            "how many vacation days do full time employees get per year",
        ),
    ],
)
def test_paraphrases_are_not_blocked(a: str, b: str) -> None:
    assert contrast_conflict(a, b) is None


def test_no_phrasing_pair_of_one_intent_is_blocked() -> None:
    data = json.loads(INTENTS.read_text(encoding="utf-8"))
    for intent in data["intents"]:
        phrasings = [normalize_question(p) for p in intent["phrasings"]]
        for i, a in enumerate(phrasings):
            for b in phrasings[i + 1 :]:
                assert contrast_conflict(a, b) is None, (intent["id"], a, b)


def test_a_substituted_name_is_the_documented_blind_spot() -> None:
    # Toronto (225 dollars) and Ottawa (175) have different answers; the guard cannot know.
    assert (
        contrast_conflict(
            "what is the nightly hotel cap in toronto", "what is the nightly hotel cap in ottawa"
        )
        is None
    )
