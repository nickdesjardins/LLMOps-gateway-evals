"""The complexity router: named signals, tiers, hints."""

from __future__ import annotations

import pytest

from app.routing.complexity import ComplexityRouter

router = ComplexityRouter(heavy_threshold=0.5)


@pytest.mark.parametrize(
    "text",
    [
        "Classify this ticket: the printer on floor 3 is offline",
        "Extract the invoice number from: INV-2291 dated 3 May",
        "What is the daily meal limit for domestic travel?",
        "How many sick days do employees get?",
        "Is a receipt needed for 20 dollars? Answer yes or no.",
        "Return the priority as JSON",
    ],
)
def test_light_requests(text: str) -> None:
    decision = router.route(text)
    assert decision.tier == "light", decision.as_dict()


@pytest.mark.parametrize(
    "text",
    [
        "Compare the travel approval rules for domestic and international trips and explain why they differ.",
        "Walk me through what to do step by step when a supplier changes bank details, and which policies apply.",
        "Analyze the trade-offs between the expense limits and the travel policy for a two-week trip abroad.",
        "What happens if I lose my laptop abroad? Who do I call? How fast?",
    ],
)
def test_heavy_requests(text: str) -> None:
    decision = router.route(text)
    assert decision.tier == "heavy", decision.as_dict()


def test_signals_are_named_and_weighted() -> None:
    decision = router.route("Compare the domestic and international meal limits")
    names = {s.name for s in decision.signals}
    assert "comparison" in names
    assert decision.decided_by == "router"
    assert decision.as_dict()["signals"][0].keys() == {"name", "weight"}


def test_retrieval_spread_pushes_towards_heavy() -> None:
    text = "What do I need before a trip?"
    assert router.route(text, documents_in_play=1).tier == "light"
    assert router.route(text, documents_in_play=3).tier == "heavy"
    assert any(s.name == "spans_three_documents" for s in router.route(text, documents_in_play=3).signals)


def test_long_requests_and_conversations_count() -> None:
    long = "Please summarise the policy. " + "Some context about the situation at hand. " * 20
    assert any(s.name == "long_request" for s in router.route(long).signals)
    assert any(s.name == "long_conversation" for s in router.route("hi", turns=6).signals)


def test_hints_and_forced_tiers_decide_outright() -> None:
    hard = "Compare and analyze every policy step by step"
    assert router.route(hard, task="extract").tier == "light"
    assert router.route(hard, task="extract").decided_by == "task hint"
    assert router.route("What is the meal limit?", task="synthesize").tier == "heavy"
    assert router.route(hard, forced_tier="light").decided_by == "client"


def test_the_score_is_bounded_and_the_threshold_is_respected() -> None:
    score, _ = router.score("compare analyze explain plan if summarize? why? " * 5)
    assert score == 1.0
    lenient = ComplexityRouter(heavy_threshold=0.0)
    assert lenient.route("Classify this").tier == "heavy"
