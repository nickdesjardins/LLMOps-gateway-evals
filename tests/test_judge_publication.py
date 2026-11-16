"""Both evaluation consumers require fresh confirmed evidence before adopting free judges."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any

import pytest

from app.config import Settings, parse_model_chain
from app.models import published
from evals import judge, score_ragas

NOW = datetime(2026, 11, 16, 12, tzinfo=UTC)
CHAIN = ["judge/confirmed:free", "judge/fallback:free"]


class Clock(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return NOW


@pytest.fixture
def publication(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any] | str], None]:
    monkeypatch.setattr(judge, "Settings", lambda: Settings(_env_file=None))
    for module in (judge, score_ragas, published):
        monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.delenv("RAGAS_JUDGE_MODELS", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_MODELS", raising=False)
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("MODEL_PUBLISHED_URL", "https://example.test/chain.json")

    def serve(document: dict[str, Any] | str) -> None:
        text = json.dumps(document) if isinstance(document, dict) else document
        monkeypatch.setattr(published, "fetch", lambda url: text)
        monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(text.encode()))

    return serve


def document(**changes: Any) -> dict[str, Any]:
    return {
        "version": published.VERSION,
        "fingerprint": published.fingerprint(),
        "tested": (NOW - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirmed": True,
        "chains": {role: CHAIN for role in published.DEFAULT_CHAINS},
        **changes,
    }


@pytest.mark.parametrize("consumer", [judge.published_judges, score_ragas.judge_models])
@pytest.mark.parametrize("age", [timedelta(days=1), timedelta(days=14)])
def test_fresh_confirmed_publications_are_adopted(
    publication: Callable[..., None], consumer: Callable[[], list[str]], age: timedelta
) -> None:
    publication(document(tested=(NOW - age).strftime("%Y-%m-%dT%H:%M:%SZ")))
    assert consumer() == CHAIN


@pytest.mark.parametrize("consumer", [judge.published_judges, score_ragas.judge_models])
@pytest.mark.parametrize(
    "changes",
    [
        {"confirmed": False},
        {"confirmed": "true"},
        {"tested": (NOW - timedelta(days=14, seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"tested": (NOW + timedelta(minutes=6)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"tested": "not a date"},
        {"tested": None},
        {"version": 2},
        {"chains": {"judge": ["vendor/paid"]}},
        {"chains": {"judge": ["@preset/alias:free"]}},
        {"chains": {"judge": [CHAIN[0], CHAIN[0]]}},
    ],
)
def test_unusable_publications_keep_the_measured_default(
    publication: Callable[..., None], consumer: Callable[[], list[str]], changes: dict[str, Any]
) -> None:
    publication(document(**changes))
    assert consumer() == parse_model_chain(judge.DEFAULT_OPENROUTER_JUDGE)


@pytest.mark.parametrize("consumer", [judge.published_judges, score_ragas.judge_models])
@pytest.mark.parametrize("text", ["not JSON", " " * 64_001], ids=["invalid-json", "oversized"])
def test_invalid_or_oversized_files_keep_the_measured_default(
    publication: Callable[..., None], consumer: Callable[[], list[str]], text: str
) -> None:
    publication(text)
    assert consumer() == parse_model_chain(judge.DEFAULT_OPENROUTER_JUDGE)


def test_explicit_models_bypass_both_publication_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("explicit judge models must not fetch a publication")

    monkeypatch.setattr(published, "fetch", unexpected)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected)
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "openrouter")
    monkeypatch.setenv("EVAL_JUDGE_MODELS", "configured/judge:free")
    monkeypatch.setenv("RAGAS_JUDGE_MODELS", "configured/judge:free")
    assert judge.GatewayJudge(provider=object()).models == ["configured/judge:free"]
    assert score_ragas.judge_models() == ["configured/judge:free"]
