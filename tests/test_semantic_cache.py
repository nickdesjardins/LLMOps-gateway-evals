"""The semantic cache: exact path, the threshold boundary, namespaces, expiry, the guard."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from app.cache.memory import MemoryVectorStore
from app.cache.semantic import SemanticCache, namespace_for, normalize_question
from app.embeddings import DIMENSIONS
from tests.conftest import TableEmbedder, vector_at


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make_cache(
    table: dict[str, np.ndarray] | None = None, **kwargs: object
) -> tuple[SemanticCache, TableEmbedder, Clock]:
    clock = Clock()
    embedder = TableEmbedder(table)
    store = MemoryVectorStore(DIMENSIONS, max_entries=int(kwargs.pop("max_entries", 100)), clock=clock)  # type: ignore[arg-type]
    cache = SemanticCache(store, embedder, clock=clock, **kwargs)  # type: ignore[arg-type]
    return cache, embedder, clock


async def store(cache: SemanticCache, namespace: str, question: str, answer: str = "a") -> None:
    await cache.store_answer(
        namespace,
        question,
        {"answer": answer},
        reference_cost_usd=Decimal("0.001"),
        actual_cost_usd=Decimal(0),
    )


def test_normalization() -> None:
    assert normalize_question("  How MANY   days?! ") == "how many days"
    assert namespace_for("rag", "v1") != namespace_for("rag", "v2")
    assert namespace_for("x").isalnum() and len(namespace_for("x")) == 24


async def test_exact_repeat_hits_without_embedding() -> None:
    cache, embedder, _ = make_cache()
    await store(cache, "ns", "How many vacation days?")
    calls = embedder.calls
    result = await cache.lookup("ns", "how many VACATION days")
    assert result.outcome == "hit" and result.match == "exact" and result.similarity == 1.0
    assert embedder.calls == calls  # no embedding on the exact path
    assert result.entry is not None and result.entry.payload == {"answer": "a"}


@pytest.mark.parametrize(
    ("similarity", "outcome"), [(0.97, "hit"), (0.9601, "hit"), (0.9599, "miss"), (0.80, "miss")]
)
async def test_the_threshold_boundary(similarity: float, outcome: str) -> None:
    stored, incoming = vector_at(similarity)
    cache, _, _ = make_cache({"stored question": stored, "incoming question": incoming}, threshold=0.96)
    await store(cache, "ns", "stored question")
    result = await cache.lookup("ns", "incoming question")
    assert result.outcome == outcome
    assert result.similarity == pytest.approx(similarity, abs=1e-4)
    assert result.nearest_question == "stored question"
    assert result.vector is not None  # kept for the write after a miss


async def test_namespaces_never_share_answers() -> None:
    stored, incoming = vector_at(0.99)
    cache, _, _ = make_cache({"q one": stored, "q two": incoming})
    await store(cache, "prompt-v1", "q one")
    assert (await cache.lookup("prompt-v2", "q two")).outcome == "miss"
    assert (await cache.lookup("prompt-v2", "q one")).outcome == "miss"
    assert (await cache.lookup("prompt-v1", "q two")).outcome == "hit"


async def test_entries_expire() -> None:
    cache, _, clock = make_cache(ttl_s=60)
    await store(cache, "ns", "question")
    clock.now += 59
    assert (await cache.lookup("ns", "question")).outcome == "hit"
    clock.now += 2
    assert (await cache.lookup("ns", "question")).outcome == "miss"
    assert await cache.store.count() == 0


async def test_the_store_is_bounded_least_recently_used_first() -> None:
    cache, _, _ = make_cache(max_entries=2)
    await store(cache, "ns", "first")
    await store(cache, "ns", "second")
    await cache.lookup("ns", "first")  # touch: "second" is now the oldest
    await store(cache, "ns", "third")
    assert await cache.store.count() == 2
    assert (await cache.lookup("ns", "first")).outcome == "hit"
    assert (await cache.lookup("ns", "second")).outcome == "miss"


async def test_the_guard_blocks_a_contradiction_and_takes_the_next_neighbour() -> None:
    base = np.random.default_rng(3).normal(size=DIMENSIONS)
    near_miss, incoming = vector_at(0.995, base=base, seed=1)
    _, paraphrase = vector_at(0.97, base=incoming, seed=2)
    table = {
        "meal limit for domestic travel": near_miss,
        "meal limit for international travel": incoming,
        "daily meal limit abroad on international trips": paraphrase,
    }
    cache, _, _ = make_cache(table)
    await store(cache, "ns", "meal limit for domestic travel", "75")
    result = await cache.lookup("ns", "meal limit for international travel")
    assert result.outcome == "miss"
    assert result.blocked and "domestic/international" in result.blocked[0][2]
    await store(cache, "ns", "daily meal limit abroad on international trips", "110")
    result = await cache.lookup("ns", "meal limit for international travel")
    assert result.outcome == "hit" and result.entry is not None and result.entry.payload["answer"] == "110"
    assert len(result.blocked) == 1  # the domestic entry was still refused on the way


async def test_without_the_guard_the_near_miss_would_be_served() -> None:
    near_miss, incoming = vector_at(0.99)
    cache, _, _ = make_cache(
        {"limit for domestic trips": near_miss, "limit for international trips": incoming},
        contrast_guard=False,
    )
    await store(cache, "ns", "limit for domestic trips", "75")
    assert (await cache.lookup("ns", "limit for international trips")).outcome == "hit"


async def test_similarity_tool_uses_the_cache_normalization() -> None:
    cache, _, _ = make_cache()
    assert await cache.similarity("Same question?", "same question") == pytest.approx(1.0, abs=1e-6)


async def test_storing_twice_replaces_the_entry() -> None:
    cache, _, _ = make_cache()
    await store(cache, "ns", "q", "old")
    await store(cache, "ns", "Q?", "new")
    assert await cache.store.count() == 1
    result = await cache.lookup("ns", "q")
    assert result.entry is not None and result.entry.payload["answer"] == "new"
