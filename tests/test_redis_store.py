"""The Redis vector store, against a fake that speaks the same commands (no server needed).

``tests/test_integration_redis.py`` runs the same store against a real Redis when
``TEST_REDIS_URL`` is set.
"""

from __future__ import annotations

import time
from decimal import Decimal

import numpy as np
import pytest

from app.cache.base import CacheEntry, CacheUnavailable
from app.cache.redis_store import RedisVectorStore
from app.cache.semantic import SemanticCache
from app.embeddings import DIMENSIONS
from tests.conftest import FakeRedis, TableEmbedder, vector_at


def entry(entry_id: str, namespace: str = "ns1", ttl: float = 600) -> CacheEntry:
    now = time.time()
    return CacheEntry(
        entry_id=entry_id,
        namespace=namespace,
        exact_key=f"exact-{entry_id}",
        question=f"question {entry_id}",
        payload={"answer": entry_id},
        created_at=now,
        expires_at=now + ttl,
        reference_cost_usd="0.0012",
    )


async def make_store(**kwargs: object) -> tuple[RedisVectorStore, FakeRedis]:
    fake = FakeRedis(**kwargs)  # type: ignore[arg-type]
    store = RedisVectorStore(
        fake, dim=DIMENSIONS, index_name="idx", key_prefix="llmops:cache:", max_entries=100
    )
    await store.ensure_index()
    return store, fake


async def test_the_index_is_hnsw_cosine_with_the_dimension_in_its_name() -> None:
    store, fake = await make_store()
    create = next(c for c in fake.commands if c[0] == "FT.CREATE")
    assert create[1] == f"idx-{DIMENSIONS}" == store.index
    flat = " ".join(map(str, create))
    assert "ON HASH PREFIX 1 llmops:cache:e:" in flat
    assert "ns TAG embedding VECTOR HNSW 6 TYPE FLOAT32" in flat
    assert f"DIM {DIMENSIONS} DISTANCE_METRIC COSINE" in flat
    await store.ensure_index()  # a second start finds the index and creates nothing
    assert sum(1 for c in fake.commands if c[0] == "FT.CREATE") == 1


async def test_a_server_without_vector_search_is_refused() -> None:
    with pytest.raises(CacheUnavailable, match="no vector search"):
        await make_store(search=False)


async def test_put_writes_the_hash_the_pointer_and_their_expiry() -> None:
    store, fake = await make_store()
    a, _ = vector_at(0.9)
    assert await store.put(entry("e1"), a)
    key = "llmops:cache:e:e1"
    assert fake.hashes[key]["ns"] == "ns1"
    assert np.frombuffer(fake.hashes[key]["embedding"], dtype=np.float32).shape == (DIMENSIONS,)
    assert fake.strings["llmops:cache:x:exact-e1"] == b"e1"
    assert 590 <= fake.ttl[key] <= 600 and 590 <= fake.ttl["llmops:cache:x:exact-e1"] <= 600


async def test_exact_lookup_and_namespace_check() -> None:
    store, _ = await make_store()
    await store.put(entry("e1"), vector_at(0.9)[0])
    found = await store.get_exact("ns1", "exact-e1")
    assert found is not None and found.payload == {"answer": "e1"} and found.reference_cost_usd == "0.0012"
    assert await store.get_exact("other", "exact-e1") is None
    assert await store.get_exact("ns1", "missing") is None


async def test_knn_query_is_namespaced_and_distance_becomes_similarity() -> None:
    store, fake = await make_store()
    stored, incoming = vector_at(0.97)
    await store.put(entry("e1"), stored)
    await store.put(entry("e2", namespace="ns2"), incoming)
    neighbors = await store.nearest("ns1", incoming, k=3)
    query = next(c for c in fake.commands if c[0] == "FT.SEARCH")
    assert query[2] == "(@ns:{ns1})=>[KNN 3 @embedding $vec AS dist]"
    assert "DIALECT" in query and query[query.index("DIALECT") + 1] == "2"
    assert [n.entry.entry_id for n in neighbors] == ["e1"]
    assert neighbors[0].similarity == pytest.approx(0.97, abs=1e-4)


async def test_the_breaker_opens_after_three_failures() -> None:
    store, fake = await make_store()
    fake.fail_next = 3
    for _ in range(3):
        assert await store.get_exact("ns1", "x") is None
    commands = len(fake.commands)
    assert await store.nearest("ns1", vector_at(0.9)[0]) == []  # skipped: no call reached Redis
    assert len(fake.commands) == commands
    assert await store.put(entry("e9"), vector_at(0.9)[0]) is False


async def test_a_full_store_refuses_writes() -> None:
    store, _ = await make_store()
    store.max_entries = 1
    assert await store.put(entry("e1"), vector_at(0.9)[0])
    store._count_at = 0  # force a fresh FT.INFO count
    assert await store.put(entry("e2"), vector_at(0.9)[0]) is False


async def test_the_semantic_cache_runs_on_the_redis_store() -> None:
    fake = FakeRedis()
    store = RedisVectorStore(fake, dim=DIMENSIONS, index_name="idx", key_prefix="t:", max_entries=100)
    await store.ensure_index()
    stored, incoming = vector_at(0.98)
    cache = SemanticCache(store, TableEmbedder({"stored q": stored, "incoming q": incoming}))
    await cache.store_answer(
        "ns", "stored q", {"answer": "x"}, reference_cost_usd=Decimal("0.002"), actual_cost_usd=Decimal(0)
    )
    assert (await cache.lookup("ns", "stored q")).match == "exact"
    result = await cache.lookup("ns", "incoming q")
    assert result.outcome == "hit" and result.match == "semantic" and result.backend == "redis"
    assert await store.clear() >= 2
