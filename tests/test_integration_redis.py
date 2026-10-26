"""The Redis vector store against a real Redis 8 (or Redis Stack). Opt-in.

Skipped unless ``TEST_REDIS_URL`` is set, so the hermetic suite stays hermetic::

    docker compose up -d redis
    TEST_REDIS_URL=redis://localhost:6379/0 python -m pytest tests/test_integration_redis.py

It uses its own index name and key prefix, and deletes its keys afterwards.
"""

from __future__ import annotations

import os
import secrets
from decimal import Decimal

import pytest

from app.cache.redis_store import RedisVectorStore
from app.cache.semantic import SemanticCache
from app.embeddings import DIMENSIONS
from tests.conftest import TableEmbedder, vector_at

URL = os.environ.get("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TEST_REDIS_URL to run against a real Redis")


@pytest.fixture(autouse=True)
def _allow_the_redis_host(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    from tests import conftest

    monkeypatch.setattr(socket.socket, "connect", conftest._real_connect)


async def test_a_real_index_answers_knn_with_cosine() -> None:
    suffix = secrets.token_hex(4)
    store = await RedisVectorStore.connect(
        URL, dim=DIMENSIONS, index_name=f"test-{suffix}", key_prefix=f"test:{suffix}:", max_entries=100
    )  # type: ignore[arg-type]
    try:
        stored, incoming = vector_at(0.97, seed=5)
        _, far = vector_at(0.50, base=stored, seed=6)
        cache = SemanticCache(
            store, TableEmbedder({"stored q": stored, "incoming q": incoming, "far q": far})
        )
        await cache.store_answer(
            "ns", "stored q", {"answer": "x"}, reference_cost_usd=Decimal("0.001"), actual_cost_usd=Decimal(0)
        )
        exact = await cache.lookup("ns", "stored q")
        assert exact.outcome == "hit" and exact.match == "exact"
        near = await cache.lookup("ns", "incoming q")
        assert (
            near.outcome == "hit"
            and near.match == "semantic"
            and near.similarity == pytest.approx(0.97, abs=1e-3)
        )
        assert (await cache.lookup("ns", "far q")).outcome == "miss"
        assert (await cache.lookup("other-namespace", "incoming q")).outcome == "miss"
        assert await store.count() >= 0
    finally:
        await store.clear()
        await store.client.execute_command("FT.DROPINDEX", store.index)
        await store.close()
