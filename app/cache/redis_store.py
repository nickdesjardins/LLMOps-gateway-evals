"""The Redis vector store: an HNSW index with cosine distance, shared by every instance.

Redis 8 (and Redis Stack before it) ships the Query Engine, which indexes a HASH field of
packed float32 values as a vector. The layout, for a key prefix ``llmops:cache:``:

* ``llmops:cache:e:<entry id>``: a HASH with the namespace (a TAG), the vector, the question,
  the payload as JSON and the costs. It expires with the entry (``EXPIRE``).
* ``llmops:cache:x:<exact key>``: a plain string pointing at the entry id, for the lookup of an
  exact repeat. That one is a GET: no embedding, no vector search.

The index is created at start-up if it is missing::

    FT.CREATE <index>-<dim> ON HASH PREFIX 1 llmops:cache:e:
        SCHEMA ns TAG embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE

and a nearest-neighbour query is one round trip::

    FT.SEARCH <index> "(@ns:{<namespace>})=>[KNN 3 @embedding $vec AS dist]"
        PARAMS 2 vec <bytes> RETURN ... SORTBY dist LIMIT 0 3 DIALECT 2

Redis reports a cosine *distance*, ``1 - cosine similarity``; this module converts it back.
The dimension is part of the index name, so a change of embedding model starts a new index
instead of failing on the old one.

The cache must never take a request down with it. Every call has a short socket timeout; a
failure is a miss (or a skipped write), and after three failures in a row the store stops
calling Redis for 30 seconds.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from app.cache.base import CacheEntry, CacheUnavailable, Neighbor

logger = logging.getLogger(__name__)

_RETURN_FIELDS = (
    "ns",
    "exact",
    "question",
    "payload",
    "created_at",
    "expires_at",
    "reference_cost_usd",
    "actual_cost_usd",
    "metadata",
    "dist",
)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _entry_from_fields(entry_id: str, fields: dict[str, str]) -> CacheEntry:
    return CacheEntry(
        entry_id=entry_id,
        namespace=fields["ns"],
        exact_key=fields["exact"],
        question=fields["question"],
        payload=json.loads(fields["payload"]),
        created_at=float(fields["created_at"]),
        expires_at=float(fields["expires_at"]),
        reference_cost_usd=fields.get("reference_cost_usd", "0"),
        actual_cost_usd=fields.get("actual_cost_usd", "0"),
        metadata=json.loads(fields.get("metadata") or "{}"),
    )


class RedisVectorStore:
    name = "redis"

    def __init__(
        self,
        client: Any,
        *,
        dim: int,
        index_name: str,
        key_prefix: str,
        max_entries: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.dim = dim
        self.index = f"{index_name}-{dim}"
        self.prefix = key_prefix
        self.max_entries = max_entries
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0
        self._count = 0
        self._count_at = 0.0

    # ------------------------------------------------------------------ set-up
    @classmethod
    async def connect(
        cls, url: str, *, dim: int, index_name: str, key_prefix: str, max_entries: int
    ) -> RedisVectorStore:
        from redis.asyncio import Redis  # imported here: the hermetic suite never connects

        client = Redis.from_url(
            url,
            protocol=2,  # the reply shapes parsed below are RESP2's
            decode_responses=False,
            socket_connect_timeout=3.0,
            socket_timeout=0.8,
            health_check_interval=30,
        )
        store = cls(client, dim=dim, index_name=index_name, key_prefix=key_prefix, max_entries=max_entries)
        await store.ensure_index()
        return store

    async def ensure_index(self) -> None:
        """Create the index when it is missing. Raises CacheUnavailable without vector search."""
        try:
            await self.client.execute_command("FT.INFO", self.index)
            return
        except Exception as exc:  # a missing index is reported as an error too
            message = str(exc).lower()
            if "unknown command" in message:
                raise CacheUnavailable(
                    "this Redis server has no vector search (FT.* commands): use Redis 8 or "
                    "Redis Stack, or set CACHE_BACKEND=memory"
                ) from exc
            if "unknown index" not in message and "no such index" not in message:
                raise CacheUnavailable(f"Redis did not answer FT.INFO ({type(exc).__name__})") from exc
        await self.client.execute_command(
            "FT.CREATE",
            self.index,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            f"{self.prefix}e:",
            "SCHEMA",
            "ns",
            "TAG",
            "embedding",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self.dim),
            "DISTANCE_METRIC",
            "COSINE",
        )
        logger.info("Created the Redis vector index %s (HNSW, cosine, %d dimensions)", self.index, self.dim)

    # ------------------------------------------------------------- the breaker
    def _available(self) -> bool:
        return self._clock() >= self._open_until

    def _ok(self) -> None:
        self._failures = 0

    def _failed(self, operation: str, exc: BaseException) -> None:
        self._failures += 1
        logger.warning("Redis %s failed (%s); treated as a cache miss", operation, type(exc).__name__)
        if self._failures >= 3:
            self._open_until = self._clock() + 30.0
            self._failures = 0
            logger.warning("Redis failed three times in a row: the cache is skipped for 30 s")

    # ------------------------------------------------------------------ reads
    async def get_exact(self, namespace: str, exact_key: str) -> CacheEntry | None:
        if not self._available():
            return None
        try:
            entry_id = await self.client.get(f"{self.prefix}x:{exact_key}")
            if entry_id is None:
                self._ok()
                return None
            raw = await self.client.hgetall(f"{self.prefix}e:{_text(entry_id)}")
            self._ok()
        except Exception as exc:
            self._failed("GET", exc)
            return None
        if not raw:
            return None
        fields = {_text(k): _text(v) for k, v in raw.items() if _text(k) != "embedding"}
        entry = _entry_from_fields(_text(entry_id), fields)
        if entry.namespace != namespace or entry.expires_at <= self._clock():
            return None
        return entry

    async def nearest(self, namespace: str, vector: np.ndarray, k: int = 3) -> list[Neighbor]:
        if not self._available():
            return []
        args: list[Any] = [
            "FT.SEARCH",
            self.index,
            f"(@ns:{{{namespace}}})=>[KNN {k} @embedding $vec AS dist]",
            "PARAMS",
            "2",
            "vec",
            vector.astype(np.float32).tobytes(),
            "RETURN",
            str(len(_RETURN_FIELDS)),
            *_RETURN_FIELDS,
            "SORTBY",
            "dist",
            "LIMIT",
            "0",
            str(k),
            "DIALECT",
            "2",
        ]
        try:
            reply = await self.client.execute_command(*args)
            self._ok()
        except Exception as exc:
            self._failed("FT.SEARCH", exc)
            return []
        # RESP2: [total, key, [field, value, ...], key, [...], ...]
        found: list[Neighbor] = []
        if not reply or int(reply[0]) == 0:
            return found
        now = self._clock()
        for i in range(1, len(reply) - 1, 2):
            key = _text(reply[i])
            flat = reply[i + 1]
            fields = {_text(flat[j]): _text(flat[j + 1]) for j in range(0, len(flat) - 1, 2)}
            entry = _entry_from_fields(key.removeprefix(f"{self.prefix}e:"), fields)
            if entry.expires_at <= now:
                continue
            similarity = float(np.clip(1.0 - float(fields["dist"]), -1.0, 1.0))
            found.append(Neighbor(entry=entry, similarity=similarity))
        found.sort(key=lambda n: -n.similarity)
        return found

    # ------------------------------------------------------------------ writes
    async def count(self) -> int:
        if self._clock() - self._count_at < 30.0:
            return self._count
        try:
            info = await self.client.execute_command("FT.INFO", self.index)
            pairs = {_text(info[i]): info[i + 1] for i in range(0, len(info) - 1, 2)}
            self._count = int(_text(pairs.get("num_docs", 0)))
            self._count_at = self._clock()
        except Exception as exc:
            self._failed("FT.INFO", exc)
        return self._count

    async def put(self, entry: CacheEntry, vector: np.ndarray) -> bool:
        if not self._available():
            return False
        if await self.count() >= self.max_entries:
            logger.info(
                "The Redis cache holds %d entries (the bound); this answer is not stored", self._count
            )
            return False
        ttl = max(1, int(entry.expires_at - self._clock()))
        key = f"{self.prefix}e:{entry.entry_id}"
        mapping = {
            "ns": entry.namespace,
            "exact": entry.exact_key,
            "question": entry.question,
            "payload": json.dumps(entry.payload, separators=(",", ":")),
            "created_at": repr(entry.created_at),
            "expires_at": repr(entry.expires_at),
            "reference_cost_usd": entry.reference_cost_usd,
            "actual_cost_usd": entry.actual_cost_usd,
            "metadata": json.dumps(entry.metadata, separators=(",", ":")),
            "embedding": vector.astype(np.float32).tobytes(),
        }
        try:
            await self.client.hset(key, mapping=mapping)
            await self.client.expire(key, ttl)
            await self.client.set(f"{self.prefix}x:{entry.exact_key}", entry.entry_id, ex=ttl)
            self._ok()
        except Exception as exc:
            self._failed("HSET", exc)
            return False
        self._count += 1
        return True

    async def clear(self) -> int:
        removed = 0
        async for key in self.client.scan_iter(match=f"{self.prefix}*", count=500):
            removed += int(await self.client.delete(key))
        self._count = 0
        return removed

    async def ping(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:  # pragma: no cover - closing a dead connection
            pass
