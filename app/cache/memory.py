"""The in-process vector store: a numpy matrix per namespace, bounded, with expiry.

Exact cosine search over every entry of a namespace: one matrix-vector product. With the
default bound of 5,000 entries of 384 dimensions that is about 7.5 MB and well under a
millisecond, so an approximate index would buy nothing here. Entries are evicted least recently
used first when the bound is reached, and expired ones are dropped as they are met.

Everything lives in this process: a restart (or a free host putting the service to sleep)
empties it. That is the trade-off Redis removes.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

from app.cache.base import CacheEntry, Neighbor


class _Namespace:
    def __init__(self, dim: int) -> None:
        self.ids: list[str] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, entry_id: str, vector: np.ndarray) -> None:
        self.ids.append(entry_id)
        self.matrix = np.vstack([self.matrix, vector.reshape(1, -1).astype(np.float32)])

    def remove(self, entry_id: str) -> None:
        index = self.ids.index(entry_id)
        del self.ids[index]
        self.matrix = np.delete(self.matrix, index, axis=0)


class MemoryVectorStore:
    name = "memory"

    def __init__(self, dim: int, max_entries: int = 5_000, clock: Callable[[], float] = time.time) -> None:
        self.dim = dim
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._exact: dict[tuple[str, str], str] = {}
        self._spaces: dict[str, _Namespace] = {}
        self._lock = asyncio.Lock()

    def _drop(self, entry_id: str) -> None:
        entry = self._entries.pop(entry_id, None)
        if entry is None:
            return
        self._exact.pop((entry.namespace, entry.exact_key), None)
        space = self._spaces.get(entry.namespace)
        if space is not None:
            space.remove(entry_id)

    def _alive(self, entry_id: str) -> CacheEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            self._drop(entry_id)
            return None
        self._entries.move_to_end(entry_id)
        return entry

    async def get_exact(self, namespace: str, exact_key: str) -> CacheEntry | None:
        async with self._lock:
            entry_id = self._exact.get((namespace, exact_key))
            return self._alive(entry_id) if entry_id else None

    async def nearest(self, namespace: str, vector: np.ndarray, k: int = 3) -> list[Neighbor]:
        async with self._lock:
            space = self._spaces.get(namespace)
            if space is None or not space.ids:
                return []
            # Expired entries are dropped as they are met, then the search runs on what is left.
            for entry_id in list(space.ids):
                self._alive_quiet(entry_id)
            if not space.ids:
                return []
            scores = space.matrix @ vector.astype(np.float32)
            order = np.argsort(-scores, kind="stable")[:k]
            found: list[Neighbor] = []
            for index in order:
                entry = self._alive(space.ids[int(index)])
                if entry is not None:
                    found.append(Neighbor(entry=entry, similarity=float(np.clip(scores[int(index)], -1, 1))))
            return found

    def _alive_quiet(self, entry_id: str) -> None:
        entry = self._entries.get(entry_id)
        if entry is not None and entry.expires_at <= self._clock():
            self._drop(entry_id)

    async def put(self, entry: CacheEntry, vector: np.ndarray) -> bool:
        async with self._lock:
            previous = self._exact.get((entry.namespace, entry.exact_key))
            if previous is not None:
                self._drop(previous)
            while len(self._entries) >= self.max_entries:
                oldest = next(iter(self._entries))
                self._drop(oldest)
            self._entries[entry.entry_id] = entry
            self._exact[(entry.namespace, entry.exact_key)] = entry.entry_id
            self._spaces.setdefault(entry.namespace, _Namespace(self.dim)).add(entry.entry_id, vector)
            return True

    async def count(self) -> int:
        return len(self._entries)

    async def clear(self) -> int:
        async with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._exact.clear()
            self._spaces.clear()
            return removed

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None
