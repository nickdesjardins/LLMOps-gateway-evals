"""The storage side of the semantic cache: entries, and the two stores that hold them.

A store answers two questions: "is this exact text cached?" (a key lookup, no embedding needed)
and "what is the nearest cached question to this vector, within this namespace?" (a vector
search). :class:`app.cache.semantic.SemanticCache` decides what a hit is; the stores only
store and search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class CacheUnavailable(RuntimeError):
    """The store cannot be used (no vector search on the server, connection refused)."""


@dataclass(frozen=True)
class CacheEntry:
    """One stored answer and what it cost to produce."""

    entry_id: str
    namespace: str
    exact_key: str
    #: The question as it was cached (normalized), shown to the client on a hit.
    question: str
    #: The response to replay: the same JSON body the miss returned.
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    #: What the miss cost at the list price of the reference model (a decimal string). Every
    #: hit avoids this much; the telemetry adds it up as "cost avoided".
    reference_cost_usd: str = "0"
    actual_cost_usd: str = "0"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Neighbor:
    entry: CacheEntry
    similarity: float


class VectorStore(Protocol):
    """A namespaced store of entries with one vector each."""

    #: "memory" or "redis": reported by /health and in every trace.
    name: str

    async def get_exact(self, namespace: str, exact_key: str) -> CacheEntry | None: ...

    async def nearest(self, namespace: str, vector: np.ndarray, k: int = 3) -> list[Neighbor]:
        """Up to ``k`` live entries of the namespace, most similar first."""
        ...

    async def put(self, entry: CacheEntry, vector: np.ndarray) -> bool:
        """Store the entry; False when the store refused it (full, or unreachable)."""
        ...

    async def count(self) -> int: ...

    async def clear(self) -> int: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...
