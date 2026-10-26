"""The semantic cache: when is a new question "the same" as one already answered?

A lookup runs two checks, cheapest first:

1. **Exact.** The question is normalized (case folded, whitespace collapsed, trailing ``?!.``
   dropped) and hashed with its namespace. A stored hash is a hit with similarity 1.0, found
   with one key lookup and no embedding at all.
2. **Semantic.** Otherwise the question is embedded and the three nearest stored questions of
   the same namespace are fetched. The first one at or above the threshold (0.96 by default)
   that the contrast guard accepts (:mod:`app.cache.contrast`: no contradicting numbers, codes
   or opposite terms) is a hit. Anything else is a miss, and the miss still reports the nearest
   similarity and what the guard blocked, which is what the threshold trade-off is made of.

The **namespace** keeps answers apart that must never be served for each other: it is a hash
of the kind of request (a RAG question, a chat completion with a given system prompt), the
prompt version, the corpus fingerprint and the embedding model. Changing a prompt or a document
therefore starts an empty cache instead of serving answers produced under the old one.

What is stored is decided by the caller (:mod:`app.rag.pipeline`, :mod:`app.gateway`): only a
complete, validated answer to a request the guard did not flag.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import numpy as np

from app.cache.base import CacheEntry, VectorStore
from app.cache.contrast import contrast_conflict
from app.embeddings import Embedder

LookupOutcome = Literal["hit", "miss", "bypass", "off"]
_SPACES = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """The form that is hashed and embedded: case folded, single spaces, no trailing ``?!.``."""
    return _SPACES.sub(" ", text.casefold()).strip().rstrip("?!. ").strip()


def namespace_for(*parts: str) -> str:
    """A short alphanumeric namespace (Redis TAG values must not need escaping)."""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class LookupResult:
    outcome: LookupOutcome
    #: "exact" or "semantic" on a hit, None otherwise.
    match: Literal["exact", "semantic"] | None = None
    #: Similarity of the nearest stored question (1.0 on an exact hit), also on a miss.
    similarity: float | None = None
    threshold: float | None = None
    entry: CacheEntry | None = None
    nearest_question: str | None = None
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0
    #: The question's vector, kept so the write after a miss does not embed it again.
    vector: np.ndarray | None = None
    normalized: str = ""
    backend: str = ""
    #: Neighbours above the threshold that the contrast guard refused: (question, similarity, why).
    blocked: tuple[tuple[str, float, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "match": self.match,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "threshold": self.threshold,
            "nearest_question": self.nearest_question,
            "embed_ms": round(self.embed_ms, 2),
            "search_ms": round(self.search_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "backend": self.backend,
            "blocked": [
                {"question": q, "similarity": round(sim, 4), "reason": why} for q, sim, why in self.blocked
            ],
        }


class SemanticCache:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        threshold: float = 0.96,
        ttl_s: int = 86_400,
        contrast_guard: bool = True,
        clock: Callable[[], float] = time.time,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.contrast_guard = contrast_guard
        self._clock = clock
        self._timer = timer

    @staticmethod
    def exact_key(namespace: str, normalized: str) -> str:
        return hashlib.sha256(f"{namespace}\x00{normalized}".encode()).hexdigest()

    async def lookup(self, namespace: str, question: str) -> LookupResult:
        started = self._timer()
        normalized = normalize_question(question)
        backend = self.store.name
        exact = await self.store.get_exact(namespace, self.exact_key(namespace, normalized))
        if exact is not None:
            elapsed = (self._timer() - started) * 1000
            return LookupResult(
                outcome="hit",
                match="exact",
                similarity=1.0,
                threshold=self.threshold,
                entry=exact,
                nearest_question=exact.question,
                search_ms=elapsed,
                total_ms=elapsed,
                normalized=normalized,
                backend=backend,
            )
        embed_started = self._timer()
        vector = (await self.embedder.aembed([normalized]))[0]
        embed_ms = (self._timer() - embed_started) * 1000
        search_started = self._timer()
        neighbors = await self.store.nearest(namespace, vector, k=3)
        search_ms = (self._timer() - search_started) * 1000
        blocked: list[tuple[str, float, str]] = []
        for neighbor in neighbors:
            if neighbor.similarity < self.threshold:
                break
            reason = contrast_conflict(normalized, neighbor.entry.question) if self.contrast_guard else None
            if reason:
                blocked.append((neighbor.entry.question, neighbor.similarity, reason))
                continue
            return LookupResult(
                outcome="hit",
                match="semantic",
                similarity=neighbor.similarity,
                threshold=self.threshold,
                entry=neighbor.entry,
                nearest_question=neighbor.entry.question,
                embed_ms=embed_ms,
                search_ms=search_ms,
                total_ms=(self._timer() - started) * 1000,
                vector=vector,
                normalized=normalized,
                backend=backend,
                blocked=tuple(blocked),
            )
        best = neighbors[0] if neighbors else None
        return LookupResult(
            outcome="miss",
            similarity=None if best is None else best.similarity,
            threshold=self.threshold,
            nearest_question=None if best is None else best.entry.question,
            embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=(self._timer() - started) * 1000,
            vector=vector,
            normalized=normalized,
            backend=backend,
            blocked=tuple(blocked),
        )

    async def store_answer(
        self,
        namespace: str,
        question: str,
        payload: dict[str, Any],
        *,
        reference_cost_usd: Decimal,
        actual_cost_usd: Decimal,
        vector: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a validated answer. Returns False when the store refused it."""
        normalized = normalize_question(question)
        if vector is None:
            vector = (await self.embedder.aembed([normalized]))[0]
        now = self._clock()
        entry = CacheEntry(
            entry_id=secrets.token_hex(8),
            namespace=namespace,
            exact_key=self.exact_key(namespace, normalized),
            question=normalized,
            payload=payload,
            created_at=now,
            expires_at=now + self.ttl_s,
            reference_cost_usd=str(reference_cost_usd),
            actual_cost_usd=str(actual_cost_usd),
            metadata=metadata or {},
        )
        return await self.store.put(entry, vector)

    async def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of two questions as the cache sees them (used by the playground)."""
        vectors = await self.embedder.aembed([normalize_question(a), normalize_question(b)])
        return float(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))
