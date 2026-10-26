"""Hybrid retrieval over the corpus: dense vectors and BM25, fused by reciprocal rank.

Dense search finds passages that mean the same thing in other words ("time off" for "leave");
BM25 finds the exact tokens a dense model blurs ("P2", "W-9", "net 45"). Each ranks every
chunk, and Reciprocal Rank Fusion merges the two rankings without having to compare a cosine
with a BM25 score::

    rrf(chunk) = 1 / (k + rank_dense) + 1 / (k + rank_bm25),  k = 60

The corpus is small (tens of chunks), so both searches are exact and in memory: the chunk
vectors are computed once at start-up, and a query costs one embedding and two sorts.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.embeddings import Embedder
from app.rag.corpus import Chunk, Corpus

RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in is it of on or our that the "
    "this to was what when where which who why will with you your we my me".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 (k1 = 1.5, b = 0.75) over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        frequency: Counter[str] = Counter()
        for counts in self.docs:
            frequency.update(counts.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float64)
        for i, counts in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg, 1e-9))
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    out[i] += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / (tf + norm)
        return out


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    rank: int
    rrf: float
    dense: float
    bm25: float

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "rank": self.rank,
            "rrf": round(self.rrf, 5),
            "dense": round(self.dense, 4),
            "bm25": round(self.bm25, 3),
        }


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 for the best score; ties keep corpus order."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


class HybridRetriever:
    def __init__(self, corpus: Corpus, embedder: Embedder) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.chunks = corpus.chunks
        texts = [c.indexed_text for c in self.chunks]
        self.matrix = embedder.embed(texts)
        self.bm25 = BM25(texts)

    async def search(self, query: str, top_k: int = 4) -> list[Retrieved]:
        vector = (await self.embedder.aembed([query]))[0]
        return self._fuse(query, vector, top_k)

    def search_sync(self, query: str, top_k: int = 4) -> list[Retrieved]:
        return self._fuse(query, self.embedder.embed([query])[0], top_k)

    def _fuse(self, query: str, vector: np.ndarray, top_k: int) -> list[Retrieved]:
        dense = self.matrix @ vector
        lexical = self.bm25.scores(query)
        fused = 1.0 / (RRF_K + _ranks(dense)) + 1.0 / (RRF_K + _ranks(lexical))
        order = np.argsort(-fused, kind="stable")[:top_k]
        return [
            Retrieved(
                chunk=self.chunks[i],
                rank=rank,
                rrf=float(fused[i]),
                dense=float(dense[i]),
                bm25=float(lexical[i]),
            )
            for rank, i in enumerate(order, start=1)
        ]
