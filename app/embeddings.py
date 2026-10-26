"""Text embeddings for the semantic cache and the retriever.

Two embedders behind one small interface:

* :class:`FastEmbedEmbedder` runs ``BAAI/bge-small-en-v1.5`` (384 dimensions, quantized ONNX,
  about 67 MB) on the CPU through fastembed. No API call, no key, no per-token cost. The Docker
  image bakes the weights in, so a cold start on a free host never downloads them.
* :class:`HashEmbedder` hashes words and word pairs into the same number of dimensions. It
  knows nothing about meaning: two sentences are close only when they share words. It exists
  so the gateway runs where no model can be downloaded, and so the tests are deterministic.
  The 0.96 threshold is calibrated for the real model, not for this one.

Every vector is L2-normalized, so the dot product of two vectors is their cosine similarity.
Inference is CPU-bound, so callers use :meth:`aembed`, which runs it in a worker thread and
lets the event loop keep serving other requests (cache hits among them).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

DIMENSIONS = 384
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
#: Words that carry no topic. Dropping them keeps the hash embedder from calling "What is X?"
#: and "What is Y?" similar.
_STOP = frozenset(
    "a an and are as at be by can could do does for from how i if in is it me my of on or our "
    "please should so than that the their them then there these this to us was we what when "
    "where which who why will with would you your".split()
)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row (a zero row stays zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(Protocol):
    """What the cache and the retriever need from an embedding model."""

    #: Names the model, so vectors from different models never meet (cache namespaces use it).
    id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """One normalized float32 row per text."""
        ...

    async def aembed(self, texts: list[str]) -> np.ndarray:
        """Same as :meth:`embed`, off the event loop."""
        ...


class _BaseEmbedder:
    id = "base"
    dim = DIMENSIONS

    def __init__(self, memo_size: int = 512) -> None:
        # Exact repeats of a text (a visitor pressing the same sample twice) skip inference.
        self._memo: OrderedDict[str, np.ndarray] = OrderedDict()
        self._memo_size = memo_size
        self._memo_lock = threading.Lock()

    def _compute(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[np.ndarray | None] = []
        missing: list[str] = []
        with self._memo_lock:
            for text in texts:
                hit = self._memo.get(text)
                if hit is not None:
                    self._memo.move_to_end(text)
                elif text not in missing:
                    missing.append(text)
                rows.append(hit)
        if missing:
            computed = self._compute(missing)
            fresh = dict(zip(missing, computed, strict=True))
            with self._memo_lock:
                for text, vector in fresh.items():
                    self._memo[text] = vector
                    self._memo.move_to_end(text)
                while len(self._memo) > self._memo_size:
                    self._memo.popitem(last=False)
            rows = [row if row is not None else fresh[text] for row, text in zip(rows, texts, strict=True)]
        return np.vstack(rows).astype(np.float32)

    async def aembed(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embed, texts)


class HashEmbedder(_BaseEmbedder):
    """Deterministic signed feature hashing of words and adjacent word pairs."""

    id = "hash-v1"

    @staticmethod
    def _features(text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        # A crude stem: "policies" and "policy", "receipts" and "receipt" meet.
        stems = [re.sub(r"(?:ies|es|s)$", "", w) if len(w) > 4 else w for w in words]
        return stems + [f"{a}_{b}" for a, b in zip(stems, stems[1:], strict=False)]

    def _compute(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                weight = 0.5 if "_" in feature else 1.0
                matrix[row, value % self.dim] += weight if (value >> 32) & 1 else -weight
        return normalize_rows(matrix)


class FastEmbedEmbedder(_BaseEmbedder):
    """bge-small-en-v1.5 (or another fastembed model) on the CPU, loaded once."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        memo_size: int = 512,
    ) -> None:
        super().__init__(memo_size=memo_size)
        from fastembed import TextEmbedding  # imported here: the tests never load the model

        self.id = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)
        # One inference at a time: two at once would not be faster on one shared CPU and would
        # double the memory peak of a 512 MB instance.
        self._lock = threading.Lock()
        probe = self._compute(["probe"])
        self.dim = int(probe.shape[1])

    def _compute(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            vectors = list(self._model.embed(texts, batch_size=32))
        return normalize_rows(np.asarray(vectors, dtype=np.float32))


def build_embedder(provider: str, model_name: str, cache_dir: str | None, threads: int | None) -> Embedder:
    """The embedder the settings ask for."""
    if provider == "hash":
        return HashEmbedder()
    logger.info("Loading embedding model %s", model_name)
    return FastEmbedEmbedder(model_name, cache_dir=cache_dir, threads=threads)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two normalized vectors, clipped to [-1, 1]."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))
