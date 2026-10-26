"""Semantic cache: exact and nearest-neighbour lookups in front of every model call."""

from app.cache.base import CacheEntry, CacheUnavailable, Neighbor, VectorStore
from app.cache.memory import MemoryVectorStore
from app.cache.semantic import LookupResult, SemanticCache, namespace_for, normalize_question

__all__ = [
    "CacheEntry",
    "CacheUnavailable",
    "LookupResult",
    "MemoryVectorStore",
    "Neighbor",
    "SemanticCache",
    "VectorStore",
    "namespace_for",
    "normalize_question",
]
