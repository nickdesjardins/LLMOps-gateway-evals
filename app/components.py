"""Everything the routes need, built once at start-up from the settings.

The order matters and is logged: the corpus, then the embedding model (loaded from the image on
the host, from the fastembed cache on a laptop), then the chunk vectors, then the cache store
(Redis when ``REDIS_URL`` is set and it has vector search, memory otherwise), then the model
provider, then telemetry. A missing provider key does not stop the service: grounded questions
already in the cache are still answered, and the rest get 503 until the key is set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app import __version__
from app.budget import DemoBudget
from app.cache.base import CacheUnavailable, VectorStore
from app.cache.memory import MemoryVectorStore
from app.cache.semantic import SemanticCache
from app.config import Settings
from app.embeddings import Embedder, build_embedder
from app.errors import ConfigurationError, NotConfigured
from app.gateway import ChatGateway
from app.models.chooser import ModelChooser
from app.pricing import PriceBook
from app.providers import Provider, build_provider
from app.rag.corpus import Corpus, load_corpus
from app.rag.pipeline import RagPipeline
from app.rag.retriever import HybridRetriever
from app.routing.cascade import ModelCascade
from app.routing.complexity import ComplexityRouter
from app.telemetry.metrics import MetricsRecorder
from app.telemetry.tracer import Sink, Telemetry

logger = logging.getLogger(__name__)


class UnconfiguredProvider:
    """Stands in when the provider has no key: every call is a clean 503."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    async def complete(self, model: str, request: Any) -> Any:
        raise NotConfigured(f"The model provider is not configured: {self.reason}.")

    async def stream(self, model: str, request: Any) -> Any:
        raise NotConfigured(f"The model provider is not configured: {self.reason}.")
        yield  # pragma: no cover - makes this an async generator

    async def aclose(self) -> None:
        return None


@dataclass
class Components:
    settings: Settings
    corpus: Corpus
    embedder: Embedder
    retriever: HybridRetriever
    store: VectorStore | None
    cache: SemanticCache | None
    cache_note: str
    provider: Provider
    provider_ready: bool
    prices: PriceBook
    router: ComplexityRouter
    cascade: ModelCascade
    metrics: MetricsRecorder
    telemetry: Telemetry
    budget: DemoBudget
    pipeline: RagPipeline
    chat: ChatGateway
    chooser: ModelChooser

    async def close(self) -> None:
        await self.chooser.stop()
        self.telemetry.flush()
        for sink in self.telemetry.sinks:
            shutdown = getattr(sink, "shutdown", None)
            if shutdown is not None:
                shutdown()
        if self.store is not None:
            await self.store.close()
        await self.provider.aclose()


async def build_store(settings: Settings, dim: int) -> tuple[VectorStore | None, str]:
    backend = settings.resolved_cache_backend()
    if backend == "off":
        return None, "CACHE_BACKEND=off"
    if backend == "redis":
        if settings.redis_url is None:
            raise ConfigurationError("CACHE_BACKEND=redis needs REDIS_URL")
        from app.cache.redis_store import RedisVectorStore

        try:
            store = await RedisVectorStore.connect(
                settings.redis_url.get_secret_value(),
                dim=dim,
                index_name=settings.redis_index_name,
                key_prefix=settings.redis_key_prefix,
                max_entries=settings.cache_max_entries,
            )
            return store, "Redis vector index (HNSW, cosine)"
        except CacheUnavailable as exc:
            logger.warning("Redis cannot hold the cache (%s); using the in-process store", exc)
            note = f"memory (Redis unusable: {exc})"
        except Exception as exc:  # connection refused, DNS, TLS: the URL is never logged
            logger.warning("Redis did not answer (%s); using the in-process store", type(exc).__name__)
            note = f"memory (Redis did not answer: {type(exc).__name__})"
        return MemoryVectorStore(dim, settings.cache_max_entries), note
    return MemoryVectorStore(dim, settings.cache_max_entries), "in-process store"


def build_sinks(settings: Settings) -> list[Sink]:
    if not settings.langfuse_enabled():
        return []
    from app.telemetry.langfuse_sink import LangfuseSink

    assert settings.langfuse_public_key is not None and settings.langfuse_secret_key is not None
    sink = LangfuseSink.create(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.environment,
        release=__version__,
        capture_content=settings.langfuse_capture_content,
    )
    logger.info(
        "Langfuse tracing on (%s, content %s)",
        settings.langfuse_base_url,
        "captured" if settings.langfuse_capture_content else "withheld",
    )
    return [sink]


async def build_components(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    provider: Provider | None = None,
    store: VectorStore | None = None,
    corpus: Corpus | None = None,
    sinks: list[Sink] | None = None,
) -> Components:
    """Build the service. The keyword arguments replace a part (the tests use them)."""
    corpus = corpus or load_corpus()
    logger.info(
        "Corpus: %d documents, %d chunks (fingerprint %s)",
        len(corpus.documents),
        len(corpus.chunks),
        corpus.fingerprint,
    )
    embedder = embedder or build_embedder(
        settings.embedding_provider,
        settings.fastembed_model,
        settings.fastembed_cache_dir,
        settings.fastembed_threads,
    )
    retriever = HybridRetriever(corpus, embedder)
    if store is not None:
        note = f"{store.name} (provided)"
    else:
        store, note = await build_store(settings, embedder.dim)
    cache = None
    if store is not None:
        cache = SemanticCache(
            store,
            embedder,
            threshold=settings.cache_similarity_threshold,
            ttl_s=settings.cache_ttl_s,
            contrast_guard=settings.cache_contrast_guard,
        )
    provider_ready = True
    if provider is None:
        try:
            provider = build_provider(settings)
        except ConfigurationError as exc:
            logger.warning("%s; model calls answer 503 until it is set", exc)
            provider = UnconfiguredProvider(settings.llm_provider, str(exc))
            provider_ready = False
    prices = PriceBook(settings.extra_prices())
    budget = DemoBudget(settings)
    chooser = ModelChooser(settings, provider, budget=budget)
    await chooser.start()
    cascade = ModelCascade(
        provider,
        chooser.chains,  # the same lists: a repair is seen by the next request
        prices=prices,
        reference_models={"light": settings.reference_model_light, "heavy": settings.reference_model_heavy},
        cross_tier=settings.cross_tier_fallback,
        deadline_s=settings.request_deadline_s,
        model_timeout_s=settings.model_timeout_s,
        listener=chooser.note,
    )
    router = ComplexityRouter(settings.router_heavy_threshold)
    metrics = MetricsRecorder()
    telemetry = Telemetry(
        sinks=build_sinks(settings) if sinks is None else sinks, metrics=metrics, keep=settings.traces_kept
    )
    pipeline = RagPipeline(
        settings=settings,
        corpus=corpus,
        retriever=retriever,
        cache=cache,
        router=router,
        cascade=cascade,
        telemetry=telemetry,
        budget=budget,
    )
    chat = ChatGateway(
        settings=settings,
        cache=cache,
        router=router,
        cascade=cascade,
        telemetry=telemetry,
        budget=budget,
        embedder_id=embedder.id,
    )
    logger.info(
        "Cache: %s, threshold %.2f. Provider: %s (light %s, heavy %s)",
        note,
        settings.cache_similarity_threshold,
        settings.llm_provider,
        ", ".join(chooser.chains["light"]),
        ", ".join(chooser.chains["heavy"]),
    )
    return Components(
        settings=settings,
        corpus=corpus,
        embedder=embedder,
        retriever=retriever,
        store=store,
        cache=cache,
        cache_note=note,
        provider=provider,
        provider_ready=provider_ready,
        prices=prices,
        router=router,
        cascade=cascade,
        metrics=metrics,
        telemetry=telemetry,
        budget=budget,
        pipeline=pipeline,
        chat=chat,
        chooser=chooser,
    )
