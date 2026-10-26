"""Replay a help-desk workload through the real pipeline and measure the semantic cache.

The workload is synthetic and deterministic: ``REQUESTS`` questions drawn from the intents of
``evals/cache_intents.json`` with Zipf-like popularity (a few questions are asked often, most
rarely, as on a real help desk), each time in one of the intent's 3 or 4 phrasings, chosen at
random with a fixed seed. Every question goes through the real :class:`RagPipeline`: guard,
cache (the real embedding model, the in-process store), retrieval, router, and on a miss the
offline provider, which answers with the retrieved passage and calls no model.

What is measured, per threshold:

* the hit rate, split into exact and semantic hits;
* **false hits**: a hit whose stored answer belongs to another intent (a wrong answer served);
* cache lookup time on hits, and the end-to-end latency of hits and misses (offline misses
  have no model latency, so the miss figure is the gateway's own overhead);
* tokens and **cost at list price**: each miss is priced with the tier's reference model
  (Claude Haiku 5.5 or Claude Opus 5.5) on the real prompt (system prefix, four passages, the
  question). The offline answer is a quote of two sentences, shorter than a model's, so output
  tokens are an underestimate; input tokens, which dominate the bill here, are exact estimates
  of the real prompt.

No key, no network, no model call.

    python -m evals.replay                     # writes site/recorded/replay.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from decimal import Decimal
from pathlib import Path

from app.cache.semantic import normalize_question
from app.components import build_components
from app.config import Settings
from app.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from app.rag.pipeline import AskOptions

ROOT = Path(__file__).resolve().parent.parent
INTENTS = ROOT / "evals" / "cache_intents.json"
OUT = ROOT / "site" / "recorded" / "replay.json"
REQUESTS = 250
SEED = 7
ZIPF_S = 1.1
THRESHOLDS = (0.96, 0.92, 0.88)


def workload(n: int = REQUESTS, seed: int = SEED) -> list[tuple[str, str]]:
    """(intent id, phrasing) pairs; the same seed always gives the same list."""
    data = json.loads(INTENTS.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    intents = list(data["intents"])
    rng.shuffle(intents)  # popularity rank independent of file order
    weights = [1 / (rank**ZIPF_S) for rank in range(1, len(intents) + 1)]
    picks = rng.choices(intents, weights=weights, k=n)
    return [(intent["id"], rng.choice(intent["phrasings"])) for intent in picks]


async def run_once(
    threshold: float, embedder: Embedder, items: list[tuple[str, str]], redis_url: str | None = None
) -> dict[str, object]:
    settings = Settings(
        _env_file=None,
        llm_provider="offline",
        cache_backend="redis" if redis_url else "memory",
        redis_url=redis_url,
        # A fresh index and prefix per run: no answer of an earlier run can be served.
        redis_index_name=f"replay-{int(threshold * 100)}-{random.randrange(10**6)}",
        redis_key_prefix=f"replay:{random.randrange(10**6)}:",
        cache_similarity_threshold=threshold,
        embedding_provider="hash" if isinstance(embedder, HashEmbedder) else "fastembed",
    )
    # Every run starts cold: the embedder's memo of recent texts would otherwise carry over.
    memo = getattr(embedder, "_memo", None)
    if memo is not None:
        memo.clear()
    components = await build_components(settings, embedder=embedder, sinks=[])
    owner: dict[str, str] = {}  # stored question -> intent that produced it
    exact = semantic = misses = false_hits = 0
    exact_lookup: list[float] = []
    semantic_lookup: list[float] = []
    hit_latency: list[float] = []
    miss_latency: list[float] = []
    blocked = 0
    for intent, question in items:
        payload = await components.pipeline.answer(question, AskOptions(client="replay"))
        cache = payload["cache"]
        blocked += len(cache.get("blocked") or [])
        if cache["outcome"] == "hit":
            exact += cache["match"] == "exact"
            semantic += cache["match"] == "semantic"
            if owner.get(cache["nearest_question"]) != intent:
                false_hits += 1
            (exact_lookup if cache["match"] == "exact" else semantic_lookup).append(cache["total_ms"])
            hit_latency.append(payload["latency_ms"])
        else:
            misses += 1
            miss_latency.append(payload["latency_ms"])
            owner[normalize_question(question)] = intent
    snapshot = components.metrics.snapshot()
    if redis_url and components.store is not None:
        await components.store.clear()
        await components.store.client.execute_command("FT.DROPINDEX", components.store.index)  # type: ignore[attr-defined]
    await components.close()
    hits = exact + semantic
    reference = Decimal(snapshot["cost"]["reference_usd"])
    avoided = Decimal(snapshot["cost"]["avoided_usd"])

    def p(values: list[float], q: int) -> float | None:
        if not values:
            return None
        return (
            round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 2)
            if len(values) > 1
            else round(values[0], 2)
        )

    return {
        "threshold": threshold,
        "requests": len(items),
        "hits": hits,
        "exact_hits": exact,
        "semantic_hits": semantic,
        "misses": misses,
        "hit_rate": round(hits / len(items), 4),
        "false_hits": false_hits,
        "blocked_by_guard": blocked,
        "exact_lookup_ms": {
            "p50": p(exact_lookup, 50),
            "p95": p(exact_lookup, 95),
            "p99": p(exact_lookup, 99),
        },
        "semantic_lookup_ms": {
            "p50": p(semantic_lookup, 50),
            "p95": p(semantic_lookup, 95),
            "p99": p(semantic_lookup, 99),
        },
        "hit_latency_ms": {"p50": p(hit_latency, 50), "p95": p(hit_latency, 95), "p99": p(hit_latency, 99)},
        "miss_overhead_ms": {
            "p50": p(miss_latency, 50),
            "p95": p(miss_latency, 95),
            "p99": p(miss_latency, 99),
        },
        "tokens": snapshot["tokens"],
        "reference_cost_usd": str(reference),
        "avoided_cost_usd": str(avoided),
        "cost_without_cache_usd": str(reference + avoided),
        "cost_reduction": snapshot["cost"]["cost_reduction"],
        "tiers": snapshot["tiers"],
        "backend": "redis" if redis_url else "memory",
    }


async def main_async(args: argparse.Namespace) -> None:
    settings = Settings(_env_file=None)
    embedder: Embedder = (
        HashEmbedder()
        if args.hash
        else FastEmbedEmbedder(settings.fastembed_model, cache_dir=settings.fastembed_cache_dir)
    )
    items = workload(args.requests, args.seed)
    distinct = len({q for _, q in items})
    runs = [await run_once(t, embedder, items, args.redis_url) for t in args.thresholds]
    record = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": embedder.id,
        "workload": {
            "requests": len(items),
            "intents": len({i for i, _ in items}),
            "distinct_questions": distinct,
            "seed": args.seed,
            "zipf_s": ZIPF_S,
            "provider": "offline (no model called; misses priced at the reference list price)",
            "cache_backend": "redis" if args.redis_url else "memory",
        },
        "reference_models": {
            "light": settings.reference_model_light,
            "heavy": settings.reference_model_heavy,
        },
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for run in runs:
        print(
            f"threshold {run['threshold']}: {run['hits']}/{run['requests']} hits "
            f"({run['exact_hits']} exact, {run['semantic_hits']} semantic), {run['false_hits']} false, "
            f"cost reduction {run['cost_reduction']:.1%}, lookup p50 exact {run['exact_lookup_ms']['p50']} ms, "
            f"semantic {run['semantic_lookup_ms']['p50']} ms (p99 {run['semantic_lookup_ms']['p99']})"
        )
    print(f"{distinct} distinct questions in {len(items)} requests. Wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a help-desk workload through the cache.")
    parser.add_argument("--requests", type=int, default=REQUESTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    parser.add_argument("--hash", action="store_true", help="the hash embedder (no model download)")
    parser.add_argument("--redis-url", help="replay against this Redis (8, or Stack) instead of memory")
    parser.add_argument("--out", type=Path, default=OUT)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
