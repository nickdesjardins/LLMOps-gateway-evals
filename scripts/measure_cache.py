"""Measure the threshold trade-off of the semantic cache with the real embedding model.

Every pair of phrasings of one intent (``evals/cache_intents.json``) should be a hit: same
question, same answer. Every pair across a "near miss" (domestic against international meal
limits, P1 against P2) must not be: the questions look alike and the answers differ. For each
threshold from 0.80 to 0.99 this counts how many true pairs would hit and how many near-miss
pairs would be served the wrong answer, with and without the contrast guard.

No model is called and no key is needed: only the embedding model, which runs locally.

    python scripts/measure_cache.py              # writes site/recorded/cache-threshold.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cache.contrast import contrast_conflict  # noqa: E402
from app.cache.semantic import normalize_question  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402

INTENTS = ROOT / "evals" / "cache_intents.json"
OUT = ROOT / "site" / "recorded" / "cache-threshold.json"
THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    settings = Settings(_env_file=None)
    embedder = FastEmbedEmbedder(settings.fastembed_model, cache_dir=settings.fastembed_cache_dir)
    data = json.loads(INTENTS.read_text(encoding="utf-8"))
    intents = {i["id"]: [normalize_question(p) for p in i["phrasings"]] for i in data["intents"]}
    texts = [p for phrasings in intents.values() for p in phrasings]
    vectors = dict(zip(texts, embedder.embed(texts), strict=True))

    def pair(kind: str, label: str, a: str, b: str) -> dict[str, object]:
        return {
            "kind": kind,
            "label": label,
            "a": a,
            "b": b,
            "similarity": round(float(vectors[a] @ vectors[b]), 4),
            "guard": contrast_conflict(a, b),
        }

    pairs = [pair("same", iid, a, b) for iid, ps in intents.items() for a, b in itertools.combinations(ps, 2)]
    pairs += [
        pair("near_miss", f"{x} / {y}", a, b)
        for x, y in data["near_misses"]
        for a in intents[x]
        for b in intents[y]
    ]
    same = [p for p in pairs if p["kind"] == "same"]
    near = [p for p in pairs if p["kind"] == "near_miss"]

    def rate(items: list[dict[str, object]], threshold: float, guarded: bool) -> float:
        hits = [p for p in items if p["similarity"] >= threshold and not (guarded and p["guard"])]  # type: ignore[operator]
        return round(len(hits) / len(items), 4)

    sweep = [
        {
            "threshold": t,
            "true_hit_rate": rate(same, t, guarded=True),
            "false_hit_rate": rate(near, t, guarded=True),
            "true_hit_rate_unguarded": rate(same, t, guarded=False),
            "false_hit_rate_unguarded": rate(near, t, guarded=False),
        }
        for t in THRESHOLDS
    ]

    probe = normalize_question("How many vacation days can a part-time employee take?")
    timings = []
    for _ in range(50):
        embedder._memo.clear()
        started = time.perf_counter()
        embedder.embed([probe])
        timings.append((time.perf_counter() - started) * 1000)

    record = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": embedder.id,
        "dimensions": embedder.dim,
        "intents": len(intents),
        "phrasings": len(texts),
        "same_pairs": len(same),
        "near_miss_pairs": len(near),
        "same_similarity": {
            "min": min(p["similarity"] for p in same),  # type: ignore[type-var]
            "median": round(statistics.median(p["similarity"] for p in same), 4),  # type: ignore[arg-type]
            "max": max(p["similarity"] for p in same),  # type: ignore[type-var]
        },
        "near_miss_similarity": {
            "median": round(statistics.median(p["similarity"] for p in near), 4),  # type: ignore[arg-type]
            "max": max(p["similarity"] for p in near),  # type: ignore[type-var]
        },
        "guard_blocks_same": sum(1 for p in same if p["guard"]),
        "query_embedding_ms": {
            "p50": round(statistics.median(timings), 2),
            "max": round(max(timings), 2),
            "machine": f"{platform.system()} {platform.machine()}, {platform.python_version()}",
        },
        "default_threshold": settings.cache_similarity_threshold,
        "sweep": sweep,
        "pairs": sorted(pairs, key=lambda p: -p["similarity"]),  # type: ignore[operator]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    at = next(s for s in sweep if s["threshold"] == settings.cache_similarity_threshold)
    print(
        f"{len(same)} same pairs (median {record['same_similarity']['median']}), {len(near)} near-miss pairs "
        f"(max {record['near_miss_similarity']['max']}). At {at['threshold']}: true hits {at['true_hit_rate']:.1%}, "
        f"false hits {at['false_hit_rate']:.1%} with the guard ({at['false_hit_rate_unguarded']:.1%} without). "
        f"Wrote {args.out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
