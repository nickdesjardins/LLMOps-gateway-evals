"""The router's decisions on the golden questions, with the real retriever, for the project page.

For every question of the golden set: the retrieval spread (how many documents the strongest
passages come from), the complexity score, the named signals and the tier. Deterministic, no
model call, no key.

    python scripts/route_examples.py            # writes site/recorded/routing.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.embeddings import build_embedder  # noqa: E402
from app.rag.corpus import load_corpus  # noqa: E402
from app.rag.pipeline import RELEVANCE_FLOOR, STRONG_MARGIN, documents_in_play  # noqa: E402
from app.rag.retriever import HybridRetriever  # noqa: E402
from app.routing.complexity import ComplexityRouter  # noqa: E402
from evals.schema import load_golden  # noqa: E402

OUT = ROOT / "site" / "recorded" / "routing.json"


def main() -> None:
    settings = Settings(_env_file=None)
    corpus = load_corpus()
    embedder = build_embedder(
        settings.embedding_provider,
        settings.fastembed_model,
        settings.fastembed_cache_dir,
        settings.fastembed_threads,
    )
    retriever = HybridRetriever(corpus, embedder)
    router = ComplexityRouter(settings.router_heavy_threshold)
    rows = []
    for case in load_golden(corpus=corpus):
        retrieved = retriever.search_sync(case.question, settings.retrieval_top_k)
        spread = documents_in_play(retrieved)
        decision = router.route(case.question, documents_in_play=spread)
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "documents_in_play": spread,
                "best_dense": round(max(r.dense for r in retrieved), 4),
                **decision.as_dict(),
            }
        )
    tiers = {tier: sum(1 for r in rows if r["tier"] == tier) for tier in ("light", "heavy")}
    record = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": embedder.id,
        "threshold": settings.router_heavy_threshold,
        "strong_margin": STRONG_MARGIN,
        "relevance_floor": RELEVANCE_FLOOR,
        "tiers": tiers,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(
        f"{len(rows)} questions: {tiers['light']} light, {tiers['heavy']} heavy. Wrote {OUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
