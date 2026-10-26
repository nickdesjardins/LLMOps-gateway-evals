"""Retrieval regression check: deterministic, no model, no key, runs on every pull request.

For every answerable golden case the real hybrid retriever (the real embedding model and BM25)
fetches the top ``k`` chunks, which are compared with the chunks the case names as holding the
answer. Four standard measures, averaged over the cases:

* **hit rate@k**: the share of cases with at least one expected chunk in the top k;
* **recall@k**: the share of expected chunks found in the top k;
* **MRR**: the mean of 1 / rank of the first expected chunk (0 when none is found);
* **context precision@k** (average precision): the precision at each rank where an expected
  chunk appears, averaged over the expected chunks. The same quantity DeepEval's and Ragas'
  context precision estimate with a judge, computed here from labels.

A change to the chunker, the corpus, the embedding model or the fusion shows up here first,
before any money is spent on generating and judging answers.

    python -m evals.retrieval_check            # writes evals/out/retrieval.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.config import Settings
from app.embeddings import build_embedder
from app.rag.corpus import load_corpus
from app.rag.retriever import HybridRetriever
from evals.schema import OUT_DIR, load_golden, write_json


def average_precision(ranked: list[str], expected: set[str]) -> float:
    found, total = 0, 0.0
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in expected:
            found += 1
            total += found / rank
    return total / len(expected) if expected else 0.0


def score_case(ranked: list[str], expected: set[str]) -> dict[str, float]:
    hits = [rank for rank, cid in enumerate(ranked, start=1) if cid in expected]
    return {
        "hit": 1.0 if hits else 0.0,
        "recall": len(set(ranked) & expected) / len(expected),
        "reciprocal_rank": 1.0 / hits[0] if hits else 0.0,
        "average_precision": average_precision(ranked, expected),
    }


def run(k: int, embedding_provider: str | None = None) -> dict[str, object]:
    settings = Settings(
        _env_file=None, **({"embedding_provider": embedding_provider} if embedding_provider else {})
    )
    corpus = load_corpus()
    cases = [c for c in load_golden(corpus=corpus) if c.expected_behavior == "answer"]
    embedder = build_embedder(
        settings.embedding_provider,
        settings.fastembed_model,
        settings.fastembed_cache_dir,
        settings.fastembed_threads,
    )
    retriever = HybridRetriever(corpus, embedder)
    rows = []
    for case in cases:
        ranked = [r.chunk.chunk_id for r in retriever.search_sync(case.question, k)]
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "expected": case.reference_context,
                "retrieved": ranked,
                **score_case(ranked, set(case.reference_context)),
            }
        )
    n = len(rows)
    summary = {
        "k": k,
        "cases": n,
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 4),
        "recall": round(sum(r["recall"] for r in rows) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
        "context_precision": round(sum(r["average_precision"] for r in rows) / n, 4),
    }
    return {
        "engine": "retrieval",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder": embedder.id,
        "corpus_version": corpus.fingerprint,
        "summary": summary,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic retrieval metrics over the golden set.")
    parser.add_argument("--k", type=int, default=Settings(_env_file=None).retrieval_top_k)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "retrieval.json")
    parser.add_argument("--hash", action="store_true", help="the hash embedder (no model download)")
    args = parser.parse_args()
    report = run(args.k, "hash" if args.hash else None)
    write_json(args.out, report)
    s = report["summary"]
    print(
        f"retrieval@{s['k']}: hit rate {s['hit_rate']:.3f}, recall {s['recall']:.3f}, MRR {s['mrr']:.3f}, context precision {s['context_precision']:.3f} over {s['cases']} cases"
    )  # type: ignore[index]
    misses = [r["id"] for r in report["cases"] if not r["hit"]]  # type: ignore[union-attr]
    if misses:
        print(f"no expected chunk in the top {args.k}: {', '.join(misses)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
