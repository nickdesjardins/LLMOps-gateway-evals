"""Score the saved answers with Ragas: faithfulness, answer relevancy, context precision.

A second, independent judge of the same answers. Ragas pins the openai SDK below version 2 and
brings LangChain with it, so it cannot share an environment with the service (openai 3). This
script therefore imports nothing from ``app/`` or ``evals/``: it reads ``answers.jsonl`` as plain
JSON and runs in its own environment (``requirements-ragas.txt``), in its own CI job.

* ``Faithfulness``: the share of the answer's claims supported by the retrieved passages.
* ``AnswerRelevancy``: questions are generated back from the answer and compared with the real
  question by embedding similarity (the same local bge-small model the gateway uses).
* ``ContextPrecisionWithReference``: whether the passages useful for the reference answer are
  ranked first.

The judge is any OpenAI-compatible endpoint: OpenRouter's free models by default
(``RAGAS_JUDGE_MODELS``, ``OPENROUTER_API_KEY``). Requests are paced
(``RAGAS_MIN_INTERVAL_S``, 3.2 s) by an HTTP hook, since Ragas sends several per metric.

    python evals/score_ragas.py                # reads evals/out/answers.jsonl, writes ragas.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

OUT_DIR = Path(__file__).resolve().parent / "out"
DEFAULT_MODELS = "nvidia/nemotron-3-super-120b-a12b:free,google/gemma-4-31b-it:free"


class Pacer:
    """At least ``interval`` seconds between two requests, as an httpx request hook."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last = 0.0
        self.count = 0
        self.lock = asyncio.Lock()

    async def __call__(self, request: Any) -> None:
        async with self.lock:
            wait = self.last + self.interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.monotonic()
            self.count += 1


def build_embeddings() -> Any:
    """bge-small through fastembed, wrapped as a Ragas embedding (no API, no key)."""
    from fastembed import TextEmbedding
    from ragas.embeddings.base import BaseRagasEmbedding

    class FastEmbedRagas(BaseRagasEmbedding):
        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model = TextEmbedding(
                model_name=model_name, cache_dir=os.environ.get("FASTEMBED_CACHE_DIR") or None
            )

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return [float(x) for x in next(iter(self.model.embed([text])))]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[float(x) for x in v] for v in self.model.embed(texts)]

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts)

    return FastEmbedRagas(os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))


def build_metrics(model: str, pacer: Pacer, embeddings: Any) -> dict[str, Any]:
    import httpx
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, Faithfulness

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ.get("RAGAS_BASE_URL", "https://openrouter.ai/api/v1"),
        max_retries=1,
        timeout=90,
        http_client=httpx.AsyncClient(event_hooks={"request": [pacer]}, timeout=90),
    )
    llm = llm_factory(model, client=client, max_tokens=2048, temperature=0.0)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        "context_precision": ContextPrecisionWithReference(llm=llm),
    }


async def score_record(
    record: dict[str, Any], chains: list[dict[str, Any]]
) -> tuple[dict[str, float | None], dict[str, str]]:
    calls = {
        "faithfulness": {
            "user_input": record["question"],
            "response": record["answer"],
            "retrieved_contexts": record["retrieved_contexts"],
        },
        "answer_relevancy": {"user_input": record["question"], "response": record["answer"]},
        "context_precision": {
            "user_input": record["question"],
            "reference": record["reference_answer"],
            "retrieved_contexts": record["retrieved_contexts"],
        },
    }
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name, kwargs in calls.items():
        scores[name] = None
        for metrics in chains:  # the next judge model when one fails
            try:
                result = await metrics[name].ascore(**kwargs)
                value = getattr(result, "value", result)
                scores[name] = None if value is None else round(float(value), 4)
                errors.pop(name, None)
                break
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return scores, errors


async def main_async(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set: the Ragas judge cannot run.", file=sys.stderr)
        return 2
    records = [
        json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    pacer = Pacer(float(os.environ.get("RAGAS_MIN_INTERVAL_S", "3.2")))
    embeddings = build_embeddings()
    models = [m.strip() for m in os.environ.get("RAGAS_JUDGE_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
    chains = [build_metrics(model, pacer, embeddings) for model in models]
    rows = []
    for number, record in enumerate(records, start=1):
        row: dict[str, Any] = {
            "id": record["id"],
            "category": record["category"],
            "scores": {},
            "metric_errors": {},
        }
        if record.get("error") is None and record["expected_behavior"] == "answer" and not record["refused"]:
            row["scores"], row["metric_errors"] = await score_record(record, chains)
        rows.append(row)
        print(f"[{number}/{len(records)}] {record['id']}: {row['scores'] or '-'}", flush=True)

    def mean(name: str) -> float | None:
        values = [r["scores"][name] for r in rows if r["scores"].get(name) is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        "cases_scored": sum(1 for r in rows if r["scores"]),
        "metric_failures": sum(len(r["metric_errors"]) for r in rows),
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
    }
    report = {
        "engine": "ragas",
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge": {"models": models, "requests": pacer.count},
        "summary": summary,
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "requests": pacer.count}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ragas metrics over saved answers.")
    parser.add_argument("--answers", type=Path, default=OUT_DIR / "answers.jsonl")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "ragas.json")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
