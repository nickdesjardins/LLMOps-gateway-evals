"""Run the real RAG pipeline over the golden set and save what it answered.

Each case goes through the same :class:`RagPipeline` the service runs (guard, retrieval,
router, cascade, validation), with the cache bypassed so every answer is produced fresh by a
model. The settings come from the environment, exactly as for the service: on GitHub Actions,
``LLM_PROVIDER=openrouter`` and the repository secret ``OPENROUTER_API_KEY``.

Calls are paced (``--pace-s``, 3.5 s by default: OpenRouter's free tier allows 20 requests a
minute) and a case that fails on every model is retried twice before it is recorded as an
error. A recorded error is never silently dropped: the gate counts it.

    python -m evals.generate_answers           # writes evals/out/answers.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.components import build_components
from app.config import Settings
from app.errors import GatewayError
from app.rag.pipeline import AskOptions
from app.rag.prompts import PROMPT_VERSION
from evals.schema import OUT_DIR, AnswerRecord, load_golden, write_json


async def generate(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.provider_configured():
        print(
            f"LLM_PROVIDER={settings.llm_provider} has no API key: nothing can be generated.", file=sys.stderr
        )
        return 2
    components = await build_components(settings)
    cases = load_golden(corpus=components.corpus)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    records: list[AnswerRecord] = []
    errors = 0
    started = time.time()
    for number, case in enumerate(cases, start=1):
        payload = None
        error = None
        for attempt in range(3):
            try:
                payload = await components.pipeline.answer(
                    case.question, AskOptions(cache="bypass", client="eval", session_id="eval-run")
                )
                break
            except GatewayError as exc:
                error = f"{exc.error_type}: {exc.message}"
                await asyncio.sleep(args.pace_s * (attempt + 2))
        reference_contexts = [components.corpus.chunk(cid).text for cid in case.reference_context]  # type: ignore[union-attr]
        if payload is None:
            errors += 1
            record = AnswerRecord(
                id=case.id,
                category=case.category,
                expected_behavior=case.expected_behavior,
                question=case.question,
                answer="",
                refused=False,
                retrieved_ids=[],
                retrieved_contexts=[],
                reference_answer=case.reference_answer,
                reference_context_ids=case.reference_context,
                reference_contexts=reference_contexts,
                must_not_contain=case.must_not_contain,
                model=None,
                tier=None,
                error=error,
            )
        else:
            retrieved_ids = [s["chunk_id"] for s in payload["sources"]]
            record = AnswerRecord(
                id=case.id,
                category=case.category,
                expected_behavior=case.expected_behavior,
                question=case.question,
                answer=payload["answer"],
                refused=payload["refused"],
                retrieved_ids=retrieved_ids,
                retrieved_contexts=[components.corpus.chunk(cid).text for cid in retrieved_ids],  # type: ignore[union-attr]
                reference_answer=case.reference_answer,
                reference_context_ids=case.reference_context,
                reference_contexts=reference_contexts,
                must_not_contain=case.must_not_contain,
                model=payload["model"],
                tier=payload["tier"],
                fallback=payload["fallback"],
                latency_ms=payload["latency_ms"],
                cost_usd=payload["cost"]["actual_usd"],
                reference_cost_usd=payload["cost"]["reference_usd"],
            )
        records.append(record)
        state = "error" if record.error else ("refused" if record.refused else "answered")
        print(
            f"[{number}/{len(cases)}] {case.id} {state} ({record.model or '-'}, {record.tier or '-'})",
            flush=True,
        )
        await asyncio.sleep(args.pace_s)
    await components.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    write_json(
        args.out.with_name("run.json"),
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "seconds": round(time.time() - started, 1),
            "provider": settings.llm_provider,
            "chains": {"light": settings.chain("light"), "heavy": settings.chain("heavy")},
            "prompt_version": PROMPT_VERSION,
            "corpus_version": components.corpus.fingerprint,
            "embedder": components.embedder.id,
            "cases": len(records),
            "errors": errors,
            "metrics": components.metrics.snapshot(),
        },
    )
    print(json.dumps({"cases": len(records), "errors": errors, "out": str(args.out)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer every golden case with the live pipeline.")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "answers.jsonl")
    parser.add_argument("--pace-s", type=float, default=3.5)
    parser.add_argument("--only", help="comma-separated case ids")
    return asyncio.run(generate(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
