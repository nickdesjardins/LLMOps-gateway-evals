"""Check the OpenRouter key and every model of both chains, before recording or deploying.

Sends one grounded question to each model of the light and the heavy chain (4 free requests with
the default chains), directly, without the cascade, and prints for each: answered or the failure
kind (with OpenRouter's own explanation when it gave one), latency, tokens, and whether the
answer was cut off at the tier's ceiling. A model that fails here makes every request fall over
to the next one: in automatic mode the service benches it and finds a replacement; with
``LIGHT_MODELS`` / ``HEAVY_MODELS`` set, replace it there (ids that end in ``:free`` from
openrouter.ai/models). Only a 401 means the key itself is wrong.

Reads the key from ``.env`` / ``.env.local``; prints nothing secret.

    python scripts/openrouter_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.providers import ChatRequest, ProviderError, build_provider  # noqa: E402
from app.rag.corpus import load_corpus  # noqa: E402
from app.rag.prompts import PromptSource, build_messages, citation_problem  # noqa: E402


async def main() -> int:
    settings = Settings()
    if settings.llm_provider != "openrouter" or settings.openrouter_api_key is None:
        print("Set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY in .env.local first.")
        return 2
    provider = build_provider(settings)
    chunk = load_corpus().chunk("leave-policy#vacation")
    assert chunk is not None
    messages = build_messages(
        "How many vacation days do full-time employees get each year?",
        [PromptSource("S1", chunk.chunk_id, chunk.title, chunk.text)],
    )
    failures = 0
    for tier in ("light", "heavy"):
        for model in settings.chain(tier):  # type: ignore[arg-type]
            request = ChatRequest(messages=messages, max_tokens=settings.max_output_tokens(tier), tier=tier)  # type: ignore[arg-type]
            started = time.perf_counter()
            try:
                result = await provider.complete(model, request)
            except ProviderError as exc:
                failures += 1
                said = f"; OpenRouter said: {exc.provider_message}" if exc.provider_message else ""
                print(f"{tier:5s} {model}: FAILED ({exc.kind}: {exc.detail}{said})")
                continue
            seconds = time.perf_counter() - started
            problem = citation_problem(result.text, 1)
            note = "cut off at the ceiling" if result.truncated else (problem or "ok")
            print(
                f"{tier:5s} {model}: {seconds:.1f} s, {result.usage.input_tokens} in / {result.usage.output_tokens} out, {note}"
            )
            print(f"      {result.text[:160]}")
            await asyncio.sleep(3.5)
    await provider.aclose()
    print("All models answered." if not failures else f"{failures} model(s) failed: see above.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
