"""The LLM-as-a-judge behind the DeepEval metrics: the gateway's own providers, as a DeepEval model.

DeepEval calls ``a_generate(prompt, schema=...)`` with one of its prompt templates and a
Pydantic class for the answer (claims, verdicts, statements). This class sends the prompt
through the same provider adapters the service uses (OpenRouter's free models by default,
Claude, OpenAI or Gemini if configured), asks for a JSON object, parses it into the schema, and
moves to the next judge model when a model fails or answers something that does not parse. A
model that answered 404 or 400 twice gets no more requests for the rest of the run. DeepEval
then sees a model that always answers in schema, or an error it records.

A judge is a model, and a small free model is a noisy judge: the gate's thresholds are set for
it, and a stronger judge (``EVAL_JUDGE_PROVIDER=anthropic``, ``EVAL_JUDGE_MODELS=claude-haiku-5-5``)
costs a few cents per run. Requests are paced to stay inside the provider's rate limit, and
every call is counted, so a report says what the judging cost.

Settings (environment):

* ``EVAL_JUDGE_PROVIDER``: openrouter (default), anthropic, openai or gemini;
* ``EVAL_JUDGE_MODELS``: an ordered chain, comma-separated. Left blank with OpenRouter, the judge
  chain the scheduled model test published is used (the same file the service reads), and the
  measured default when nothing usable is published: the judge stays on free models that work;
* ``EVAL_JUDGE_MIN_INTERVAL_S``: at least this many seconds between two calls (3.2 by default).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from decimal import Decimal
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from app.config import DEFAULT_OPENROUTER_JUDGE, Settings, parse_model_chain  # noqa: E402
from app.pricing import PriceBook  # noqa: E402
from app.providers import ChatRequest, Provider, ProviderError, build_provider  # noqa: E402

DEFAULT_JUDGES = {
    "openrouter": DEFAULT_OPENROUTER_JUDGE,
    "anthropic": "claude-haiku-5-5",
}
#: Failure kinds that say a model cannot be called from this account (the service's rule 2,
#: app/models/rules.py): twice, and the judge stops sending it requests for the rest of the run.
BENCH_KINDS = frozenset({"not_found", "bad_request"})
BENCH_AFTER = 2


def published_judges() -> list[str]:
    """The judge chain the scheduled model test published, or the measured default."""
    from app.models import published

    settings = Settings()
    if settings.model_published_url:
        try:
            return published.parse(published.fetch(settings.model_published_url)).chains["judge"]
        except Exception as exc:  # nothing published yet, a stale file, no network
            print(f"Judge: the published chain was not used ({exc}); using the measured default.")
    return parse_model_chain(DEFAULT_OPENROUTER_JUDGE)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """The outermost JSON object in a reply, tolerating code fences and text around it."""
    match = _JSON_OBJECT.search(text.strip())
    if not match:
        raise ValueError("no JSON object in the reply")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))  # a trailing comma, the usual slip


class GatewayJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        provider: Provider | None = None,
        models: list[str] | None = None,
        *,
        min_interval_s: float | None = None,
        max_tokens: int = 2_048,
    ) -> None:
        judge_provider = os.environ.get("EVAL_JUDGE_PROVIDER", "openrouter")
        named = os.environ.get("EVAL_JUDGE_MODELS", "").strip()
        if models:
            self.models = list(models)
        elif named:
            self.models = parse_model_chain(named)
        elif judge_provider == "openrouter":
            self.models = published_judges()
        else:
            self.models = parse_model_chain(DEFAULT_JUDGES.get(judge_provider, ""))
        if not self.models:
            raise ValueError("EVAL_JUDGE_MODELS must name at least one judge model")
        if provider is None:
            overrides: dict[str, Any] = {
                "llm_provider": judge_provider,
                "light_models": ",".join(self.models),
                "heavy_models": ",".join(self.models),
            }
            provider = build_provider(Settings(**overrides))
        self.provider = provider
        self.min_interval_s = (
            float(os.environ.get("EVAL_JUDGE_MIN_INTERVAL_S", "3.2"))
            if min_interval_s is None
            else min_interval_s
        )
        self.max_tokens = max_tokens
        self.calls = 0
        self.failures: dict[str, int] = {}
        self.benched: list[str] = []
        self._strikes: dict[str, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = Decimal(0)
        self._prices = PriceBook()
        self._last = 0.0
        self._lock = asyncio.Lock()
        super().__init__(model=self.models[0])

    # -------------------------------------------------------------- DeepEval API
    def load_model(self, *args: Any, **kwargs: Any) -> GatewayJudge:
        return self

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"gateway-judge ({', '.join(self.models)})"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        return asyncio.run(self.a_generate(prompt, schema=schema))

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        last_error = "no judge model answered"
        callable_models = [m for m in self.models if m not in self.benched]
        if not callable_models:
            raise RuntimeError(
                f"every judge model answered 404 or 400 twice ({', '.join(self.benched)}): "
                "blank EVAL_JUDGE_MODELS to use the published chain"
            )
        for model in callable_models:
            for _ in range(2):  # a reply that does not parse gets one more try on the same model
                await self._pace()
                request = ChatRequest(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    tier="heavy",
                    temperature=0.0,
                    tag="eval-judge",
                    json_mode=schema is not None,
                )
                try:
                    completion = await self.provider.complete(model, request)
                except ProviderError as exc:
                    self.failures[f"{model}: {exc.kind}"] = self.failures.get(f"{model}: {exc.kind}", 0) + 1
                    last_error = f"{model}: {exc.detail}"
                    if exc.kind == "auth":
                        raise RuntimeError("the judge's provider refused the key") from exc
                    if exc.kind in BENCH_KINDS:
                        self._strikes[model] = self._strikes.get(model, 0) + 1
                        if self._strikes[model] >= BENCH_AFTER and model not in self.benched:
                            self.benched.append(model)
                    break  # next model
                self.calls += 1
                self.input_tokens += completion.usage.input_tokens
                self.output_tokens += completion.usage.output_tokens
                self.cost += (
                    completion.provider_cost_usd or self._prices.cost(model, completion.usage) or Decimal(0)
                )
                if schema is None:
                    return completion.text
                try:
                    return schema.model_validate(parse_json_object(completion.text))
                except (ValueError, ValidationError) as exc:
                    self.failures[f"{model}: unparsable"] = self.failures.get(f"{model}: unparsable", 0) + 1
                    last_error = f"{model}: reply did not match {schema.__name__} ({type(exc).__name__})"
        raise RuntimeError(last_error)

    async def _pace(self) -> None:
        async with self._lock:
            wait = self._last + self.min_interval_s - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    def usage(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "provider": self.provider.name,
            "calls": self.calls,
            "failures": self.failures,
            "benched": self.benched,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": str(self.cost),
        }
