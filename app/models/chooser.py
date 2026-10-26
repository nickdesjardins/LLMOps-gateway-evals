"""Keeps the service's free chains working while it runs: read, look, bench, repair.

**Automatic or pinned.** With ``LLM_PROVIDER=openrouter``, a key, and ``LIGHT_MODELS`` /
``HEAVY_MODELS`` left blank, the service looks after its chains itself (*automatic*). Naming a
tier's models *pins* that tier: those exact models are called and never replaced. Other
providers are *fixed*: their models are paid and named by the operator.

**At start-up** (no model is called): the chains start from the measured defaults, then from
``model_cache.json`` if this machine saved newer ones, then from the published file
(:mod:`app.models.published`) if its test is newer still.

**After a request a model could not answer** (the cascade reports every attempt): rate limits,
timeouts and upstream errors are the moment, not the model, and nothing happens. A 404, a 400
or a 403 is looked into in the background, at most once a minute per model: the published file is read
again (a repair may already be waiting there), OpenRouter's list is checked, and the model is
probed once with the ``grounded`` case. A model OpenRouter no longer lists, or that answers the
probe with a 404, 400 or 403 again, is **benched**: taken out of every chain at once. Its place is
filled by a test of up to three free models that are in no chain, with this project's cases
(:mod:`app.models.cases`), by the rules of :mod:`app.models.rules`: repair, never rank. Tests
started by the service are at least 15 minutes apart, and in demo mode they are paid from the
same budget as visitors (as the client ``model-test``, 30 calls a day at most).

The chains are kept in memory and, where the disk allows, in ``model_cache.json`` (never
committed). A host that forgets its files starts again from the published chains or the
defaults. Every id the chooser adopts must match ``vendor/model:free``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import DEFAULT_MEASURED_ON, Settings, Tier
from app.models import published, rules
from app.models.catalogue import free_models, is_free_id
from app.models.testing import STAMP, ModelReport, Pacer, run_test, try_model
from app.providers.base import Provider

logger = logging.getLogger(__name__)

TIERS: tuple[Tier, ...] = ("light", "heavy")
#: Failure kinds worth a look: the model may be gone, or closed to this account.
LOOK_AT = frozenset({"not_found", "bad_request", "forbidden"})
LOOK_EVERY_S = 60.0


def _stamp(moment: datetime) -> str:
    return moment.strftime(STAMP)


class ModelChooser:
    def __init__(
        self,
        settings: Settings,
        provider: Provider,
        *,
        budget: Any | None = None,
        list_free: Callable[[], list[str]] | None = None,
        fetch_published: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.budget = budget
        self.chains: dict[Tier, list[str]] = {tier: settings.chain(tier) for tier in TIERS}
        self.judge: list[str] = list(published.DEFAULT_CHAINS["judge"])
        self.mode = self._mode()
        self.origin: dict[str, Any] = {
            "source": "measured default",
            "tested": f"{DEFAULT_MEASURED_ON}T00:00:00Z",
        }
        self.benched: dict[str, str] = {}
        self.last_test: dict[str, Any] | None = None
        self._tested = datetime.strptime(DEFAULT_MEASURED_ON, "%Y-%m-%d").replace(tzinfo=UTC)
        self._list_free = list_free or (lambda: free_models(settings.openrouter_base_url))
        url = settings.model_published_url
        self._fetch_published = fetch_published or ((lambda: published.fetch(url)) if url else None)
        self._clock = clock
        self._looked: dict[str, float] = {}
        self._last_test_at: float | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ state
    def _mode(self) -> str:
        s = self.settings
        if s.llm_provider == "offline":
            return "offline"
        if s.llm_provider != "openrouter":
            return "fixed"
        if s.openrouter_api_key is None:
            return "no_key"
        if s.chain_pinned("light") and s.chain_pinned("heavy"):
            return "pinned"
        return "automatic"

    def _managed(self, tier: Tier) -> bool:
        return self.mode == "automatic" and not self.settings.chain_pinned(tier)

    def _use(
        self, chains: dict[str, list[str]], origin: dict[str, Any], tested: datetime, save: bool = True
    ) -> None:
        for tier in TIERS:
            chain = [m for m in chains.get(tier, []) if is_free_id(m) and m not in self.benched]
            if self._managed(tier) and chain:
                self.chains[tier][:] = chain[: rules.PLACES]  # in place: the cascade holds this list
        judge = [m for m in chains.get("judge", []) if is_free_id(m)]
        if judge:
            self.judge = judge[: rules.PLACES]
        self.origin = origin
        self._tested = tested
        if save:
            self._save()

    # ------------------------------------------------------------ the cache file
    def _cache_path(self) -> Path:
        return Path(self.settings.model_cache_path)

    def _load_cache(self) -> None:
        path = self._cache_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("fingerprint") != published.fingerprint():
                logger.info("Ignoring %s: it was saved by another version of the model test", path)
                return
            tested = datetime.strptime(data["tested"], STAMP).replace(tzinfo=UTC)
        except (OSError, ValueError, KeyError, TypeError):
            logger.info("Ignoring %s: it cannot be read", path)
            return
        self.benched = {m: when for m, when in (data.get("benched") or {}).items() if is_free_id(m)}
        if tested > self._tested:
            self._use(
                data.get("chains") or {},
                {"source": "this machine's last choice", "tested": data["tested"]},
                tested,
                save=False,
            )

    def _save(self) -> None:
        if self.mode != "automatic":
            return
        data = {
            "fingerprint": published.fingerprint(),
            "tested": _stamp(self._tested),
            "chains": {**self.chains, "judge": self.judge},
            "benched": self.benched,
            "origin": self.origin,
        }
        path = self._cache_path()
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            logger.info("Could not save the chosen chains in %s; they stay in memory", path)

    # ---------------------------------------------------------- published file
    async def read_published(self, *, reason: str) -> bool:
        """Adopt the published chains if they are newer than the ones in use. True when adopted."""
        if self._fetch_published is None or self.mode != "automatic":
            return False
        try:
            text = await asyncio.to_thread(self._fetch_published)
            found = published.parse(text)
        except published.Unusable as exc:
            logger.info("The published chains were not used (%s)", exc)
            return False
        except Exception as exc:  # 404 before the first publication, a network error
            logger.info(
                "The published chains could not be read (%s); the chains stay as they are", type(exc).__name__
            )
            return False
        if found.tested <= self._tested:
            return False
        if any(m in self.benched for chain in found.chains.values() for m in chain):
            logger.info("The published chains name a model benched here; they were not used")
            return False
        self._use(
            found.chains,
            {"source": "published", "tested": _stamp(found.tested), "read": reason},
            found.tested,
        )
        logger.info("Using the published chains tested %s: %s", _stamp(found.tested), self._describe())
        return True

    async def start(self) -> None:
        self._load_cache()
        await self.read_published(reason="start-up")
        if self.mode == "automatic":
            logger.info("Model choice: automatic (%s). %s", self.origin["source"], self._describe())
        else:
            logger.info("Model choice: %s. %s", self.mode, self._describe())

    def _describe(self) -> str:
        return (
            "; ".join(f"{tier}: {', '.join(chain)}" for tier, chain in self.chains.items())
            + f"; judge: {', '.join(self.judge)}"
        )

    # ------------------------------------------------------------ after failures
    def note(self, attempts: Sequence[Any]) -> None:
        """Called by the cascade after every request, with its attempts. Never raises."""
        if self.mode != "automatic":
            return
        now = self._clock()
        if len(attempts) > 1 and all(a.outcome in {"forbidden", "auth"} for a in attempts):
            logger.warning(
                "Every model answered 403: the key itself is closed (a spent budget, a guardrail); "
                "nothing benched"
            )
            return
        suspects = []
        for attempt in attempts:
            if attempt.outcome in LOOK_AT and attempt.model not in self.benched:
                if now - self._looked.get(attempt.model, -1e9) >= LOOK_EVERY_S:
                    self._looked[attempt.model] = now
                    suspects.append(attempt.model)
        if not suspects:
            return
        try:
            task = asyncio.get_running_loop().create_task(self.look(suspects))
        except RuntimeError:
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def look(self, models: Sequence[str]) -> list[str]:
        """Probe models that failed a real request; bench those that cannot be called. Returns them."""
        async with self._lock:
            await self.read_published(reason="after a failed request")
            try:
                listed = set(await asyncio.to_thread(self._list_free))
            except Exception as exc:
                logger.info(
                    "OpenRouter's model list could not be read (%s); nothing benched", type(exc).__name__
                )
                return []
            verdicts: dict[str, str] = {}
            probes = []
            for model in models:
                if model not in listed:
                    verdicts[model] = "OpenRouter no longer lists it"
                    continue
                report = await self._probe(model)
                if report is None or not report.unreachable:
                    continue
                probes.append(report)
                first = report.results[0]
                said = f"; OpenRouter said: {first.provider_said}" if first.provider_said else ""
                verdicts[model] = f"a probe could not reach it either ({first.failure}: {first.why}{said})"
            refused = rules.key_refused(probes) if len(probes) > 1 else None
            if refused:  # every probe refused: the key, not the models
                logger.warning("%s; nothing benched", refused)
                verdicts = {m: why for m, why in verdicts.items() if why.startswith("OpenRouter no longer")}
            benched = []
            for model, why in verdicts.items():
                self.benched[model] = _stamp(datetime.now(UTC))
                for chain in [*(self.chains[t] for t in TIERS if self._managed(t)), self.judge]:
                    if model in chain:
                        if len(chain) > 1:
                            chain.remove(model)
                        else:  # rule 4: a chain is never emptied
                            logger.warning("Kept %s: it is the last model of its chain", model)
                benched.append(model)
                logger.warning("Benched %s: %s", model, why)
            if benched:
                self.origin = {**self.origin, "benched_after": "a failed request"}
                self._save()
                if self.settings.model_repair:
                    await self.repair(reason=f"benched {', '.join(benched)}")
            return benched

    async def _probe(self, model: str) -> ModelReport | None:
        try:
            return await try_model(
                self.provider, model, pace=Pacer(0), before_call=self._charge, case_ids=["grounded"]
            )
        except Exception as exc:  # a spent budget, a network error: look again later
            logger.info("Probe of %s not sent (%s)", model, type(exc).__name__)
            return None

    async def _charge(self, _model: str) -> None:
        if self.budget is not None:
            await self.budget.charge_model_call("model-test")

    async def repair(self, *, reason: str) -> dict[str, Any] | None:
        """Fill free places with tested free models. At most one test every 15 minutes."""
        now = self._clock()
        if self._last_test_at is not None and now - self._last_test_at < rules.SECONDS_BETWEEN_TESTS:
            logger.info("No test now (%s): the last one was less than 15 minutes ago", reason)
            return None
        if all(len(self.chains[t]) >= rules.PLACES for t in TIERS) and len(self.judge) >= rules.PLACES:
            return None
        self._last_test_at = now
        try:
            listed = await asyncio.to_thread(self._list_free)
        except Exception:
            return None
        in_use = {m for chain in (*self.chains.values(), self.judge) for m in chain}
        candidates = [m for m in listed if m not in in_use and m not in self.benched][
            : rules.RUNTIME_CANDIDATES
        ]
        if not candidates:
            logger.warning("Nothing to test: every free model is in a chain or benched")
            return None
        logger.info("Testing %s to fill free places (%s)", ", ".join(candidates), reason)
        try:
            reports = await run_test(
                self.provider,
                candidates,
                seconds_between_starts=rules.SECONDS_BETWEEN_STARTS,
                before_call=self._charge,
            )
        except Exception as exc:
            logger.info("The test stopped (%s)", type(exc).__name__)
            return None
        refused = rules.key_refused(reports)
        if refused:  # every candidate refused: the key, not the models
            logger.warning("%s; no candidate benched", refused)
            return None
        by_model = {r.model: r for r in reports}
        for report in reports:
            if report.unreachable:
                self.benched[report.model] = _stamp(datetime.now(UTC))
        placings: dict[str, dict[str, str]] = {}
        for tier in TIERS:
            if self._managed(tier):
                chain, placings[tier] = rules.repair(self.chains[tier], by_model, role=tier)
                self.chains[tier][:] = chain
        self.judge[:], placings["judge"] = rules.repair(self.judge, by_model, role="judge")
        self.last_test = {
            "at": _stamp(datetime.now(UTC)),
            "reason": reason,
            "models": [r.as_dict() for r in reports],
            "placings": placings,
        }
        self.origin = {"source": "repaired here", "tested": self.last_test["at"]}
        self._tested = datetime.now(UTC)
        self._save()
        logger.info("Chains after the test: %s", self._describe())
        return self.last_test

    async def wait(self) -> None:
        """Let background looks finish (tests and shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def picture(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "chains": {tier: list(chain) for tier, chain in self.chains.items()},
            "judge": list(self.judge),
            "origin": self.origin,
            "benched": dict(self.benched),
            "last_test": self.last_test,
            "published_url": self.settings.model_published_url,
        }
