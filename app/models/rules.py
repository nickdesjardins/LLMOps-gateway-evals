"""What a test of the free models concludes, as pure functions: no file, no clock, no network.

The rules, in four sentences (the same as the Phase 1 and Phase 3 services):

1. **Repair, never rank.** A model in a chain keeps its place and its order whatever one test
   says about its answers: four cases can rule a model out, they cannot rank two models that
   both pass, and free models are rate limited in bursts.
2. **A model leaves only on evidence that it cannot be called**: OpenRouter no longer lists it,
   or a call to it from this account answered 404 or 400 twice (a real request, then a probe or
   a test). A rate limit or a timeout is the moment, not the model.
3. **A free place goes to a model that passed** every case its role needs, behind the models
   already there, in the order a person chose (:data:`PREFERENCE`), then by speed.
4. **A chain is never emptied**: when nothing is left that can be called, the old chain stays
   and the owner is told.

Changing the order of two models that both work stays a person's job: measure, then change the
defaults in ``app/config.py`` (README, "Free models that stay working").
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from app.models.cases import Role
from app.models.testing import ModelReport

#: Models in a chain, at most: two like the measured default, and one spare.
PLACES = 3
#: Free places a runtime repair may fill in one go, and so the models it tests at most.
RUNTIME_CANDIDATES = 3
#: Seconds between request starts during a test: 10 a minute, half of OpenRouter's 20, so a
#: visitor's request still finds room while a test runs.
SECONDS_BETWEEN_STARTS = 6.0
#: Two tests started by the service itself are at least this far apart.
SECONDS_BETWEEN_TESTS = 15 * 60

#: Model families in the order a person prefers them when two candidates both pass, per role.
#: Measured on 26 October 2026 (benchmarks/2026-10-26/); a family is the id without ":free".
PREFERENCE: dict[Role, tuple[str, ...]] = {
    "light": ("nvidia/nemotron-3.5-lightning", "cohere/north-mini-code", "nvidia/nemotron-3.5-super-vl"),
    "heavy": ("nvidia/nemotron-3-ultra-550b-a55b", "recursive/flint-v1.0", "dots-studio/dots-3-note-preview"),
    "judge": ("dots-studio/dots-3-note-preview", "nvidia/nemotron-3-ultra-550b-a55b", "recursive/flint-v1.0"),
}


def _family(model: str) -> str:
    return model.removesuffix(":free")


def order_candidates(reports: Iterable[ModelReport], role: Role) -> list[ModelReport]:
    """Passing candidates for a role: the person's preference first, then the fastest."""
    preference = PREFERENCE.get(role, ())

    def key(report: ModelReport) -> tuple[int, float]:
        family = _family(report.model)
        rank = preference.index(family) if family in preference else len(preference)
        return rank, report.median_s if report.median_s is not None else 1e9

    return sorted((r for r in reports if role in r.roles()), key=key)


def repair(
    chain: Sequence[str],
    reports: Mapping[str, ModelReport],
    *,
    role: Role,
    gone: Iterable[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """The chain after a test: (models, what became of each). ``gone``: benched or unlisted."""
    gone = set(gone)
    placings: dict[str, str] = {}
    kept: list[str] = []
    for model in chain:
        report = reports.get(model)
        if model in gone or (report is not None and report.unreachable):
            placings[model] = "dropped"
            continue
        kept.append(model)
        placings[model] = "kept"
    for report in order_candidates(reports.values(), role):
        if len(kept) >= PLACES:
            break
        if report.model in kept or report.model in gone:
            continue
        kept.append(report.model)
        placings[report.model] = "added"
    if not kept:
        # Never empty a chain: the old one stays, and whoever reads the record is told.
        return list(chain), {model: "kept (nothing could replace it)" for model in chain}
    return kept, placings


def members_failing(chain: Sequence[str], reports: Mapping[str, ModelReport], role: Role) -> list[str]:
    """Models of a chain that a test judged and that failed their role: the owner's call."""
    failing = []
    for model in chain:
        report = reports.get(model)
        if report is not None and report.judged and not report.unreachable and role not in report.roles():
            failing.append(model)
    return failing
