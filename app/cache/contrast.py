"""The contrast guard: a cheap lexical check on every semantic cache hit.

An embedding model scores short questions by what they are about, and two questions can be
about the same thing while asking for different answers. Measured with bge-small-en-v1.5 on
this project's intents (``scripts/measure_cache.py``, ``site/recorded/cache-threshold.json``):

* "what is the daily meal limit when travelling domestically" and "... internationally": 0.961
* "how fast must a p1 ticket get a first response" and "... a p2 ticket ...": 0.960

Both are above the 0.96 threshold, and both answers would be wrong. Real paraphrases of one
question, meanwhile, have a median similarity of 0.889. A threshold alone cannot separate the
two populations, so a semantic hit must also pass this check: the two questions may not
**contradict each other** on

* **numbers**: both mention numbers and the numbers differ (10,000 against 2,500);
* **codes**: both mention short codes and they differ (P1 against P2, W-9 against W-8);
* **polarity**: one says domestic and the other international, above and below, before and
  after, full-time and part-time, maximum and minimum.

Only contradictions count. A word present on one side and absent on the other ("before an
international trip" against "travel outside the country") is not one, which is why no real
paraphrase of the test set is blocked. What the guard cannot see is a substituted name
("hotel cap in Toronto" against "... in Ottawa", 0.909): that is the residual risk the
threshold has to cover.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_CODE = re.compile(r"\b[a-z]{1,4}-?\d{1,3}\b")

#: Pairs of opposite term sets. A question uses side 0, side 1, both, or neither.
_POLARITIES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        "domestic/international",
        frozenset({"domestic", "domestically", "inside the country", "within the country", "national"}),
        frozenset(
            {"international", "internationally", "outside the country", "abroad", "foreign", "overseas"}
        ),
    ),
    (
        "above/below",
        frozenset({"above", "over", "more than", "larger than", "greater than", "exceeding", "higher than"}),
        frozenset({"below", "under", "less than", "smaller than", "fewer than", "lower than"}),
    ),
    ("before/after", frozenset({"before"}), frozenset({"after"})),
    ("full-time/part-time", frozenset({"full-time", "full time"}), frozenset({"part-time", "part time"})),
    ("maximum/minimum", frozenset({"maximum"}), frozenset({"minimum"})),
    (
        "allowed/forbidden",
        frozenset({"allowed", "permitted"}),
        frozenset({"forbidden", "prohibited", "not allowed"}),
    ),
)

_COMPILED = tuple(
    (
        name,
        re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(left, key=len, reverse=True)) + r")\b"),
        re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(right, key=len, reverse=True)) + r")\b"),
    )
    for name, left, right in _POLARITIES
)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def contrast_conflict(a: str, b: str) -> str | None:
    """Why two questions must not share an answer, or None. Both are lower-cased by the caller."""
    numbers_a, numbers_b = _numbers(a), _numbers(b)
    if numbers_a and numbers_b and numbers_a != numbers_b:
        return f"numbers differ ({', '.join(sorted(numbers_a))} against {', '.join(sorted(numbers_b))})"
    codes_a, codes_b = set(_CODE.findall(a)), set(_CODE.findall(b))
    if codes_a and codes_b and codes_a != codes_b:
        return f"codes differ ({', '.join(sorted(codes_a))} against {', '.join(sorted(codes_b))})"
    for name, left, right in _COMPILED:
        side_a = {0} if left.search(a) else set()
        side_a |= {1} if right.search(a) else set()
        side_b = {0} if left.search(b) else set()
        side_b |= {1} if right.search(b) else set()
        if side_a and side_b and side_a != side_b:
            return f"opposite terms ({name})"
    return None
