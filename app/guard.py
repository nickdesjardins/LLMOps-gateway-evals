"""Input guards that run before any model call: prompt-injection defanging and token ceilings.

Defanging does not try to understand an attack; no regular expression can. It does three
mechanical things, then hands the model text it can only read as data:

1. **Normalize.** Unicode NFKC, then zero-width characters, bidirectional overrides and other
   control characters are removed, so ``ign​ore`` and full-width letters cannot hide a
   phrase from the checks or from a human reading the trace.
2. **Neutralize delimiters.** Chat-template markers (``<|im_start|>``, ``[INST]``), fake role
   headers (``### System:``) and anything that imitates the tags the gateway wraps untrusted
   text in are rewritten so they no longer parse as structure.
3. **Flag.** Known phrasings (instruction override, role hijack, prompt exfiltration) are
   reported as signals. The request goes on (``INJECTION_POLICY=defang``) or is refused
   (``block``); either way the trace records the signals and the answer is never cached.

The real defence is in the prompt (``app/rag/prompts.py``): untrusted text travels inside tags
with a random suffix the attacker cannot predict, and the system prompt says what those tags
mean. The gateway's own instructions sit in a static prefix the user never edits.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

#: Zero-width characters, bidirectional controls and the BOM: invisible, so never needed.
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤⁦-⁩﻿]")
#: C0 and C1 control characters other than tab and newline.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

#: Chat-template tokens of the common model families.
_TEMPLATE_TOKENS = re.compile(
    r"<\|\s*(?:im_start|im_end|system|user|assistant|endoftext|eot_id|start_header_id|"
    r"end_header_id|begin_of_text)\s*\|>|\[/?INST\]|<</?SYS>>",
    re.IGNORECASE,
)
#: A line that pretends to open a new role: "### System:", "SYSTEM:", "Assistant:".
_ROLE_HEADER = re.compile(
    r"^(?P<lead>[ \t#>*_-]*)(?P<role>system|assistant|developer|user)\s*(?P<sep>:|\])",
    re.IGNORECASE | re.MULTILINE,
)
#: Tags that imitate the gateway's own wrappers or a role block.
_FAKE_TAGS = re.compile(
    r"</?\s*(?:untrusted_input|question|context|sources?|system|assistant|developer|user|instructions?)"
    r"[\w-]*\s*>",
    re.IGNORECASE,
)

#: (signal name, pattern). Matched on normalized, lower-cased text.
_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(?:previous|prior|above|earlier|all|any|your|the|system)\b[^.\n]{0,30}"
            r"\b(?:instructions?|rules|prompts?|directions|guidelines|context)\b"
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"\byou are (?:now|no longer)\b|\bact as (?:an?|the) (?:unrestricted|jailbroken|"
            r"different)\b|\bdeveloper mode\b|\bdo anything now\b|\bjailbreak\b"
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|print|show|repeat|output|display|leak|tell me)\b[^.\n]{0,30}"
            r"\b(?:system|hidden|initial|original|developer)\s+(?:prompt|instructions?|message)"
        ),
    ),
    (
        "secret_request",
        re.compile(
            r"\b(?:api[_ -]?key|password|secret|token|credentials?)\b[^.\n]{0,30}"
            r"\b(?:reveal|send|print|give|share|output)\b|\b(?:reveal|send|print|give|share)"
            r"\b[^.\n]{0,30}\b(?:api[_ -]?key|password|credentials?)\b"
        ),
    ),
    ("template_tokens", _TEMPLATE_TOKENS),
    ("role_header", re.compile(r"^[ \t#>*_-]*(?:system|developer)\s*:", re.MULTILINE)),
    ("fake_tags", _FAKE_TAGS),
    ("encoded_payload", re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")),
)


@dataclass(frozen=True)
class GuardReport:
    """What the guard found and changed. ``signals`` is empty for ordinary text."""

    signals: tuple[str, ...] = ()
    neutralized: int = 0
    removed_characters: int = 0

    @property
    def suspicious(self) -> bool:
        return bool(self.signals)

    def as_dict(self) -> dict[str, object]:
        return {
            "detected": self.suspicious,
            "signals": list(self.signals),
            "neutralized": self.neutralized,
            "removed_characters": self.removed_characters,
        }


@dataclass
class _Counter:
    count: int = 0


def normalize(text: str) -> tuple[str, int]:
    """NFKC, then invisible and control characters removed. Returns (text, characters removed)."""
    folded = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL.sub("", _INVISIBLE.sub("", folded))
    return cleaned, len(folded) - len(cleaned)


def detect(text: str) -> tuple[str, ...]:
    """The signal names found in already-normalized text, in a fixed order."""
    lowered = text.lower()
    return tuple(name for name, pattern in _SIGNALS if pattern.search(lowered))


def defang(text: str) -> tuple[str, GuardReport]:
    """Normalize, detect, then neutralize structure-looking markup. The meaning stays readable."""
    cleaned, removed = normalize(text)
    signals = detect(cleaned)
    counter = _Counter()

    def _template(match: re.Match[str]) -> str:
        counter.count += 1
        # "<|im_start|>" -> "(im_start)": still readable in a trace, no longer a token.
        inner = re.sub(r"[<>|\[\]/]", "", match.group(0)).strip()
        return f"({inner.lower()})"

    def _role(match: re.Match[str]) -> str:
        counter.count += 1
        return f"{match.group('lead')}{match.group('role')} (quoted){match.group('sep')}"

    def _tag(match: re.Match[str]) -> str:
        counter.count += 1
        return match.group(0).replace("<", "(").replace(">", ")")

    result = _TEMPLATE_TOKENS.sub(_template, cleaned)
    result = _FAKE_TAGS.sub(_tag, result)
    result = _ROLE_HEADER.sub(_role, result)
    return result, GuardReport(signals=signals, neutralized=counter.count, removed_characters=removed)


# ---------------------------------------------------------------------- token ceilings


def estimate_tokens(text: str) -> int:
    """A provider-neutral estimate: about four characters per token for English text.

    Used for the input ceiling and when a provider reports no usage. Real counts from the
    provider always replace it in the telemetry, which marks estimates as such.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate for a chat request: the text of every message plus a few tokens of framing each."""
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages) + 2


def clamp_max_tokens(requested: int | None, ceiling: int) -> int:
    """The output budget sent to the provider: the client's number, never above the ceiling."""
    if requested is None or requested <= 0:
        return ceiling
    return min(requested, ceiling)
