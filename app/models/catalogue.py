"""OpenRouter's own word on its models: which free ones it lists today, and the account's allowance.

Two public facts, one request each, no free request spent:

* ``GET {base}/models`` lists every model; the free ones are those whose id ends in ``:free``
  and whose prompt and completion prices are zero. A model of the chain that is no longer
  listed has been withdrawn.
* ``GET {base}/key`` (with the key) tells how many free requests the account may still send
  today, which the scheduled test reads before it spends any.

A listed model can still be unusable from one account (NVIDIA answered "Function not found for
account" for a listed model on 26 October 2026), which is why the chooser also probes a model
before it benches it (:mod:`app.models.chooser`).
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from typing import Any

#: The only shape a model id the gateway chooses for itself may have. A typo or a bad file can
#: therefore never name a paid model.
FREE_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,95}:free")

Fetch = Callable[[str], Any]


def is_free_id(value: object) -> bool:
    return isinstance(value, str) and FREE_MODEL_ID.fullmatch(value) is not None


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - a fixed https host
        return json.load(response)


def free_models(base_url: str, fetch: Fetch | None = None) -> list[str]:
    """Every free text model OpenRouter lists, in its own order (newest first)."""
    data = (fetch or _get_json)(base_url.rstrip("/") + "/models")
    out: list[str] = []
    for model in data.get("data", []):
        model_id = model.get("id")
        pricing = model.get("pricing") or {}
        modality = (model.get("architecture") or {}).get("output_modalities") or ["text"]
        if not is_free_id(model_id) or "text" not in modality:
            continue
        try:
            if float(pricing.get("prompt", 0)) or float(pricing.get("completion", 0)):
                continue
        except (TypeError, ValueError):
            continue
        out.append(model_id)
    return out


def allowance(
    base_url: str, key: str, fetch: Callable[[str, dict[str, str]], Any] | None = None
) -> dict[str, Any]:
    """The key's own record: limits and, when OpenRouter reports it, free requests left today."""
    getter = fetch or (lambda url, headers: _get_json(url, headers))
    data = getter(base_url.rstrip("/") + "/key", {"Authorization": f"Bearer {key}"}).get("data", {})
    free = data.get("free_model_daily_requests") or {}
    return {
        "is_free_tier": data.get("is_free_tier"),
        "remaining_today": free.get("remaining"),
        "limit_today": free.get("limit"),
    }
