# Free-model measurements

Each dated folder holds what `python -m app.models.testing` saved: every free model OpenRouter
listed that day, the four cases of `app/models/cases.py` sent to each through the gateway's own
provider adapter, and every answer with its grade, the reason it was wrong, its latency, and the
failure kind when there was no answer. Nothing in these files is edited by hand, so the default
chains in `app/config.py` can be re-derived from them without calling a model.

## 2026-10-26

* `models.json`: the first pass over all 17 free models, paced 4 seconds apart. Its `synthesis`
  case asked a question the first passage answered alone and required two citations, so every
  model failed it: a grading mistake, fixed before the second pass (the case now asks a
  two-part question that needs both passages).
* `models-pass2.json`: the second pass over the 13 models that the first pass had not ruled out,
  paced 6 seconds apart, with the corrected case.

What the two passes say together:

| Model | Grounded | Refusal | Synthesis | Judge | Median | Verdict |
|---|---|---|---|---|---|---|
| `nvidia/nemotron-3.5-lightning:free` | right | right | right | right | 0.96 s | all roles |
| `nvidia/nemotron-3.5-super-vl:free` | right | right | right | right | 1.07 s | all roles |
| `cohere/north-mini-code:free` | right | right | right | right | 1.08 s | all roles |
| `dots-studio/dots-3-note-preview:free` | right | right | right | right | 2.42 s | all roles |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | right | right | right | right | 2.89 s | all roles |
| `recursive/flint-v1.0:free` | right | right | right | right | 4.31 s | all roles |
| `liquid/lfm-2.5-2.6b:free`, `xingchen-agi/xing4.0-29b-a4b:free` | 429 | 429 | | right | | judge only (rate limited) |
| `poolside/laguna-s-2.1:free` | right (pass 1) | 429 | wrong (no citation) | right | | judge only |
| `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free` | 429 | 429 | 429 | 429 | | not judged |
| `nvidia/nemotron-3-super-120b-a12b:free` | 404 | | | | | unreachable from this account |
| `thinkingmachines/inkling:free`, `thinkingmachines/inkling-small:free` | 403 | | | | | unreachable: "only available on agentic harnesses" (recorded as `auth` that day; a 403 is `forbidden` since) |
| `nvidia/nemotron-3.5-content-safety:free` | wrong | wrong | wrong | wrong | | failed (a safety classifier, not a chat model) |

The defaults chosen from it, and why, are in `app/config.py` and in the README, "Free models
that stay working".
