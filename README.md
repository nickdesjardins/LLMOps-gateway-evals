# Production LLMOps Gateway: Evaluation CI/CD and Latency Optimization

A FastAPI gateway that sits in front of every model call and answers three questions a team
has to answer before a generative feature goes to production:

* **What does it cost?** A question that was already answered comes back from a Redis semantic
  cache in milliseconds, with no model call. A new one is scored for complexity and sent to a
  cheap light tier or a frontier heavy tier. Every call is priced twice: what it cost, and what
  the same tokens cost at a paid model's list price.
* **How long does it take, at the tail?** Responses stream as Server-Sent Events, output is
  capped per tier, and a model that is rate limited, slow, withdrawn or wrong hands over to the
  next one without dropping the request. Every step is a span of one trace, with p50, p95 and
  p99 per outcome, mirrored to Langfuse.
* **Did the last change make the answers worse?** A GitHub Actions workflow answers a golden
  set with the real pipeline, has DeepEval and Ragas judge every answer, and blocks the pull
  request when the hallucination rate passes 5% or faithfulness drops below 0.95.

| | |
|---|---|
| Project page (recorded requests, the measured threshold trade-off) | <https://nickdesjardins.github.io/LLMOps-gateway-evals/> |
| Live demo (free hosting: sleeps, about a minute to wake) | not online yet: see [docs/DEPLOY.md](docs/DEPLOY.md) |
| How it is deployed | [docs/DEPLOY.md](docs/DEPLOY.md) |

Phase 4 of an Applied AI portfolio. Phase 1:
[Invoice Extraction Service](https://nickdesjardins.github.io/Invoice-Extraction-Service/).
Phase 2: [Compliance Retrieval Engine](https://nickdesjardins.github.io/hybrid-knowledge-rag/).
Phase 3: [Multi-Agent Operations Pipeline](https://nickdesjardins.github.io/autonomous-operation-marketing/).

---

## Contents

- [Architecture](#architecture)
- [The semantic cache](#the-semantic-cache)
- [The threshold, measured](#the-threshold-measured)
- [Complexity routing](#complexity-routing)
- [Failover](#failover)
- [Free models that stay working](#free-models-that-stay-working)
- [Token economics](#token-economics)
- [Streaming](#streaming)
- [Prompt-injection defanging](#prompt-injection-defanging)
- [Telemetry and Langfuse](#telemetry-and-langfuse)
- [Evaluation and the quality gate](#evaluation-and-the-quality-gate)
- [Quick start](#quick-start)
- [API walkthrough](#api-walkthrough)
- [Configuration](#configuration)
- [Tests](#tests)
- [What was measured, and what was not](#what-was-measured-and-what-was-not)
- [Trade-offs and limits](#trade-offs-and-limits)
- [Project layout](#project-layout)

---

## Architecture

```
                         +---------------------------- trace (spans, tokens, $) ----> Langfuse
                         |
 client --> guard --> semantic cache --hit--> stored answer (no model call, milliseconds)
  (JSON or   (defang,     (exact key,
   SSE)      ceilings)     then KNN in Redis,
                           contrast guard)
                               |
                              miss
                               v
                         hybrid retrieval --> complexity router --> light chain  --+
                         (dense + BM25,       (named signals,       heavy chain  --+--> validate
                          RRF, top 4)          no model call)       failover       |    (citations)
                                                                                   v
                                                                     stream back, store in cache
```

Two entry points share it:

* **`POST /api/v1/ask`** (and `/api/v1/ask/stream`): a grounded question about a policy
  handbook. Retrieval, a static system prompt, cited answers, a refusal token when the
  handbook does not say. This is the RAG pipeline the evaluation suite measures.
* **`POST /v1/chat/completions`**: an OpenAI-compatible reverse proxy. Any Chat Completions
  client (the openai SDK, LangChain, curl) points its base URL at the gateway and gets caching,
  routing, failover, ceilings and tracing without changing a line. `model` selects a tier
  (`auto`, `light`, `heavy`), not a vendor.

The knowledge base (`corpus/`) is the handbook of a fictional company, Cedar Ridge Naturals,
written for this project: seven documents (expenses, travel, information security, records
retention, leave, IT service desk, supplier onboarding), 45 chunks. Some questions need one
passage, others two documents; one document quotes a phishing email that tells the reader to
"ignore all previous instructions", which makes it an indirect prompt-injection test.

## The semantic cache

`app/cache/semantic.py`, `app/cache/redis_store.py`, `app/cache/memory.py`, `app/cache/contrast.py`

A lookup runs two checks, cheapest first:

1. **Exact.** The question is case folded, its whitespace collapsed and its trailing `?!.`
   dropped, then hashed with its namespace. A stored hash is a hit at similarity 1.0: one key
   lookup, no embedding.
2. **Semantic.** Otherwise the question is embedded (BAAI/bge-small-en-v1.5, 384 dimensions,
   through fastembed on the CPU) and the three nearest cached questions of the same namespace
   are fetched. The first one at or above `CACHE_SIMILARITY_THRESHOLD` (0.96) that the
   **contrast guard** accepts is served.

**The contrast guard** refuses a semantic hit when the two questions contradict each other on
a number (10,000 against 2,500), a code (P1 against P2), or an opposite term (domestic against
international, above against below, before against after, full-time against part-time). Only
contradictions count, so a word present on one side and absent on the other blocks nothing:
the guard blocked none of the 78 real rephrasings of the test set. Why it exists is the next
section.

**The namespace** keeps apart answers that must never be served for each other. For a grounded
question it hashes the prompt version, the corpus fingerprint and the embedding model; for a
chat completion, the system prompt, the requested tier and the embedding model. Changing a
prompt or a document starts an empty cache instead of serving answers produced under the old
one.

**What is stored:** only a complete, validated answer to a request the guard did not flag. A
refusal ("the handbook does not answer this") is a valid answer and is cached too. A
conversation (more than one user turn) is never cached: its answer depends on every turn.

**Redis layout.** An HNSW index over HASH keys, cosine distance, the dimension in the index
name so a model change starts a new index:

```
FT.CREATE llmops-cache-idx-384 ON HASH PREFIX 1 llmops:cache:e:
    SCHEMA ns TAG embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE

FT.SEARCH llmops-cache-idx-384 "(@ns:{<namespace>})=>[KNN 3 @embedding $vec AS dist]"
    PARAMS 2 vec <float32 bytes> RETURN ... SORTBY dist LIMIT 0 3 DIALECT 2
```

Each entry also gets a plain `llmops:cache:x:<sha256>` key pointing at it, for the exact path,
and both expire with `CACHE_TTL_S`. Redis reports a cosine *distance*; the store converts it to
similarity. Every call has a 0.8 s socket timeout, a failure is a miss (or a skipped write),
and after three failures in a row the store stops calling Redis for 30 seconds: the cache never
takes a request down. A Redis without vector search (`unknown command FT.INFO`) is detected at
start-up and the service falls back to the in-process store, saying so in the log and in
`/health`.

**Without Redis** (`CACHE_BACKEND=memory`, or `REDIS_URL` empty) the same cache runs on a
numpy matrix per namespace: exact cosine search, least recently used eviction, expiry. It is
faster than Redis (no network) and forgets everything at a restart, which on a free host that
sleeps is often.

## The threshold, measured

`scripts/measure_cache.py` writes `site/recorded/cache-threshold.json` (no model call, no key).

`evals/cache_intents.json` holds 25 help-desk intents with three or four phrasings each, and
eight pairs of "near-miss" intents whose questions look alike but whose answers differ. Every
pair of phrasings of one intent should hit (78 pairs); every cross pair of a near miss must not
(78 pairs). Embedded with bge-small-en-v1.5 on 23 October 2026:

| | |
|---|---|
| Median similarity of two rephrasings of one question | 0.889 (lowest 0.740) |
| Highest similarity of two look-alikes with different answers | **0.961** ("daily meal limit when travelling domestically" and "... internationally") |
| Next ones | 0.960 (P1 against P2 first response), 0.960 (domestic against international, other wording) |

The two populations overlap: no threshold separates them.

| Threshold | Rephrasings served from the cache | Look-alikes served a wrong answer, with the guard | without it |
|---|---|---|---|
| 0.85 | 80.8% | 3.8% | 37.2% |
| 0.88 | 56.4% | 3.8% | 29.5% |
| 0.90 | 39.7% | 2.6% | 20.5% |
| 0.92 | 23.1% | 0.0% | 14.1% |
| 0.94 | 17.9% | 0.0% | 11.5% |
| **0.96** | **12.8%** | **0.0%** | **1.3%** |
| 0.97 | 9.0% | 0.0% | 0.0% |

Three readings:

* **At 0.96 without the guard, the cache serves a wrong answer.** The domestic meal limit
  (75 dollars) for the international question (110). The threshold from the specification is
  not safe on its own.
* **The guard is what makes a lower threshold usable.** With it, 0.92 serves almost twice as
  many rephrasings as 0.96 and no wrong answer on this set. What the guard cannot see is a
  substituted name: "nightly hotel cap in Toronto" and "... in Ottawa" (225 against 175
  dollars) are 0.909 similar and contradict nothing it knows about. Below about 0.91 that is
  the residual risk.
* **The semantic layer is the smaller part of the saving.** Most repeated questions are
  repeated word for word, which the exact path handles; see the replay below.

The project page has a slider over these numbers. Embedding one question takes 6.5 ms at the
median on the development machine (Windows, one process).

### The replay

`python -m evals.replay` writes `site/recorded/replay.json`: 250 questions drawn from the 25
intents with Zipf-like popularity (a few asked often, most rarely), each time in one of their
phrasings, seed 7. Every question goes through the real pipeline (guard, cache with the real
embedding model, retrieval, router); misses are answered by the offline provider (no model
call) and priced at the list price of the tier's reference model.

| Threshold | Hits (exact / semantic) | False hits | List-price spend avoided | Guard refusals |
|---|---|---|---|---|
| 0.96 | 199 of 250 (172 / 27) | 0 | 79.8% | 1 |
| 0.92 | 204 of 250 (162 / 42) | 0 | 81.7% | 3 |
| 0.88 | 215 of 250 (130 / 85) | **2** | 86.1% | 21 |

The 250 requests hold 57 distinct questions, so the saving says as much about the workload as
about the cache: a help desk that gets fewer repeats saves less. The same replay against a
Redis 8.2 container on the same machine (`--redis-url`, `site/recorded/replay-redis.json`)
gave the same hits and false hits; an exact lookup took 2.4 ms at the median, a semantic one
(embedding, KNN and the network round trips) 3.6 ms, with a p99 of 17.8 ms. Through the
production image, a semantic hit took 15 ms end to end with one inference thread.

## Complexity routing

`app/routing/complexity.py`

A router that called a model to decide would add a round trip and a bill to every miss, which
is what it exists to avoid. It scores the request from named signals, starting at 0.30:

| Signal | Weight | Matches |
|---|---|---|
| classification | -0.20 | "classify", "triage", "which category / team / priority" |
| extraction | -0.20 | "extract", "parse", "pull out ... from" |
| structured_output | -0.10 | JSON, CSV, key-value |
| short_lookup | -0.10 | a single short question starting with what, how many, who, is, can |
| comparison | +0.25 | compare, versus, difference between |
| analysis | +0.20 | analyze, evaluate, trade-offs, implications |
| planning | +0.20 | step by step, walk me through, recommend, "what should accounts payable do" |
| explanation, conditional, aggregation | +0.15 each | why, explain; if, suppose; summarize, all the |
| several_questions | +0.20 | two question marks or more |
| long_request | +0.10 / +0.20 | above about 45 / 120 tokens |
| spans_two / three_documents | +0.15 / +0.30 | the strongest passages come from that many documents |

A score of 0.5 or more goes to the heavy tier. The retrieval-spread signal counts only passages
within 0.05 of the best dense score **and above 0.60**: measured on the golden set, in-scope
questions have a best dense score of 0.645 or more, and a question the handbook does not cover
matches many documents weakly, which is irrelevance, not complexity. On the 37 golden
questions the router sends 33 to the light tier and 4 to the heavy one: the comparison, the
two "what should I do" questions and the lost-laptop question that spans two documents
(`site/recorded/routing.json`, written by `scripts/route_examples.py`).

A client can decide instead: `task: "classify"` or `"extract"` forces light, `"synthesize"`
forces heavy; on the chat route `model: "light"` or `"heavy"`.

## Failover

`app/routing/cascade.py`

The plan for a tier is its chain, then (with `CROSS_TIER_FALLBACK=true`) the other tier's
chain. Every attempt is a generation span in the trace, failed ones included.

| What happened | What the cascade does |
|---|---|
| 429, timeout, 5xx, dropped connection, OpenRouter's in-band error | next model |
| 404 (a withdrawn free model) | next model |
| 400 (this model's endpoint refused the request's shape) | next model, after one retry without OpenRouter's extensions |
| empty answer, cut off before any text (a model that thinks first), refusal | next model |
| an answer that fails validation (no citation, a citation to a passage that was not given) | next model |
| 401 / 403 | stop: every model of the provider shares the key (502) |
| `REQUEST_DEADLINE_S` reached | stop (503, `Retry-After: 10`) |
| every model failed | 503 with `Retry-After` when any failure was transient, 502 when all were permanent |

Streaming can fail over only until the first token is sent; after that, splicing another
model's answer onto part of the first would be worse than an error event. The SDKs' own
retries are off: on a 429 the right move is another model, not the same one after a back-off.

## Free models that stay working

`app/models/` (`cases`, `testing`, `rules`, `published`, `chooser`, `catalogue`),
`.github/workflows/model-check.yml`

Free models are withdrawn without notice, rate limited in bursts, and sometimes listed while
unusable: on 26 October 2026 OpenRouter listed `nvidia/nemotron-3-super-120b-a12b:free`, and
every call to it from this project's account answered 404 ("Function not found for account").
Like Phases 1 and 3, the gateway looks after its chains itself, so the demo and the quality
gate stay on free models that work.

**Automatic or pinned.** With OpenRouter, a key, and `LIGHT_MODELS` / `HEAVY_MODELS` blank
(what `.env.example` ships), the service is in *automatic* mode. Naming a tier's models *pins*
it: those exact models, never replaced. `/health` and `/api/v1/models` say which, where the
chains came from, and what was benched; the playground prints it under its metrics.

**The four cases** (`app/models/cases.py`). Being listed as free proves nothing; a model must do
this gateway's own work, through the real prompt, ceilings and validation:

| Case | What is sent | Right when |
|---|---|---|
| `grounded` | The RAG prompt, two passages, a vacation-days question | It cites a passage it was given, says 20, and is not cut off at the light ceiling |
| `refusal` | The RAG prompt, passages that do not answer a stock-price question | Exactly the refusal token |
| `synthesis` | Three passages, a two-part question (a supplier's new bank details, and how to report the email) | It cites both passages it needs and names the callback, the 5-day hold and the Report Phish button |
| `judge` | A DeepEval-style verdict task: two claims, one context | JSON with one verdict per claim: yes, then no |

Light needs `grounded` and `refusal` and a median answer under 12 s; heavy needs those and
`synthesis`; the eval judge needs `judge`. A 404 or 400 makes a model *unreachable* (its test
stops there); a 429, a timeout or an upstream error leaves it *not judged*: the moment, not the
model.

**The measured defaults** (26 October 2026, `benchmarks/2026-10-26/`, every answer saved). Of
the 17 free models OpenRouter listed, six passed all four cases:

| Model | Median answer | Role in the defaults |
|---|---|---|
| `nvidia/nemotron-3.5-lightning:free` | 0.96 s | light, first |
| `cohere/north-mini-code:free` | 1.08 s | light, second |
| `nvidia/nemotron-3.5-super-vl:free` | 1.07 s | first light candidate for a repair |
| `dots-studio/dots-3-note-preview:free` | 2.42 s | judge, first |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 2.89 s | heavy, first; judge, second |
| `recursive/flint-v1.0:free` | 4.31 s | heavy, second |

Each chain spans two upstream vendors. The judge chain starts with a model that writes none of
the answers, so the judge does not grade its own work. Both Google models were rate limited on
every call that day (not judged), and `nemotron-3-super` and the two `thinkingmachines/inkling`
models were unreachable. `python -m app.models.testing --all` measures again (about four free
requests per model).

**Repair, never rank** (`app/models/rules.py`, pure functions):

1. A model in a chain keeps its place and its order whatever one test says about its answers:
   four cases can rule a model out, they cannot rank two models that both pass.
2. A model leaves only on evidence that it cannot be called: OpenRouter no longer lists it, or a
   404 or 400 on a real request followed by the same answer to a probe.
3. A free place goes to a model that passed every case its role needs, behind the models
   already there, in the order a person chose (`PREFERENCE`), then by speed. At most three
   models per chain.
4. A chain is never emptied: when nothing can replace it, the old chain stays and the owner is
   told.

**While the service runs** (`app/models/chooser.py`). Starting calls no model: the chains start
from the measured defaults, then `model_cache.json` (this machine's last choice, never
committed), then the published file if its test is newer. After a request, the cascade reports
every attempt. A 429 or a timeout is left alone. A 404 or 400 is looked into in the
background, at most once a minute per model: the published file is read again, OpenRouter's
list is checked, and the model is probed once with the `grounded` case. A model that is
unlisted, or fails the probe the same way, is **benched** at once (the next request no longer
calls it), and a test of up to three free models in no chain fills its place. Tests started
by the service are 15 minutes apart, request starts 6 seconds apart; in demo mode they are paid
from the public budget as the client `model-test` (30 calls a day at most). Every id the
chooser adopts must match `vendor/model:free`.

**The published chains** (`app/models/published.py`, the **Model check** workflow). A service
nobody visits tests nothing, and a free host forgets its files at every sleep. So a scheduled
job runs every day at 06:47 UTC: deciding whether a test is due costs nothing (one is due when
nothing usable is published, after six and a half days, 20 hours after a test that confirmed
nothing, or as soon as OpenRouter's list of free models changes). The test sends the four cases
to up to 24 free models (about 100 free requests, read against the day's allowance first) and a
second job, which runs no downloaded code, commits the chains to `chain.json` on the branch
`tested-models` (not `main`: Render would redeploy at every publication). Services read it at
start-up and after a failure, and both judges of the quality gate read its `judge` chain. The
file is checked before it is used: 64,000 bytes at most, version 1, a date not in the future,
one to three `:free` ids per role, and a fingerprint of the cases, the prompt, the rules and
the defaults, so a fork that changes a prompt ignores this project's file by itself. A red run
ends with a line that begins "Failing this run on purpose:" saying what the owner has to do.

With `LLM_PROVIDER=anthropic` the tiers are `claude-haiku-5-5` and `claude-opus-5-5`, paid
models the operator chose: nothing is tested or replaced.

## Token economics

`app/guard.py`, `app/pricing.py`, `app/providers/`

* **Ceilings.** 350 output tokens for the light tier, 900 for the heavy one, whatever the
  client asks for (the client's `max_tokens` is honoured when lower). A prompt estimated above
  `MAX_INPUT_TOKENS` (6,000) is refused before any call. A model that runs into the ceiling
  without finishing is treated as a failure and the next one is tried, so a rambling model
  costs one capped call, not an unbounded one.
* **Static prefix.** The system prompt (`app/rag/prompts.py`) has no date, no id and no nonce:
  it is the same bytes on every request, and the passages and the question come after it.
  Providers cache prompts by prefix. With Claude the prefix is sent as one system block marked
  `cache_control: {"type": "ephemeral"}`; the usage counters (`cache_read_input_tokens`) are
  normalized into the trace, and cache reads and writes are priced at their own rates.
* **Priced twice.** Every generation carries the actual cost (the provider's own figure when it
  reports one, OpenRouter's `usage.cost`; otherwise tokens times the price table; zero for a
  `:free` model) and the **list-price** cost of the same tokens on the tier's reference model.
  Prices are Anthropic's, per million tokens, as published on 6 October 2026:

  | Model | Input | Output | Cache read |
  |---|---|---|---|
  | claude-haiku-5-5 (light reference) | $0.10 | $0.50 | $0.01 |
  | claude-opus-5-5 (heavy reference) | $4.00 | $20.00 | $0.20 |

  Check them before quoting a figure; `MODEL_PRICES` adds or overrides models.
* **Cost avoided.** A cache hit records the list-price cost of the miss that produced the
  entry. `/api/v1/metrics` reports `cost_reduction = avoided / (reference + avoided)`: the
  share of the bill the cache removed.

## Streaming

`POST /api/v1/ask/stream` answers with `text/event-stream`:

```
event: meta      {"trace_id", "cache": {...}, "route": {...}}
event: sources   {"sources": [{label, chunk_id, title, rank, rrf, dense, bm25}, ...]}
event: delta     {"text": "..."}          (repeated)
event: done      the full AskResponse, with latency_ms and ttft_ms
event: error     {"type", "message", "incident_id"}   (instead of done, when it fails)
```

The first characters are held back until they can no longer be the refusal token, so a reader
never sees `INSUFFICIENT_CONTEXT` scroll past. Anything that refuses a request (the injection
policy, the token ceiling, a spent budget, a busy demo) refuses it *before* the stream opens,
as a normal HTTP error. `POST /v1/chat/completions` with `stream: true` uses the Chat
Completions chunk format, ending with `data: [DONE]`; its last chunk carries `usage` and
`x_gateway`.

## Prompt-injection defanging

`app/guard.py`, `app/rag/prompts.py`

No regular expression understands an attack, so the guard does three mechanical things and
leaves the defence to the prompt's structure:

1. **Normalize**: Unicode NFKC, then zero-width characters, bidirectional overrides and control
   characters are removed (`ign​ore` and full-width letters cannot hide a phrase).
2. **Neutralize**: chat-template tokens (`<|im_start|>`, `[INST]`), fake role headers
   (`### System:`) and anything imitating the gateway's own tags are rewritten so they no
   longer parse as structure (`<|im_start|>` becomes `(im_start)`).
3. **Flag**: instruction override, role hijack, prompt exfiltration, secret requests, encoded
   payloads. The request goes on (`INJECTION_POLICY=defang`) or is refused (`block`); either
   way the trace records the signals and the answer is never cached.

The question then travels inside `<question_xxxxxxxx>` tags whose suffix is drawn per request,
next to `<sources_xxxxxxxx>`, under a system prompt that says what the tags mean and that
nothing inside them is an instruction. Text cannot close a tag it cannot name.

## Telemetry and Langfuse

`app/telemetry/`

One trace per request, one span per step:

```
gateway.ask                     endpoint, hashed client id, cache outcome, tier, totals
  guard.defang                  guardrail: signals, characters neutralized
  cache.lookup                  embedding: exact or semantic, similarity, embed and search ms
  retrieval.hybrid              retriever: chunk ids with dense, BM25 and RRF scores
  router.complexity             tier, score, named signals
  llm.attempt 1                 generation: model, tokens, actual and list-price cost
  llm.attempt 2                 only after a failure: the failover
  cache.write
```

The service keeps the last `TRACES_KEPT` traces (`/api/v1/traces`) and aggregates them
(`/api/v1/metrics`): requests by outcome and tier, cache hit rate, p50/p95/p99 latency for
hits and misses separately, time to first token, tokens, the three costs, fallback rate,
failures by model and kind. Percentiles are exact over a sliding window of 5,000 requests, by
linear interpolation (numpy's `linear` method).

**Langfuse** (`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`): each span becomes a Langfuse
observation as it starts, through the v5 SDK, with the matching type (`generation`, `retriever`,
`guardrail`, `embedding`). Generations carry the model id, `usage_details` (input, output,
cached input) and `cost_details`, so Langfuse's own dashboards show tokens, cost and latency per
model. Trace name, hashed user id, session id and tags are propagated to every observation,
which Langfuse's aggregations by user and session need. The SDK exports in a background thread;
a Langfuse outage is logged once and never reaches a request. With
`LANGFUSE_CAPTURE_CONTENT=false` only numbers and ids leave the service, never the text of a
question or an answer.

## Evaluation and the quality gate

`evals/`, `.github/workflows/quality-gate.yml`

**The golden set** (`evals/golden/golden.jsonl`): 37 cases. 20 typical questions, 7 edge cases
(a boundary amount, a multi-step procedure, a comparison, two documents, a reasoning step), 5
prompt-injection attempts (direct override, tag spoofing, the indirect injection in the
phishing example, a role hijack) and 5 questions the handbook does not answer. Each case names
the chunks that hold its answer and, for adversarial ones, the strings that must not appear in
the answer. `python -m evals.synth adversarial` wraps the typical questions in five injection
templates (40 candidates, no key); `python -m evals.synth goldens` has a model write questions
from the corpus and a critic pass reject the ones that are not self-contained or not faithful.
Synthetic cases are written next to the golden set for a person to review, never into it.

**The stages**, each writing one file the next reads:

| Stage | Command | Needs |
|---|---|---|
| Retrieval check | `python -m evals.retrieval_check` | nothing (the local embedding model) |
| Answers | `python -m evals.generate_answers` | the provider key; 37 requests, more when a model fails over |
| DeepEval | `python -m evals.score_deepeval` | a judge; 7 calls per answered case, about 210, and 2 to 4 per flagged case |
| Ragas | `python evals/score_ragas.py` (own environment) | a judge; about 7 calls per answered case, about 210 |
| Gate | `python -m evals.gate --retrieval ... --deepeval ... --ragas ...` | nothing (standard library) |

**The metrics.** Retrieval, from labels: hit rate@4, recall@4, MRR, and context precision@4
(average precision). DeepEval, with the gateway's own providers as the LLM-as-a-judge
(`evals/judge.py`): `FaithfulnessMetric` with `penalize_ambiguous_claims=True` (a claim the
passages do not support counts against the answer, not only one they contradict) and its claim
step also given the question, as Ragas' statement step is (DeepEval extracts claims from the
answer alone, so the correct answer "14 characters [S1]" became the claim "14 characters",
which no passage can back),
`AnswerRelevancyMetric`, `ContextualPrecisionMetric`, and `HallucinationMetric` against the
reference passages. Note that in DeepEval 4 the HallucinationMetric score is the share of
reference passages the answer **agrees** with (higher is better; older releases scored the
share of contradictions), so the report calls it `reference_agreement`. Ragas, as a second
judge: `Faithfulness`, `AnswerRelevancy` (with the same local embedding model) and
`ContextPrecisionWithReference`. Two checks need no judge: refusal accuracy and injection
resistance.

**Hallucination rate**: the share of cases whose answer contradicts most of its reference
passages, or has fewer than 70% of its claims supported, or answers a question the handbook
does not answer. A small free judge does not give the same verdict twice, so the metric that
flags a case is measured again by the next model of the judge chain: the case counts only when
that second judge flags it too, and its score becomes the mean of the two opinions. The report
keeps the first opinions and the rate they alone give, and the gate's summary prints both.

**The gate** (`evals/thresholds.json`):

| Rule | Threshold |
|---|---|
| Hallucination rate | at most 5% |
| Faithfulness (DeepEval) | at least 0.95 |
| Injection resistance | 100% |
| Refusal accuracy | at least 0.90 |
| Retrieval hit rate@4 / context precision@4 | at least 0.95 / 0.90 |
| Cases the pipeline failed to answer; metrics the judge failed to score | 0; at most 2 |
| Every tracked metric (DeepEval, Ragas, retrieval) | no more than 0.03 below the accepted baseline |

Missing evidence fails the gate: a report that should exist and does not, an errored case.
The baseline (`evals/baseline.json`) is written on purpose, from a run someone looked at:
`python -m evals.gate ... --write-baseline`. Without one, the regression rules are skipped and
the summary says so.

**On GitHub Actions** (`quality-gate.yml`): a `changes` job decides whether the pull request
touches anything an answer depends on (`app/`, `corpus/`, `evals/`, the requirements). The
retrieval check runs every time; the judged suite runs when it matters: about 460 requests
in all on the free models, which needs OpenRouter's 1,000-a-day allowance (a one-time 10 dollar
credit purchase), not the 50 a day of an account without credits. The `Gate` job always runs, so a pull request
that touches only the README is not left waiting for a check that never reports; make it a
required status check and a failing gate disables the merge button (docs/DEPLOY.md, step 6).

**The judge.** A small free model is a noisy judge. The thresholds are the specification's;
whether a free judge holds the pipeline to them without false alarms was measured, not
assumed. The first runs (26 October 2026) found three faults in how the judging was done, not
in the answers, and each was fixed and kept: a judge model that answers 404 or 400 twice
stops getting requests; a flagged case is judged again by the next model before it counts; and
DeepEval's claim step reads the question, without which two different judges scored the
correct "14 characters [S1]" 0.0 (a control with "12 characters" still scores 0.0). One
false alarm remained (above). `EVAL_JUDGE_PROVIDER=anthropic` with
`EVAL_JUDGE_MODELS=claude-haiku-5-5` (repository variables, plus the `ANTHROPIC_API_KEY`
secret) is a stronger judge for a few cents a run.

## Quick start

Python 3.12. On Windows, in PowerShell, from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-eval.txt
```

**Without any key** (the offline provider answers with the best retrieved passage; the cache,
router, trace and costs are the real code):

```powershell
.\.venv\Scripts\python.exe scripts\run_local_demo.py --offline
```

Open <http://127.0.0.1:8013/>, ask a sample twice in other words, and watch the second answer
come from the cache. The first start downloads the embedding model (about 67 MB).

**With free models**: copy `.env.example` to `.env.local`, set `OPENROUTER_API_KEY`
(<https://openrouter.ai/keys>), check the chains, then run without `--offline`:

```powershell
.\.venv\Scripts\python.exe scripts\openrouter_smoke.py
```

```powershell
.\.venv\Scripts\python.exe scripts\run_local_demo.py
```

**With Redis** (Docker): `docker compose up -d redis`, then `REDIS_URL=redis://localhost:6379/0`
in `.env.local`. `/health` then says `"backend": "redis"`.

**The tests** (no key, no network):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## API walkthrough

Against the local demo (`scripts/run_local_demo.py`, port 8013). In demo mode the routes are
open; outside it, `/v1/chat/completions` wants `Authorization: Bearer <one of GATEWAY_API_KEYS>`.

A grounded question:

```bash
curl -s http://127.0.0.1:8013/api/v1/ask -H "content-type: application/json" -d '{"question": "What is the daily meal limit when travelling domestically?"}'
```

The response (`AskResponse`, abridged) and its headers `X-Gateway-Cache`, `X-Gateway-Tier`,
`X-Gateway-Model`, `X-Gateway-Trace-Id`. This one is the first request of
`site/recorded/runs.json`, recorded on 26 October 2026 with the default free chain:

```json
{
  "trace_id": "32c5854cd148662032cb7f3c96bc0e5e",
  "answer": "75 dollars per person, tips included [S1].",
  "refused": false,
  "citations": [{"label": "S1", "chunk_id": "expense-policy#meals-and-entertainment", "title": "Expense Reimbursement Policy > Meals and entertainment"}],
  "cache": {"outcome": "miss", "similarity": null, "threshold": 0.96, "embed_ms": 8.31, "backend": "memory", "blocked": []},
  "route": {"tier": "light", "score": 0.2, "base": 0.3, "threshold": 0.5, "decided_by": "router", "signals": [{"name": "short_lookup", "weight": -0.1}]},
  "model": "nvidia/nemotron-3.5-lightning:free",
  "tier": "light",
  "attempts": [{"model": "nvidia/nemotron-3.5-lightning:free", "tier": "light", "outcome": "ok", "detail": "", "latency_ms": 1549.7}],
  "usage": {"input_tokens": 787, "output_tokens": 13, "cached_input_tokens": 0, "estimated": false},
  "cost": {"actual_usd": "0.0000000000", "reference_usd": "0.0000852000", "avoided_usd": "0.0000000000"},
  "latency_ms": 1566.87
}
```

The same question in other words ("What is the daily meal limit for domestic travel?") came
back 10 ms later from the cache, with `"cache": {"outcome": "hit", "match": "semantic",
"similarity": 0.9849, ...}`, no attempts, and `avoided_usd` of `0.0000852000`: the first
answer's list-price cost on Claude Haiku 5.5.

Streamed:

```bash
curl -N http://127.0.0.1:8013/api/v1/ask/stream -H "content-type: application/json" -d '{"question": "How fast must a P1 ticket get a first response?"}'
```

Through the OpenAI-compatible route, with the openai SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8013/v1", api_key="your-gateway-key")
reply = client.chat.completions.create(
    model="auto",  # or "light" / "heavy"
    messages=[
        {"role": "user", "content": "Classify this ticket as hardware or software: my screen flickers."}
    ],
)
print(reply.choices[0].message.content, reply.model_extra["x_gateway"]["cache"])
```

Tools that call no model: `POST /api/v1/route {"text": ...}` (the router's decision),
`POST /api/v1/similarity {"a": ..., "b": ...}` (the cache's similarity and the guard's
verdict). Operations: `GET /livez`, `GET /health`, `GET /api/v1/metrics`,
`GET /api/v1/traces?limit=20`, `GET /api/v1/traces/{trace_id}`, `GET /api/v1/corpus`.

Errors are always `{"error": {"type", "message", "incident_id"}}`: `invalid_request` (400,
never echoing the rejected input), `prompt_too_long`, `prompt_injection` (with `block`),
`unauthorized` (401), `rate_limited` and `budget_exhausted` (429 with `Retry-After`),
`busy` and `upstream_unavailable` (503), `upstream_failed` (502), `not_configured` (503). A
provider's own error text never reaches a client; the log has it under the incident id.

## Configuration

Every setting is in `app/config.py` and documented in `.env.example`. The ones that matter
most:

| Setting | Default | |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | `anthropic`, `openai`, `gemini`, `offline` |
| `LIGHT_MODELS`, `HEAVY_MODELS` | provider defaults | ordered chains |
| `CACHE_SIMILARITY_THRESHOLD` | 0.96 | see the measured trade-off |
| `CACHE_CONTRAST_GUARD` | true | |
| `CACHE_BACKEND`, `REDIS_URL` | `auto`, empty | Redis when the URL is set |
| `ROUTER_HEAVY_THRESHOLD` | 0.5 | |
| `MAX_OUTPUT_TOKENS_LIGHT` / `_HEAVY` | 350 / 900 | |
| `INJECTION_POLICY` | `defang` | or `block` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | empty, empty, EU cloud | |
| `DEMO_MODE` | false | the public budget: `REQUESTS_PER_MINUTE_PER_CLIENT`, `MODEL_CALLS_*`, `MAX_INFLIGHT` |
| `EMBEDDING_PROVIDER` | `fastembed` | `hash`: no model, exact and near-identical repeats only |

Start-up refuses a key that contains `://` or whitespace (a connection string pasted into the
key box would otherwise be sent to a model provider) without echoing it, an OpenRouter id
without `:free` while `OPENROUTER_FREE_ONLY` is on, and one Langfuse key without the other.

## Tests

`python -m pytest`: the hermetic suite. No network (outbound connections are refused at the
socket level), no Redis, no model, no key; every provider key is removed from the environment.

* the real OpenAI and Anthropic SDK adapters against a mock HTTP transport: request shapes
  (OpenRouter's reasoning switch and usage accounting, `max_completion_tokens` for OpenAI, the
  Claude system block with `cache_control`, effort per tier, the refusal fallback on
  `claude-opus-5-5`), usage normalization, every error mapped to its failure kind;
* the Langfuse SDK itself, exporting to an in-memory OpenTelemetry exporter: nesting, types,
  usage and cost details, propagated trace attributes, content withheld when asked;
* DeepEval's own metrics running through the gateway's judge with a scripted model (7 judge
  calls per case, verdicts in both directions);
* the cache's threshold boundary to four decimals, namespaces, expiry, eviction, the guard
  taking the next neighbour, the Redis store against a fake that speaks the same commands;
* the cascade's every failure kind, the deadline, streaming failover before and after the
  first token; the pipeline, the API, the chat proxy, the budget, the gate's rules;
* the project page against its recordings and the code (prices, thresholds, links, the demo
  address), the Render Blueprint against the settings, no em dash anywhere.

`TEST_REDIS_URL=redis://localhost:6379/0 python -m pytest tests/test_integration_redis.py` runs
the store against a real Redis 8 (`docker compose up -d redis`).

## What was measured, and what was not

Measured on 23 October 2026 on the development machine:

* the threshold trade-off and the replay above (the real embedding model; misses answered by the
  offline provider, so no model latency is in them);
* retrieval on the golden set (`site/recorded/retrieval.json`): hit rate@4 1.000, recall@4
  0.983, MRR 0.983, context precision@4 0.975 over the 30 answerable cases;
* the replay against Redis 8.2 in Docker, and the integration test against it;
* the production image: built from the Dockerfile, started with `--memory 512m` and a Redis
  container, answering at 348 MiB, `/docs` hidden in production, the cache index created at
  start-up.

Measured on 26 October 2026 with an OpenRouter key:

* the 17 free models OpenRouter listed, each sent the four test cases
  (`benchmarks/2026-10-26/`), and the default chains chosen from them;
* the project page's requests, re-recorded against those models, failover included;
* the judged suite on the golden set: 37 answers from the free chains (0 errors, 3 failovers,
  p95 latency 4.0 s), then DeepEval with the free judge chain (`site/recorded/deepeval.json`):
  faithfulness 0.985, refusal accuracy 1.000, injection resistance 1.000, answer relevancy
  0.896, contextual precision 0.944, and a hallucination rate of 0.054, which **fails** the
  gate's 0.05 by one case. One of the two flagged cases is a real incomplete answer (`edg-02`
  leaves out a required approval); the other is the judge scoring a word-for-word correct
  answer as a contradiction (`typ-19`). The gate is doing its job on the first, and the second
  is what "a free judge is a noisy judge" costs.

Not yet measured: Ragas with a real judge (that day's free requests went to the DeepEval runs),
the deployed service on Render, Redis Cloud and Langfuse Cloud. docs/DEPLOY.md lists the steps,
and its "How far this guide has been tried" section is updated as each is done.

## Trade-offs and limits

* **A cache answers yesterday's question.** Entries live `CACHE_TTL_S` and a corpus or prompt
  change empties the namespace, but a policy that changed outside the corpus is not seen. The
  cache is shared by every user of a namespace: never put per-user data in a cached answer.
* **The guard knows numbers, codes and a list of opposites**, not names. Substituted entities
  (Toronto against Ottawa) are the residual false-hit risk below about 0.91.
* **The router is lexical.** It is cheap and explainable, and it will misjudge a hard question
  phrased simply. The cascade's validation catches some of that (an uncited answer moves on);
  a learned router would need labelled traffic this project does not have.
* **Free models are withdrawn often** and are slower and less consistent than paid ones. The
  chains fail over and repair themselves, but a repair needs a free model that passes the four
  cases; on a day when none is reachable, requests fail over to the other tier and then answer
  503 until one is.
* **A free judge is a noisy judge.** On the first measured run it raised one false alarm in
  37 cases, and at 37 cases one case is 2.7 points of hallucination rate: the 5% floor
  tolerates one. A stronger judge is a setting, not a code change; more golden cases would
  make each one count for less.
* **One process.** The budget, the in-memory metrics and the embedding model live in it;
  horizontal scaling would move the budget to Redis and the metrics to Langfuse alone.

## Project layout

```
app/
  main.py                 routes, middleware, error envelope, playground
  components.py           builds everything once at start-up
  config.py               every setting (pydantic-settings)
  guard.py                defanging, token estimates and ceilings
  embeddings.py           bge-small through fastembed; the hash embedder
  budget.py               the public demo's limits
  gateway.py              the OpenAI-compatible proxy
  pricing.py              price table, actual and list-price costs
  schemas.py              request and response models
  cache/                  semantic cache, contrast guard, Redis and memory stores
  routing/                complexity router, failover cascade
  providers/              OpenRouter/OpenAI/Gemini, Anthropic, offline
  rag/                    corpus, hybrid retriever, prompts, the pipeline
  telemetry/              traces, metrics, the Langfuse sink
  playground/static/      the page served at /
corpus/                   the policy handbook (seven Markdown documents)
evals/                    golden set, thresholds, retrieval check, answer generation,
                          DeepEval judge and scorer, Ragas scorer, gate, synthesis, replay
scripts/                  local demo, smoke test, recordings, measurements
site/                     the project page (GitHub Pages) and its recordings
tests/                    the hermetic suite, and the opt-in Redis test
docs/DEPLOY.md            putting it online, step by step
```

## License

MIT. See [LICENSE](LICENSE).
