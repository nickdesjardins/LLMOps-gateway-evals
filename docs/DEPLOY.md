# Putting the showcase online, step by step

**Status (23 October 2026): not online yet.** Everything below has been prepared and checked
locally (see [How far this guide has been tried](#how-far-this-guide-has-been-tried)); none of
the steps that create an account, a repository or a service has been performed.

| What | Where | Address |
|---|---|---|
| The **project page**: recorded requests, the measured threshold, wakes the demo | GitHub Pages | <https://nickdesjardins.github.io/production-LLMOps-gateway-evals/> (after step 2) |
| The **live demo**: the API with its playground at `/` | Render (free web service, Docker) | `https://llmops-gateway.onrender.com/` (after step 5; Render may add letters) |
| The **semantic cache**: Redis 8 with vector search | Redis Cloud (free database, 30 MB), optional | not public |
| The **traces**: Langfuse | Langfuse Cloud (free Hobby plan), optional | not public |
| The **language models** | OpenRouter free models | not public |

The project page is the address to give to people. The live demo sleeps after 15 minutes
without a visit and needs about a minute to wake; the project page starts waking it as soon as
it opens.

In this guide, **the service name** is the part of the demo's address between `https://` and
`.onrender.com`. `render.yaml` asks for `llmops-gateway`; if Render adds letters to it, use the
name it gave wherever this guide shows that one.

## Where each secret goes

| Secret | This GitHub repository | Render | Your computer |
|---|---|---|---|
| OpenRouter key (`sk-or-v1-...`) | Actions secret `OPENROUTER_API_KEY`, read by the quality gate (step 7) | `OPENROUTER_API_KEY`, typed once in step 5 | `.env.local` |
| Redis Cloud URL (holds the database password) | **never** | `REDIS_URL`, typed once in step 5 | `.env.local`, only to test against it |
| Langfuse keys (`pk-lf-...`, `sk-lf-...`) | optional Actions secrets `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (the eval runs then appear in Langfuse) | typed once in step 5 | `.env.local` |
| Anthropic key, for a stronger judge (optional) | Actions secret `ANTHROPIC_API_KEY` | not needed | `.env.local` |

`tests.yml` and `pages.yml` use no secret at all. Never paste a key into `render.yaml`, a
commit, a chat or a terminal command. Start-up refuses a key that contains `://` or whitespace,
without echoing it, so pasting the Redis URL into the key box cannot send its password to a
model provider.

---

## Step 1: put the code on GitHub (10 minutes)

Create an empty **public** repository named `production-LLMOps-gateway-evals` on
<https://github.com/new> (no README, no licence, no `.gitignore`: the folder has them). Then, in
PowerShell, in the project folder:

```powershell
git init
```

```powershell
git add -A
```

Check that no private file is about to be published. The only line printed must be
`.env.example`:

```powershell
git ls-files | Select-String -Pattern "\.env"
```

`.gitignore` keeps `.env`, `.env.local`, the virtual environments and `evals/out/` out. If
anything else appears, stop and fix `.gitignore` before committing. Otherwise:

```powershell
git commit -m "Production LLMOps gateway: semantic cache, routing, tracing, evaluation in CI"
```

```powershell
git branch -M main
```

```powershell
git remote add origin git@github.com:nickdesjardins/production-LLMOps-gateway-evals.git
```

```powershell
git push -u origin main
```

The **Tests** and **Quality gate** workflows start at once. The gate runs only its retrieval
check on a push to main (the judged suite runs on pull requests), so it needs no secret yet.

## Step 2: turn on GitHub Pages (5 minutes)

1. Open the repository, then **Settings**, then **Pages**.
2. Under **Build and deployment > Source**, choose **GitHub Actions**. Ignore the templates
   GitHub suggests: the project has its own (`.github/workflows/pages.yml`).
3. Open the **Actions** tab, **Project page** in the left list, **Run workflow**, branch `main`,
   **Run workflow**. It runs `tests/test_site.py` first and publishes `site/` only if that
   passes.
4. When it is green, open <https://nickdesjardins.github.io/production-LLMOps-gateway-evals/>.
   Until step 6 the page says that the live demo is not online yet.

## Step 3 (optional): a Redis database with vector search (10 minutes)

Without it the cache lives in the service's memory and starts empty after every sleep. With it,
cached answers survive restarts and the trace says `"backend": "redis"`.

1. Sign in to <https://redis.io/try-free/> (Redis Cloud) and create a **free** database: cloud
   AWS, region **us-east-1** (Render's Virginia region is next door). The cache needs the
   Query Engine (vector search), which Redis Cloud databases include; the check in point 3
   proves it for yours.
2. Open the database, then **Connect**. Copy the connection URL with its password (it starts
   with `redis://` or `rediss://`). Paste it into `.env.local` as `REDIS_URL=`.
3. Check it from your computer (no model is called):

   ```powershell
   .\.venv\Scripts\python.exe -m evals.replay --redis-url "<the URL>" --requests 30 --thresholds 0.96 --out evals/out/redis-check.json
   ```

   It creates its own index, replays 30 questions, prints the hit counts and the lookup times,
   and deletes what it wrote. A database without vector search fails with "this Redis server has
   no vector search".

The free database holds 30 MB. `render.yaml` bounds the cache at 2,000 entries
(`CACHE_MAX_ENTRIES`), which fits; entries expire after a week (`CACHE_TTL_S`).

## Step 4 (optional): a Langfuse project (5 minutes)

1. Sign in to Langfuse Cloud and create an organization and a project. Note which region you
   chose: <https://cloud.langfuse.com> (EU) or <https://us.cloud.langfuse.com> (US).
   `render.yaml` sets the US one; change `LANGFUSE_BASE_URL` there if you chose the EU.
2. In the project, **Settings > API keys > Create new API keys**. Keep the public key
   (`pk-lf-...`) and the secret key (`sk-lf-...`) for step 5, and put them in `.env.local` too.
3. With them in `.env.local`, start the local demo (`scripts\run_local_demo.py`) and ask one
   question: a trace named `gateway.ask` appears in Langfuse within a few seconds, with one
   observation per step and the model call as a generation with tokens and cost.

## Step 5: create the live demo on Render (15 minutes)

1. Go to <https://dashboard.render.com>, **New > Blueprint**.
2. Find `production-LLMOps-gateway-evals` and click **Connect**. If it is not in the list, give
   Render's GitHub app access to it (GitHub **Settings > Applications > Render > Configure**).
3. **Blueprint name**: anything. **Branch**: `main`.
4. Render lists **one web service, `llmops-gateway`, plan Free**, Docker runtime. Check that it
   says Free.
5. Fill in the boxes Render shows, **only this once**:
   * `OPENROUTER_API_KEY`: the `sk-or-v1-...` key.
   * `REDIS_URL`: the URL from step 3, or leave it empty.
   * `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`: from step 4, or leave both empty.
6. **Deploy Blueprint**. The build installs the requirements and bakes the embedding model into
   the image (several minutes). The log should then show, in this order:
   * `Loading embedding model BAAI/bge-small-en-v1.5`;
   * with Redis, `Created the Redis vector index llmops-cache-idx-384 (HNSW, cosine, 384
     dimensions)` (only the first time);
   * `Cache: Redis vector index (HNSW, cosine), threshold 0.96. Provider: openrouter (...)`, or
     `Cache: in-process store, ...` without Redis;
   * with Langfuse, `Langfuse tracing on (...)`;
   * `Startup complete: llmops-gateway v1.0.0`.
7. Check it:
   * `https://llmops-gateway.onrender.com/livez` shows `"status":"alive"`;
   * `https://llmops-gateway.onrender.com/health` shows `"status":"ok"`, `"demo_mode":true`,
     `"configured":true` and the cache backend you expect;
   * `https://llmops-gateway.onrender.com/` opens the playground. Ask a sample, then its
     paraphrase: the second answer must say "cache hit (semantic)".

If a value was typed wrong: service page > **Environment** > edit > **Save and deploy**.

## Step 6: link the project page to the live demo (3 minutes)

`site/index.html` holds the demo's address in one place. Today the line reads:

```html
<a class="btn btn--primary btn--lg" id="demo-link" href="https://llmops-gateway.invalid/#ask">Open the live demo</a>
```

1. Replace `https://llmops-gateway.invalid` with the service's address, from `https://` up to
   `.onrender.com`. Keep `/#ask`.
2. Check it: `.\.venv\Scripts\python.exe -m pytest tests/test_site.py -q` must pass.
3. Commit and push. The **Project page** workflow publishes it (Render does not rebuild for a
   change in `site/`).
4. Put the address in this guide's table and in the README's.

## Step 7: give the quality gate its key, and make it required (10 minutes)

1. Repository **Settings > Secrets and variables > Actions > New repository secret**:
   `OPENROUTER_API_KEY`. A key made only for CI (credit limit 1 dollar, auto top-up off) can be
   withdrawn without touching the service's. A judged run is about 460 free requests: the
   account needs OpenRouter's 1,000-a-day allowance (a one-time 10 dollar credit purchase).
   Optionally add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (and the variable
   `LANGFUSE_BASE_URL`) to see the eval runs in Langfuse.
2. **Actions > Quality gate > Run workflow** (branch `main`, "Run the judged suite" ticked). It
   answers the 37 golden cases and judges them (20 to 40 minutes on free models). Open the run's
   summary: the gate's table, and any case that hallucinated, leaked or refused wrongly.
3. If the numbers are right, accept them as the baseline: download the artifact
   **quality-gate-reports**, unzip it into `evals/out/`, then

   ```powershell
   .\.venv\Scripts\python.exe -m evals.gate --retrieval evals/out/retrieval.json --deepeval evals/out/deepeval.json --ragas evals/out/ragas.json --write-baseline
   ```

   and copy the three reports into `site/recorded/` so the project page shows them. Commit
   `evals/baseline.json` and `site/recorded/*.json`.
4. **Settings > Rules > Rulesets > New branch ruleset**: target the default branch, enable
   **Require status checks to pass**, add the check **Gate** (from "Quality gate"), and
   **Tests**. A pull request whose gate fails can then not be merged.

If the free judge raises false alarms (a correct answer scored unfaithful), set the repository
variables `EVAL_JUDGE_PROVIDER=anthropic` and `EVAL_JUDGE_MODELS=claude-haiku-5-5` and the secret
`ANTHROPIC_API_KEY`: a stronger judge, a few cents per run.

## Step 8: record the requests for the project page (5 minutes)

With `OPENROUTER_API_KEY` in `.env.local`:

```powershell
.\.venv\Scripts\python.exe scripts\openrouter_smoke.py
```

Every model of both chains should answer. Replace any that fails in `LIGHT_MODELS` /
`HEAVY_MODELS` (ids ending in `:free` from <https://openrouter.ai/models>) and run it again.
Then:

```powershell
.\.venv\Scripts\python.exe scripts\record_runs.py
```

It sends the ten scenarios of the page through the real app (about 12 free requests) and adds
the failover scenario (a model id that does not exist, then the real chain). Run
`.\.venv\Scripts\python.exe -m pytest tests/test_site.py -q`, then commit
`site/recorded/runs.json` and push.

---

## Good to know about the free plans

* **Sleeping.** The demo sleeps after 15 minutes without a visit and takes about a minute to
  wake; the project page wakes it. The in-process cache, the metrics and the recent traces start
  empty after each wake; Redis and Langfuse keep theirs.
* **Memory.** The image runs in 512 MB: measured at 348 MiB with the embedding model loaded and
  a Redis connection, in a container limited to 512 MB on the development machine.
* **Hours.** A workspace gets 750 free instance hours a month, shared by every free service of
  the workspace (Projects 1 to 3 included). Sleeping services use none.
* **OpenRouter's free allowance** is per account and per UTC day (50 requests without credits,
  1,000 after a one-time 10 dollar purchase), shared by the demo, the quality gate, local runs
  and the other projects on the same account. `render.yaml` caps the demo at 300 model calls a
  day (`MODEL_CALLS_PER_DAY`); cache hits are not counted.
* **Free endpoints may keep and train on what they are sent.** The handbook is fictional; keep
  it that way.

## How far this guide has been tried

**Verified (23 October 2026), on the development machine:**

* The hermetic suite and `ruff check` / `ruff format --check` pass.
* The Docker image builds from the Dockerfile (781 MB), starts with `--memory 512m`, loads the
  baked embedding model without a download, connects to a Redis 8.2 container, creates the
  vector index and answers: a semantic hit through Redis in 15 ms, the contrast guard refusing
  the international look-alike, `/docs` hidden with `ENVIRONMENT=production`, 348 MiB in use.
* `tests/test_integration_redis.py` passes against Redis 8.2 (`docker compose up -d redis`), and
  the cache replay against it gives the same hits as the in-memory store.
* The playground and the project page were opened in a browser and exercised (offline provider):
  streaming, the trace waterfall, the metrics, the similarity lab, the threshold slider.
* `render.yaml`, the Dockerfile and `.env.example` are compared with `app/config.py` by
  `tests/test_site.py`.

**Not verified:** every step above that creates something (the repository, Pages, Redis Cloud,
Langfuse Cloud, the Render service, the repository secrets and the ruleset); answers from the
free models (the default chains are untested except `nemotron-3-super`, step 8 checks them);
the judged suite on GitHub Actions (DeepEval and Ragas with a real judge); the Langfuse export
to a real Langfuse (the SDK was exercised with an in-memory exporter only).

Update both lists when a step is done.

---

## What protects the public demo

| Risk | Protection |
|---|---|
| Someone spends the day's free requests | Per visitor: 20 requests a minute, 30 model calls a day. Everyone together: 12 model calls a minute, 300 a day, 3 at once. A failover attempt counts as a call. Cache hits are not counted against the model budget, so they keep working when it is spent. |
| Someone pretends to be many visitors | The address comes from `CF-Connecting-IP`, which Cloudflare (in front of Render) overwrites; IPv6 visitors are grouped by their /64; addresses are kept only as salted hashes. |
| A typo makes the service call a paid model | `OPENROUTER_FREE_ONLY=true` refuses any id without `:free` at start-up; in demo mode a request's `model` names a tier, never a model id. |
| A runaway answer | 350 / 900 output tokens per tier, a 6,000-token input ceiling, a 32 KB body limit, questions capped at 500 characters. |
| A prompt injection | Defanged and flagged; the question travels as tagged data under a static system prompt; flagged answers are never cached. |
| A wrong answer served from the cache | The namespace isolates prompts and corpus versions; the contrast guard refuses contradicting numbers, codes and opposite terms; the threshold trade-off is measured. |
| An error message leaks a key or a provider's detail | Clients get a type, a sentence and an incident id; validation errors never echo the input; the provider's text stays in the log. |

## Day-to-day

**Run the showcase locally:** `.\.venv\Scripts\python.exe scripts\run_local_demo.py` (demo mode,
port 8013; `--offline` calls no model).

**Change the threshold or the limits:** edit `render.yaml` and push; the service rebuilds.

**Change a prompt or a document:** the cache namespace changes with it, so the cache starts
empty; re-record the page (step 8) and let the quality gate judge the change on its pull
request. `tests/test_site.py` fails until `site/recorded/runs.json` is re-recorded with the new
prompt version.

**Replace the OpenRouter key:** new key in `.env.local`, run the smoke script, then Render
**Environment** > `OPENROUTER_API_KEY` > **Save and deploy**, and the repository secret.

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| The deploy fails with "OPENROUTER_API_KEY does not look like a key" | Another line, or the Redis URL, was pasted into the key box | Render **Environment**: paste the `sk-or-v1-...` value, **Save and deploy**. |
| The log says "Redis cannot hold the cache (this Redis server has no vector search ...)" | The database has no Query Engine | Use a Redis 8 (or Redis Stack) database; the service runs on the in-process cache meanwhile. |
| The log says "Redis did not answer (...)" | A wrong URL or password, or the database is paused | Check the URL on Redis Cloud's **Connect** page. The service runs on the in-process cache meanwhile. |
| `/health` says `"status": "degraded"` with `"configured": false` | No provider key | Set `OPENROUTER_API_KEY`. Cached answers are still served. |
| Requests answer 503 "Every model is busy or failing right now" | Every model of both chains failed (the log names each model and the failure kind) | Wait a minute. If it persists, run `scripts\openrouter_smoke.py` and replace the failing ids. |
| Requests answer 502 "No configured model could take this request" | Every model answered 404 or 400: the chain is gone | Pick new `:free` ids, set `LIGHT_MODELS` / `HEAVY_MODELS` in Render's **Environment**. |
| Requests answer 429 | A budget rule was reached; the message says which, `Retry-After` says when | Nothing to fix; raise the limits in `render.yaml` if the allowance allows it. |
| The quality gate fails with "the report evals/out/deepeval.json was not produced" | The judged suite could not run: no `OPENROUTER_API_KEY` secret, a pull request from a fork, or the judge failed | Read the failed job's log. For a fork, a maintainer runs the workflow on a branch of the repository. |
| The gate fails on faithfulness with answers that look right | The judge is noisy | Read the cases in the run summary; try the Anthropic judge (step 7). |
| The project page still says the demo is not online | Step 6 not done, or the service failing to start | Open `/livez` on the service; then fix the link or read Render's **Logs**. |
