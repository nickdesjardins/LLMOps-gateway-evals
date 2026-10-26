"""The published pages and the deployment files are checked against their sources.

Covers the project page (``site/``, GitHub Pages), the playground (served by the API at ``/``),
the recordings in ``site/recorded/``, the documentation, and the deployment files that must
agree with the settings. Hermetic: no network, no Redis, no model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.pricing import BUILTIN_PRICES
from app.rag.prompts import PROMPT_VERSION
from app.schemas import AskResponse, ChatCompletionResponse

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
RECORDED = SITE / "recorded"
PLAYGROUND = ROOT / "app" / "playground" / "static"
PAGES = [SITE / "index.html", PLAYGROUND / "index.html"]
SECRET_MARKERS = ("sk-or-", "sk-ant-", "Bearer ", "rediss://", "pk-lf-", "sk-lf-")
REPOSITORY = "https://github.com/nickdesjardins/LLMOps-gateway-evals"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _recorded(name: str) -> dict:
    path = RECORDED / name
    if not path.is_file():
        pytest.skip(f"{name} is not recorded yet")
    return json.loads(_read(path))


# ------------------------------------------------------------------------- the pages


@pytest.mark.parametrize("page", PAGES, ids=["site", "playground"])
def test_pages_declare_a_policy_and_use_no_inline_code(page: Path) -> None:
    html = _read(page)
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert csp, "every page declares a content security policy"
    policy = csp.group(1)
    for directive in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "connect-src 'self'"):
        assert directive in policy, directive
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "no inline scripts"
    assert not re.search(r"\son[a-z]+\s*=", html), "no inline event handlers"
    assert "<style" not in html and not re.search(r'\sstyle="', html), "no inline styles"


def test_scripts_never_inject_html() -> None:
    for path in (SITE / "assets" / "site.js", PLAYGROUND / "assets" / "app.js"):
        script = _read(path)
        assert (
            "innerHTML" not in script
            and "insertAdjacentHTML" not in script
            and "document.write" not in script
        ), path.name


def test_the_project_page_may_load_the_demo_icon_and_nothing_else_from_outside() -> None:
    policy = re.search(r'content="([^"]+)"', _read(SITE / "index.html")).group(1)  # type: ignore[union-attr]
    assert "img-src 'self' https://*.onrender.com" in policy
    assert "onrender.com" not in _read(PLAYGROUND / "index.html")


def test_the_live_demo_address_is_written_once_and_is_well_formed() -> None:
    html = _read(SITE / "index.html")
    links = re.findall(r'id="demo-link" href="([^"]+)"', html)
    assert len(links) == 1
    assert re.fullmatch(r"https://[a-z0-9-]+\.(onrender\.com|invalid)/#ask", links[0]), links[0]
    assert html.count(".onrender.com/") <= 1
    script = _read(SITE / "assets" / "site.js")
    icon = re.search(r'new URL\("(/playground/assets/[^"]+)"', script)
    assert icon and (PLAYGROUND / "assets" / icon.group(1).rsplit("/", 1)[1]).is_file()
    assert "GIVE_UP_MS = 180000" in script  # three minutes, as the page says
    assert 'id="ask"' in _read(PLAYGROUND / "index.html")  # the #ask anchor the link opens


def test_assets_exist() -> None:
    html = _read(SITE / "index.html")
    for relative in re.findall(r'(?:href|src)="((?!https?:|#|\./)[^"]+)"', html):
        if relative.startswith("recorded/"):
            continue  # recordings degrade gracefully when absent
        assert (SITE / relative).is_file(), relative
    found = re.findall(r'(?:href|src)="(/playground/assets/[^"]+)"', _read(PLAYGROUND / "index.html"))
    assert len(found) >= 4
    for absolute in found:
        assert (PLAYGROUND / "assets" / absolute.rsplit("/", 1)[1]).is_file(), absolute


@pytest.mark.parametrize("name", ["theme.js", "favicon.svg"])
def test_shared_assets_are_the_same_bytes_on_both_pages(name: str) -> None:
    assert (SITE / "assets" / name).read_bytes() == (PLAYGROUND / "assets" / name).read_bytes(), name


def test_code_links_point_to_files_that_exist() -> None:
    html = _read(SITE / "index.html")
    linked = set(
        re.findall(r"blob/main/((?:app|evals|scripts|tests|docs)/[A-Za-z_/.]+\.(?:py|json|md))", html)
    )
    assert len(linked) >= 7
    for path in linked:
        assert (ROOT / path).is_file(), path
    assert f"{REPOSITORY}#readme" in html and "blob/main/docs/DEPLOY.md" in html


def test_pages_link_the_repository_and_the_sibling_projects() -> None:
    assert REPOSITORY in _read(PLAYGROUND / "index.html")
    html = _read(SITE / "index.html")
    assert REPOSITORY in html
    for sibling in ("Invoice-Extraction-Service", "hybrid-knowledge-rag", "autonomous-operation-marketing"):
        assert f"https://nickdesjardins.github.io/{sibling}/" in html


def test_the_page_prices_match_the_price_book() -> None:
    script = _read(SITE / "assets" / "site.js")
    for tier, model in (("light", "claude-haiku-5-5"), ("heavy", "claude-opus-5-5")):
        match = re.search(rf'{tier}: {{ model: "{model}", input: ([\d.]+), output: ([\d.]+) }}', script)
        assert match, tier
        assert float(match.group(1)) == float(BUILTIN_PRICES[model].input)
        assert float(match.group(2)) == float(BUILTIN_PRICES[model].output)


def test_the_ragas_judge_default_is_the_service_s() -> None:
    from app.config import DEFAULT_OPENROUTER_JUDGE

    match = re.search(r'^DEFAULT_MODELS = "([^"]+)"', _read(ROOT / "evals" / "score_ragas.py"), re.MULTILINE)
    assert match and match.group(1) == DEFAULT_OPENROUTER_JUDGE


def test_the_published_thresholds_are_the_gate_s() -> None:
    assert json.loads(_read(RECORDED / "thresholds.json")) == json.loads(
        _read(ROOT / "evals" / "thresholds.json")
    )


# --------------------------------------------------------------- no em dash anywhere


def test_no_em_dash_in_pages_docs_prompts_or_code() -> None:
    em_dash = chr(0x2014)  # spelled as a code point: the character itself may not appear
    checked = 0
    for pattern in (
        "*.md",
        "site/**/*.*",
        "app/**/*.*",
        "evals/**/*.*",
        "scripts/*.py",
        "corpus/*.md",
        "docs/*.md",
        ".github/workflows/*.yml",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.txt",
        "Dockerfile",
        ".env.example",
    ):
        for path in ROOT.glob(pattern):
            if (
                path.is_file()
                and path.suffix not in {".pyc", ".svg"}
                and "__pycache__" not in path.parts
                and "out" not in path.parts
            ):
                assert em_dash not in path.read_text(encoding="utf-8"), path.relative_to(ROOT)
                checked += 1
    assert checked > 60


# -------------------------------------------------------------------- deployment


def test_the_render_blueprint_agrees_with_the_settings() -> None:
    blueprint = _read(ROOT / "render.yaml")

    def value(key: str) -> str:
        match = re.search(rf"- key: {key}\n\s+value: \"?([^\"\n#]+?)\"?\s*(?:#.*)?\n", blueprint)
        assert match, key
        return match.group(1).strip()

    assert "plan: free" in blueprint and "runtime: docker" in blueprint
    assert "healthCheckPath: /livez" in blueprint
    assert value("DEMO_MODE") == "true" and value("LLM_PROVIDER") == "openrouter"
    assert value("TRUST_PROXY_HEADERS") == "true" and value("CLIENT_IP_HEADER")
    assert float(value("CACHE_SIMILARITY_THRESHOLD")) == Settings(_env_file=None).cache_similarity_threshold
    for secret in ("OPENROUTER_API_KEY", "REDIS_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        assert re.search(rf"- key: {secret}\n\s+sync: false", blueprint), secret
    for marker in SECRET_MARKERS:
        assert marker not in blueprint, marker
    fields = {name.upper() for name in Settings.model_fields}
    for key in re.findall(r"- key: ([A-Z_]+)", blueprint):
        assert key in fields, key
    dockerfile = _read(ROOT / "Dockerfile")
    assert value("FASTEMBED_MODEL") in dockerfile  # the baked model is the configured one
    assert "COPY corpus ./corpus" in dockerfile
    assert "corpus" not in _read(ROOT / ".dockerignore")


def test_the_env_template_names_only_real_settings() -> None:
    fields = {name.upper() for name in Settings.model_fields}
    eval_only = {
        "EVAL_JUDGE_PROVIDER",
        "EVAL_JUDGE_MODELS",
        "EVAL_JUDGE_MIN_INTERVAL_S",
        "RAGAS_JUDGE_MODELS",
        "RAGAS_MIN_INTERVAL_S",
    }
    names = re.findall(r"^([A-Z_]+)=", _read(ROOT / ".env.example"), re.MULTILINE)
    assert len(names) > 40
    for name in names:
        assert name in fields or name in eval_only, name


def test_gitignore_protects_local_secrets() -> None:
    lines = _read(ROOT / ".gitignore").splitlines()
    assert ".env" in lines and ".env.*" in lines and "!.env.example" in lines
    dockerignore = _read(ROOT / ".dockerignore").splitlines()
    assert ".env" in dockerignore and ".env.*" in dockerignore


def test_the_quality_gate_workflow_enforces_the_specification() -> None:
    workflow = _read(ROOT / ".github" / "workflows" / "quality-gate.yml")
    assert "pull_request:" in workflow
    assert "python -m evals.gate" in workflow and "if: ${{ !cancelled() }}" in workflow
    assert "requirements-ragas.txt" in workflow and "requirements-eval.txt" in workflow
    thresholds = json.loads(_read(ROOT / "evals" / "thresholds.json"))
    assert thresholds["deepeval"]["hallucination_rate_max"] == 0.05
    assert thresholds["deepeval"]["faithfulness_min"] == 0.95


# ------------------------------------------------------------------ the recordings


def test_recorded_runs_are_real_api_responses() -> None:
    record = _recorded("runs.json")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["recorded_at"])
    assert record["stack"]["prompt_version"] == PROMPT_VERSION, "re-record after changing the prompt"
    ids = [run["id"] for run in record["runs"]]
    assert len(ids) == len(set(ids)) >= 8
    for run in record["runs"]:
        assert run["status"] == 200, run["id"]
        if run["endpoint"] == "POST /v1/chat/completions":
            ChatCompletionResponse.model_validate(run["response"])
        else:
            AskResponse.model_validate(run["response"])
        assert run["trace"] and run["trace"]["spans"], run["id"]
    blob = json.dumps(record)
    for marker in SECRET_MARKERS:
        assert marker not in blob, marker


def test_the_recorded_story_holds() -> None:
    runs = {run["id"]: run for run in _recorded("runs.json")["runs"]}
    assert runs["exact-hit"]["response"]["cache"]["match"] == "exact"
    assert runs["semantic-hit"]["response"]["cache"]["match"] == "semantic"
    blocked = runs["guard-block"]["response"]["cache"]
    assert (
        blocked["outcome"] == "miss" and blocked["similarity"] >= blocked["threshold"] and blocked["blocked"]
    )
    assert runs["heavy"]["response"]["tier"] == "heavy"
    assert runs["refusal"]["response"]["refused"] is True
    assert runs["injection"]["response"]["injection"]["detected"] is True
    assert "PWNED" not in runs["injection"]["response"]["answer"]
    assert runs["chat-hit"]["response"]["x_gateway"]["cache"]["outcome"] == "hit"
    events = [e["event"] for e in runs["stream"]["events"]]
    assert events[0] == "meta" and events[-1] == "done" and "delta" in events
    if "failover" in runs:
        attempts = runs["failover"]["response"]["attempts"]
        assert attempts[0]["outcome"] in {"not_found", "bad_request"} and attempts[-1]["outcome"] == "ok"


def test_the_threshold_measurement_is_consistent() -> None:
    data = _recorded("cache-threshold.json")
    assert data["same_pairs"] == sum(1 for p in data["pairs"] if p["kind"] == "same")
    assert [s["threshold"] for s in data["sweep"]] == sorted(s["threshold"] for s in data["sweep"])
    rates = [s["true_hit_rate"] for s in data["sweep"]]
    assert rates == sorted(rates, reverse=True), "a higher threshold can only hit less"
    default = next(s for s in data["sweep"] if s["threshold"] == data["default_threshold"])
    assert default["false_hit_rate"] <= default["false_hit_rate_unguarded"]
    assert data["guard_blocks_same"] == 0


@pytest.mark.parametrize("name", ["replay.json", "replay-redis.json"])
def test_the_replay_adds_up(name: str) -> None:
    data = _recorded(name)
    for run in data["runs"]:
        assert run["hits"] + run["misses"] == run["requests"] == data["workload"]["requests"]
        assert run["exact_hits"] + run["semantic_hits"] == run["hits"]
        assert 0 <= run["cost_reduction"] <= 1


def test_redis_and_memory_replays_agree() -> None:
    memory, redis = _recorded("replay.json"), _recorded("replay-redis.json")
    assert redis["workload"]["cache_backend"] == "redis"
    for a, b in zip(memory["runs"], redis["runs"], strict=True):
        assert (a["threshold"], a["hits"], a["false_hits"]) == (b["threshold"], b["hits"], b["false_hits"])


def test_the_routing_and_retrieval_records_match_the_golden_set() -> None:
    from evals.schema import load_golden

    golden = load_golden()
    routing = _recorded("routing.json")
    assert [r["id"] for r in routing["rows"]] == [c.id for c in golden]
    assert routing["tiers"]["light"] + routing["tiers"]["heavy"] == len(golden)
    retrieval = _recorded("retrieval.json")
    assert retrieval["summary"]["cases"] == sum(1 for c in golden if c.expected_behavior == "answer")
