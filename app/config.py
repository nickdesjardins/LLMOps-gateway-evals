"""Environment-driven configuration (pydantic-settings).

Every number the gateway works with lives here: the similarity threshold of the semantic cache,
the token ceilings, the router's cut-off, the model chains, the demo budget. Values come from
the process environment first, then from ``.env`` and ``.env.local`` (both git-ignored;
``.env.example`` documents every name).

Four ways to run it:

* **Free (the default):** ``LLM_PROVIDER=openrouter`` with two chains of ``:free`` models, a
  light one and a heavy one. Cost is zero; the telemetry still prices every call at the list
  price of a paid reference model, so the token economics stay visible.
* **Paid, drop-in:** ``LLM_PROVIDER=anthropic`` (Claude through the official SDK, with prompt
  caching on the static system prefix), ``openai`` or ``gemini`` (both through the Chat
  Completions API).
* **Offline:** ``LLM_PROVIDER=offline`` answers with the best retrieved passage and calls no
  model at all. For a first look on a laptop, for the cache replay and for the tests.
* **Without Redis:** ``CACHE_BACKEND=memory`` keeps the semantic cache in the process. Redis is
  used when ``REDIS_URL`` is set and the server has vector search (Redis 8, or Redis Stack).
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LlmProvider = Literal["openrouter", "openai", "anthropic", "gemini", "offline"]
EmbeddingProvider = Literal["fastembed", "hash"]
CacheBackend = Literal["auto", "redis", "memory", "off"]
InjectionPolicy = Literal["defang", "block"]
Reasoning = Literal["default", "off", "low", "medium", "high"]
AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
Tier = Literal["light", "heavy"]

#: Free OpenRouter models per tier, chosen from OpenRouter's list on 23 October 2026. The light
#: chain holds small models with few active parameters; the heavy chain holds the largest free
#: models that follow instructions well. Each chain mixes two upstream providers, so one outage
#: cannot take out a tier. nvidia/nemotron-3-super-120b-a12b was measured in Project 3 (8 of 9
#: correct, about 13 s with reasoning off); the other three are listed, not yet measured here:
#: scripts/openrouter_smoke.py checks every model of both chains before a recording.
DEFAULT_OPENROUTER_LIGHT = "google/gemma-4-26b-a4b-it:free,nvidia/nemotron-3.5-lightning:free"
DEFAULT_OPENROUTER_HEAVY = "nvidia/nemotron-3-super-120b-a12b:free,google/gemma-4-31b-it:free"

#: The current Claude models: the cheapest for the light tier, the default frontier model for
#: the heavy one.
DEFAULT_ANTHROPIC_LIGHT = "claude-haiku-5-5"
DEFAULT_ANTHROPIC_HEAVY = "claude-opus-5-5"

#: The offline provider has one "model": the best retrieved passage, quoted.
OFFLINE_MODEL = "offline/extractive"

#: Placeholder values a template may ship. Treated as "not configured".
_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {"your-api-key", "sk-or-v1-...", "sk-...", "sk-ant-...", "changeme", "replace-me", "xxx"}
)


def parse_model_chain(raw: str) -> list[str]:
    """Split an ordered, comma-separated chain of model ids (trimmed, de-duplicated, non-empty)."""
    models: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in models:
            models.append(model)
    return models


def is_free_model(model_id: str) -> bool:
    """True for an OpenRouter id that cannot cost money (the ``:free`` suffix)."""
    return model_id.endswith(":free")


def _clean_key(value: SecretStr | None, name: str) -> SecretStr | None:
    """None for an empty or placeholder key; an error for something that is not a key at all."""
    if value is None:
        return None
    text = value.get_secret_value().strip().strip("'\"")
    if not text or text.lower() in _PLACEHOLDER_KEYS:
        return None
    # A connection string or a second line pasted into the key box must never be sent to a
    # model provider. The message never repeats the value.
    if "://" in text or any(ch.isspace() for ch in text):
        raise ValueError(f"{name} does not look like a key (was another line pasted along with it?)")
    return SecretStr(text)


class Settings(BaseSettings):
    """Runtime configuration, resolved once per process (see :func:`get_settings`)."""

    model_config = SettingsConfigDict(
        # Both are read; values in .env.local override .env. Neither is committed.
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
        env_parse_none_str="null",
        # A rejected value (a key, a Redis URL with its password) is never echoed into a log.
        hide_input_in_errors=True,
        protected_namespaces=("settings_",),
    )

    # ------------------------------------------------------------------ application
    service_name: str = Field(default="llmops-gateway", description="Reported by /health.")
    environment: Environment = Field(default="development", description="production hides /docs.")
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    gateway_api_keys: SecretStr | None = Field(
        default=None,
        description=(
            "Comma-separated keys accepted in the Authorization: Bearer header of "
            "/v1/chat/completions. Outside demo mode that route is closed without one."
        ),
    )
    cors_allow_origins: str = Field(
        default="", description="Comma-separated browser origins allowed to call the API."
    )
    playground_enabled: bool = Field(default=True, description="Serve the playground at /.")
    max_body_bytes: int = Field(default=32_768, ge=1_024, le=1_048_576)

    # ------------------------------------------------------------------ public demo
    demo_mode: bool = Field(
        default=False,
        description=(
            "Public showcase: routes open, every model call counted against the budget below, "
            "model names in requests ignored (the router always decides)."
        ),
    )
    trust_proxy_headers: bool = Field(
        default=False, description="Read the client address from CLIENT_IP_HEADER (Render)."
    )
    client_ip_header: str = Field(default="CF-Connecting-IP")
    requests_per_minute_per_client: int = Field(default=20, ge=1, le=10_000)
    model_calls_per_day_per_client: int = Field(default=40, ge=1, le=100_000)
    model_calls_per_minute: int = Field(
        default=15, ge=1, le=10_000, description="All visitors together (OpenRouter allows 20)."
    )
    model_calls_per_day: int = Field(default=600, ge=1, le=1_000_000)
    max_inflight: int = Field(default=3, ge=1, le=100, description="Model calls running at once.")
    question_max_chars: int = Field(default=500, ge=20, le=8_000)

    # ------------------------------------------------------------------ providers
    llm_provider: LlmProvider = Field(default="openrouter")
    openrouter_api_key: SecretStr | None = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_free_only: bool = Field(
        default=True, description="Refuse any OpenRouter model id that does not end in :free."
    )
    openrouter_reasoning: Reasoning = Field(
        default="off",
        description=(
            "OpenRouter's unified reasoning control. Thinking tokens count against the output "
            "ceiling, so a model that thinks first can run out of budget before it answers."
        ),
    )
    openrouter_app_name: str | None = Field(default="llmops-gateway")
    openai_api_key: SecretStr | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Gemini's OpenAI-compatible endpoint.",
    )
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_effort_light: AnthropicEffort = Field(default="low")
    anthropic_effort_heavy: AnthropicEffort = Field(default="medium")
    light_models: str = Field(
        default="", description="Ordered chain for the light tier. Blank: the provider default."
    )
    heavy_models: str = Field(
        default="", description="Ordered chain for the heavy tier. Blank: the provider default."
    )
    cross_tier_fallback: bool = Field(
        default=True,
        description="When every model of a tier fails, try the other tier before giving up.",
    )
    model_timeout_s: float = Field(default=45.0, gt=0, le=600)
    request_deadline_s: float = Field(
        default=90.0, gt=0, le=1_800, description="Whole cascade, every attempt included."
    )

    # ------------------------------------------------------------------ token ceilings
    max_output_tokens_light: int = Field(default=350, ge=16, le=32_000)
    max_output_tokens_heavy: int = Field(default=900, ge=16, le=64_000)
    max_input_tokens: int = Field(
        default=6_000,
        ge=256,
        le=1_000_000,
        description="Estimated prompt size above which a request is refused before any call.",
    )

    # ------------------------------------------------------------------ embeddings
    embedding_provider: EmbeddingProvider = Field(
        default="fastembed",
        description="fastembed: a real local model. hash: deterministic token hashing, no model.",
    )
    fastembed_model: str = Field(default="BAAI/bge-small-en-v1.5")
    fastembed_cache_dir: str | None = Field(default=None, description="Model directory.")
    fastembed_threads: int | None = Field(default=None, ge=1, le=64)

    # ------------------------------------------------------------------ semantic cache
    cache_backend: CacheBackend = Field(
        default="auto", description="auto: Redis when REDIS_URL is set, memory otherwise."
    )
    redis_url: SecretStr | None = Field(default=None, description="redis:// or rediss:// URL.")
    redis_index_name: str = Field(default="llmops-cache-idx", pattern=r"^[A-Za-z0-9_:-]{1,64}$")
    redis_key_prefix: str = Field(default="llmops:cache:", pattern=r"^[A-Za-z0-9_:-]{1,64}$")
    cache_similarity_threshold: float = Field(
        default=0.96,
        ge=0.5,
        le=1.0,
        description="Cosine similarity at or above which a stored answer is served.",
    )
    cache_contrast_guard: bool = Field(
        default=True,
        description="Refuse a semantic hit whose numbers, codes or opposite terms contradict the question.",
    )
    cache_ttl_s: int = Field(default=86_400, ge=60, le=30 * 86_400)
    cache_max_entries: int = Field(default=5_000, ge=10, le=1_000_000)

    # ------------------------------------------------------------------ routing and RAG
    router_heavy_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Complexity score from which the heavy tier runs."
    )
    retrieval_top_k: int = Field(default=4, ge=1, le=12)
    injection_policy: InjectionPolicy = Field(
        default="defang",
        description="defang: neutralize and answer. block: refuse a request that looks like one.",
    )

    # ------------------------------------------------------------------ telemetry
    langfuse_public_key: SecretStr | None = Field(default=None)
    langfuse_secret_key: SecretStr | None = Field(default=None)
    langfuse_base_url: str = Field(default="https://cloud.langfuse.com")
    langfuse_capture_content: bool = Field(
        default=True, description="Send question and answer text to Langfuse, not only numbers."
    )
    traces_kept: int = Field(default=200, ge=10, le=10_000, description="Recent traces in memory.")
    reference_model_light: str = Field(
        default="claude-haiku-5-5", description="Prices the light tier when the call itself is free."
    )
    reference_model_heavy: str = Field(default="claude-opus-5-5")
    model_prices: str = Field(
        default="",
        description=(
            'Extra prices, JSON: {"model-id": [input, output, cache_read]} in USD per million '
            "tokens. Adds to or overrides app/pricing.py."
        ),
    )

    # ------------------------------------------------------------------- validators
    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_case_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator(
        "openrouter_api_key",
        "openai_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="after",
    )
    @classmethod
    def _keys_look_like_keys(cls, value: SecretStr | None, info: object) -> SecretStr | None:
        name = getattr(info, "field_name", "key").upper()
        return _clean_key(value, name)

    @field_validator("redis_url", mode="after")
    @classmethod
    def _redis_url_scheme(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        text = value.get_secret_value().strip().strip("'\"")
        if not text:
            return None
        if not text.startswith(("redis://", "rediss://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return SecretStr(text)

    @field_validator("model_prices")
    @classmethod
    def _prices_are_json(cls, value: str) -> str:
        if not value.strip():
            return ""
        try:
            table = json.loads(value)
        except ValueError as exc:
            raise ValueError("MODEL_PRICES must be a JSON object") from exc
        if not isinstance(table, dict):
            raise ValueError("MODEL_PRICES must be a JSON object")
        for model, prices in table.items():
            if not isinstance(prices, list) or len(prices) not in (2, 3):
                raise ValueError(f"MODEL_PRICES[{model!r}] must be [input, output, cache_read?]")
            try:
                if any(Decimal(str(price)) < 0 for price in prices):
                    raise ValueError(f"MODEL_PRICES[{model!r}] has a negative price")
            except InvalidOperation as exc:
                raise ValueError(f"MODEL_PRICES[{model!r}] holds a value that is not a number") from exc
        return value

    @model_validator(mode="after")
    def _chains_are_usable(self) -> Settings:
        light, heavy = self.chain("light"), self.chain("heavy")
        if not light or not heavy:
            raise ValueError(
                f"LIGHT_MODELS and HEAVY_MODELS must name the models to use with "
                f"LLM_PROVIDER={self.llm_provider} (ids from the provider's model list)"
            )
        if self.llm_provider == "openrouter" and self.openrouter_free_only:
            paid = [model for model in light + heavy if not is_free_model(model)]
            if paid:
                raise ValueError(f"OPENROUTER_FREE_ONLY is on and these ids do not end in :free: {paid}")
        if self.max_output_tokens_light > self.max_output_tokens_heavy:
            raise ValueError("MAX_OUTPUT_TOKENS_LIGHT cannot exceed MAX_OUTPUT_TOKENS_HEAVY")
        if bool(self.langfuse_public_key) != bool(self.langfuse_secret_key):
            raise ValueError("Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or neither")
        return self

    # ---------------------------------------------------------------------- helpers
    def chain(self, tier: Tier) -> list[str]:
        """The ordered models of one tier, after provider defaults."""
        raw = self.light_models if tier == "light" else self.heavy_models
        if raw.strip():
            return parse_model_chain(raw)
        if self.llm_provider == "openrouter":
            return parse_model_chain(
                DEFAULT_OPENROUTER_LIGHT if tier == "light" else DEFAULT_OPENROUTER_HEAVY
            )
        if self.llm_provider == "anthropic":
            return [DEFAULT_ANTHROPIC_LIGHT if tier == "light" else DEFAULT_ANTHROPIC_HEAVY]
        if self.llm_provider == "offline":
            return [OFFLINE_MODEL]
        return []  # openai and gemini: the operator names the models

    def max_output_tokens(self, tier: Tier) -> int:
        return self.max_output_tokens_light if tier == "light" else self.max_output_tokens_heavy

    def provider_key(self) -> SecretStr | None:
        """The key of the configured provider (None for offline or when it is missing)."""
        return {
            "openrouter": self.openrouter_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "offline": None,
        }[self.llm_provider]

    def provider_configured(self) -> bool:
        return self.llm_provider == "offline" or self.provider_key() is not None

    def resolved_cache_backend(self) -> Literal["redis", "memory", "off"]:
        if self.cache_backend == "auto":
            return "redis" if self.redis_url is not None else "memory"
        return self.cache_backend

    def langfuse_enabled(self) -> bool:
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    def gateway_keys(self) -> list[str]:
        if self.gateway_api_keys is None:
            return []
        return [k.strip() for k in self.gateway_api_keys.get_secret_value().split(",") if k.strip()]

    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    def extra_prices(self) -> dict[str, list[str]]:
        if not self.model_prices.strip():
            return {}
        return {
            model: [str(price) for price in prices] for model, prices in json.loads(self.model_prices).items()
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, read once."""
    settings = Settings()
    logging.getLogger(__name__).debug("Settings loaded for %s", settings.service_name)
    return settings
