"""Settings: defaults, provider chains, and the refusals that protect keys at start-up."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import (
    DEFAULT_ANTHROPIC_HEAVY,
    DEFAULT_ANTHROPIC_LIGHT,
    Settings,
    is_free_model,
    parse_model_chain,
)
from tests.conftest import make_settings


def test_defaults_match_the_specification() -> None:
    settings = Settings(_env_file=None)
    assert settings.cache_similarity_threshold == 0.96
    assert settings.cache_contrast_guard is True
    assert settings.router_heavy_threshold == 0.5
    assert settings.max_output_tokens_light < settings.max_output_tokens_heavy
    assert settings.llm_provider == "openrouter"
    assert all(is_free_model(m) for m in settings.chain("light") + settings.chain("heavy"))
    assert set(settings.chain("light")).isdisjoint(settings.chain("heavy"))


def test_each_free_chain_spans_two_upstream_vendors() -> None:
    settings = Settings(_env_file=None)
    for tier in ("light", "heavy"):
        vendors = {model.split("/")[0] for model in settings.chain(tier)}  # type: ignore[arg-type]
        assert len(vendors) == 2, tier


def test_chain_parsing_trims_and_deduplicates() -> None:
    assert parse_model_chain(" a:free, b:free ,a:free,, ") == ["a:free", "b:free"]


def test_anthropic_defaults_are_the_current_models() -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="sk-ant-test-only")
    assert settings.chain("light") == [DEFAULT_ANTHROPIC_LIGHT] == ["claude-haiku-5-5"]
    assert settings.chain("heavy") == [DEFAULT_ANTHROPIC_HEAVY] == ["claude-opus-5-5"]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_openai_and_gemini_need_named_models(provider: str) -> None:
    with pytest.raises(ValidationError, match="LIGHT_MODELS and HEAVY_MODELS"):
        Settings(_env_file=None, llm_provider=provider)
    ok = Settings(_env_file=None, llm_provider=provider, light_models="small-model", heavy_models="big-model")
    assert ok.chain("light") == ["small-model"]


def test_free_only_refuses_a_paid_openrouter_id() -> None:
    with pytest.raises(ValidationError, match="do not end in :free"):
        make_settings(light_models="anthropic/claude-opus-5-5")
    allowed = make_settings(light_models="anthropic/claude-opus-5-5", openrouter_free_only=False)
    assert allowed.chain("light") == ["anthropic/claude-opus-5-5"]


def test_a_connection_string_in_a_key_box_is_refused_without_echoing_it() -> None:
    secret = "postgresql://user:hunter2@db.example/prod"
    with pytest.raises(ValidationError) as caught:
        make_settings(openrouter_api_key=secret)
    assert "does not look like a key" in str(caught.value)
    assert "hunter2" not in str(caught.value)


def test_placeholder_keys_count_as_missing() -> None:
    settings = make_settings(openrouter_api_key="sk-or-v1-...")
    assert settings.openrouter_api_key is None
    assert settings.provider_configured() is False
    assert make_settings(llm_provider="offline", openrouter_api_key=None).provider_configured() is True


def test_langfuse_needs_both_keys() -> None:
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"):
        make_settings(langfuse_public_key="pk-lf-test")
    assert make_settings(langfuse_public_key="pk-lf-a", langfuse_secret_key="sk-lf-b").langfuse_enabled()


def test_redis_url_scheme_and_backend_resolution() -> None:
    with pytest.raises(ValidationError, match="redis://"):
        make_settings(redis_url="https://cache.example")
    assert make_settings(cache_backend="auto").resolved_cache_backend() == "memory"
    assert (
        make_settings(
            cache_backend="auto", redis_url="rediss://u:p@cache.example:6380"
        ).resolved_cache_backend()
        == "redis"
    )


def test_ceilings_are_ordered() -> None:
    with pytest.raises(ValidationError, match="MAX_OUTPUT_TOKENS_LIGHT"):
        make_settings(max_output_tokens_light=1000, max_output_tokens_heavy=500)


def test_model_prices_must_be_a_table_of_numbers() -> None:
    with pytest.raises(ValidationError):
        make_settings(model_prices="not json")
    with pytest.raises(ValidationError):
        make_settings(model_prices='{"m": [1]}')
    with pytest.raises(ValidationError):
        make_settings(model_prices='{"m": ["a", "b"]}')
    ok = make_settings(model_prices='{"gpt-x": [0.15, 0.6]}')
    assert ok.extra_prices() == {"gpt-x": ["0.15", "0.6"]}


def test_gateway_keys_and_origins_are_lists() -> None:
    settings = make_settings(
        gateway_api_keys=" k1 , k2,", cors_allow_origins="https://a.example, https://b.example"
    )
    assert settings.gateway_keys() == ["k1", "k2"]
    assert settings.cors_origins() == ["https://a.example", "https://b.example"]
