"""Model providers behind one interface, and the factory that picks one from the settings."""

from __future__ import annotations

from app.config import Settings
from app.errors import ConfigurationError
from app.providers.base import (
    ChatRequest,
    Completion,
    FailureKind,
    Provider,
    ProviderError,
    StreamEvent,
)

__all__ = [
    "ChatRequest",
    "Completion",
    "FailureKind",
    "Provider",
    "ProviderError",
    "StreamEvent",
    "build_provider",
]


def build_provider(settings: Settings) -> Provider:
    """The provider the settings ask for. Raises ConfigurationError when its key is missing."""
    if settings.llm_provider == "offline":
        from app.providers.offline import OfflineProvider

        return OfflineProvider()
    key = settings.provider_key()
    if key is None:
        raise ConfigurationError(
            f"LLM_PROVIDER={settings.llm_provider} needs its API key "
            f"({settings.llm_provider.upper()}_API_KEY)"
        )
    if settings.llm_provider == "anthropic":
        from app.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=key.get_secret_value(),
            timeout_s=settings.model_timeout_s,
            effort_light=settings.anthropic_effort_light,
            effort_heavy=settings.anthropic_effort_heavy,
        )
    from app.providers.openai_compat import OpenAICompatibleProvider

    base_url = {
        "openrouter": settings.openrouter_base_url,
        "openai": settings.openai_base_url,
        "gemini": settings.gemini_base_url,
    }[settings.llm_provider]
    return OpenAICompatibleProvider(
        settings.llm_provider,
        api_key=key.get_secret_value(),
        base_url=base_url,
        timeout_s=settings.model_timeout_s,
        reasoning=settings.openrouter_reasoning,
        app_name=settings.openrouter_app_name,
    )
