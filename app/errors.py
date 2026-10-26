"""Errors the gateway raises, and the one shape a client ever sees.

A client gets ``{"error": {"type", "message", "incident_id"}}`` and an HTTP status. The message
is written for the client: it never carries a provider's own error text (which can hold a key
fragment, a quota or an account name), a stack trace or the rejected input. The full story goes
to the log under the same incident id.
"""

from __future__ import annotations

import secrets
from typing import Literal

ErrorType = Literal[
    "invalid_request",
    "prompt_too_long",
    "prompt_injection",
    "unauthorized",
    "rate_limited",
    "budget_exhausted",
    "busy",
    "upstream_unavailable",
    "upstream_failed",
    "not_configured",
    "not_found",
    "internal_error",
]


def new_incident_id() -> str:
    """Short, unguessable, safe to show: ``inc_`` and 10 hex characters."""
    return "inc_" + secrets.token_hex(5)


class GatewayError(Exception):
    """An error with a client-facing type, status and message."""

    status_code: int = 500
    error_type: ErrorType = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: ErrorType | None = None,
        retry_after_s: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type
        self.retry_after_s = retry_after_s
        self.incident_id = new_incident_id()

    def body(self) -> dict[str, dict[str, str]]:
        return {
            "error": {
                "type": self.error_type,
                "message": self.message,
                "incident_id": self.incident_id,
            }
        }

    def headers(self) -> dict[str, str]:
        return {"Retry-After": str(self.retry_after_s)} if self.retry_after_s else {}


class InvalidRequest(GatewayError):
    status_code = 400
    error_type = "invalid_request"


class PromptTooLong(GatewayError):
    status_code = 400
    error_type = "prompt_too_long"


class PromptInjectionRefused(GatewayError):
    status_code = 400
    error_type = "prompt_injection"


class Unauthorized(GatewayError):
    status_code = 401
    error_type = "unauthorized"


class RateLimited(GatewayError):
    status_code = 429
    error_type = "rate_limited"


class BudgetExhausted(GatewayError):
    status_code = 429
    error_type = "budget_exhausted"


class Busy(GatewayError):
    status_code = 503
    error_type = "busy"


class NotConfigured(GatewayError):
    status_code = 503
    error_type = "not_configured"


class UpstreamUnavailable(GatewayError):
    """Every model of the cascade failed for a reason that may pass (429, timeout, 5xx)."""

    status_code = 503
    error_type = "upstream_unavailable"


class UpstreamFailed(GatewayError):
    """Every model of the cascade failed and retrying will not help (bad key, every id gone)."""

    status_code = 502
    error_type = "upstream_failed"


class NotFound(GatewayError):
    status_code = 404
    error_type = "not_found"


class ConfigurationError(RuntimeError):
    """A setting makes the service unable to start. The message names the setting, never a value."""
