"""Shared fixtures. The whole suite is hermetic: no network, no Redis, no model, no key.

* Outbound connections are refused at the socket level, except to the loopback address
  (asyncio on Windows talks to itself through a loopback socket pair). A test that reached the
  internet would fail, not pass slowly.
* Every provider key is removed from the environment and ``Settings`` is built with
  ``_env_file=None``, so a developer's ``.env.local`` never leaks into a test.
* Models are :class:`ScriptedProvider`; embeddings are the deterministic hash embedder or a
  :class:`TableEmbedder` that returns chosen vectors; Redis is :class:`FakeRedis`.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.embeddings import DIMENSIONS, HashEmbedder, normalize_rows
from app.pricing import Usage
from app.providers.base import ChatRequest, Completion, ProviderError, StreamEvent

_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "REDIS_URL",
    "GATEWAY_API_KEYS",
    "LLM_PROVIDER",
    "DEMO_MODE",
)


# ----------------------------------------------------------------- no network


_real_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    host = address[0] if isinstance(address, tuple) else address
    try:
        if ipaddress.ip_address(host).is_loopback:
            return _real_connect(self, address)
    except ValueError:
        pass
    raise RuntimeError(f"the hermetic suite refused a network connection to {host!r}")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_provider": "openrouter",
        "openrouter_api_key": "sk-or-v1-test-only",
        "embedding_provider": "hash",
        "cache_backend": "memory",
        "light_models": "light/a:free,light/b:free",
        "heavy_models": "heavy/a:free,heavy/b:free",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


# ----------------------------------------------------------------- embeddings


class TableEmbedder(HashEmbedder):
    """Returns the vector a test assigned to a text; hashes anything else."""

    id = "table-test"

    def __init__(self, table: dict[str, np.ndarray] | None = None) -> None:
        super().__init__(memo_size=0)
        self.table = {k: v for k, v in (table or {}).items()}
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        rows = []
        for text in texts:
            if text in self.table:
                rows.append(normalize_rows(self.table[text].reshape(1, -1))[0])
            else:
                rows.append(super()._compute([text])[0])
        return np.vstack(rows).astype(np.float32)


def vector_at(
    similarity: float, base: np.ndarray | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors whose cosine similarity is exactly ``similarity``."""
    rng = np.random.default_rng(seed)
    a = base if base is not None else rng.normal(size=DIMENSIONS)
    a = a / np.linalg.norm(a)
    noise = rng.normal(size=DIMENSIONS)
    orthogonal = noise - (noise @ a) * a
    orthogonal /= np.linalg.norm(orthogonal)
    b = similarity * a + np.sqrt(max(0.0, 1 - similarity**2)) * orthogonal
    return a.astype(np.float32), b.astype(np.float32)


# ----------------------------------------------------------------- providers


Script = Callable[[str, ChatRequest], Any]


class ScriptedProvider:
    """A provider whose answers are written by the test, per model.

    ``responses[model]`` is a list consumed in order; each item is a string (the answer), a
    ProviderError to raise, or a callable taking the request. A model without a script answers
    ``default`` (a cited sentence by default).
    """

    name = "openrouter"

    def __init__(
        self, responses: dict[str, list[Any]] | None = None, default: str = "Twenty days a year [S1]."
    ) -> None:
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.default = default
        self.calls: list[tuple[str, ChatRequest]] = []
        self.cost: Decimal | None = None
        self.usage = Usage(input_tokens=900, output_tokens=40)

    def _next(self, model: str, request: ChatRequest) -> Any:
        self.calls.append((model, request))
        queue = self.responses.get(model)
        item = queue.pop(0) if queue else self.default
        if callable(item) and not isinstance(item, ProviderError):
            item = item(model, request)
        if isinstance(item, BaseException):
            raise item
        return item

    async def complete(self, model: str, request: ChatRequest) -> Completion:
        item = self._next(model, request)
        if isinstance(item, Completion):
            return item
        return Completion(
            model=model, text=str(item), finish_reason="stop", usage=self.usage, provider_cost_usd=self.cost
        )

    async def stream(self, model: str, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        item = self._next(model, request)
        if isinstance(item, list):  # a list of pieces, possibly ending with an exception
            text = ""
            for piece in item:
                if isinstance(piece, BaseException):
                    raise piece
                text += piece
                yield StreamEvent(kind="delta", text=piece)
        else:
            text = str(item)
            for piece in text.split(" "):
                yield StreamEvent(kind="delta", text=piece + " ")
        yield StreamEvent(
            kind="done",
            completion=Completion(
                model=model,
                text=text.strip(),
                finish_reason="stop",
                usage=self.usage,
                provider_cost_usd=self.cost,
                ttft_ms=5.0,
            ),
        )

    async def aclose(self) -> None:
        return None


def rate_limited(model: str = "m") -> ProviderError:
    return ProviderError(model, "rate_limited", "rate limited (429)", status=429)


# ----------------------------------------------------------------- Redis


class FakeRedis:
    """The handful of Redis commands RedisVectorStore uses, with exact KNN in numpy."""

    def __init__(self, *, search: bool = True) -> None:
        self.search = search
        self.strings: dict[str, bytes] = {}
        self.hashes: dict[str, dict[str, Any]] = {}
        self.ttl: dict[str, int] = {}
        self.indexes: dict[str, list[Any]] = {}
        self.commands: list[tuple[Any, ...]] = []
        self.fail_next = 0

    def _maybe_fail(self) -> None:
        if self.fail_next:
            self.fail_next -= 1
            raise ConnectionError("Connection reset by peer")

    async def execute_command(self, *args: Any) -> Any:
        self._maybe_fail()
        self.commands.append(args)
        name = str(args[0]).upper()
        if not self.search and name.startswith("FT."):
            raise Exception(f"ERR unknown command '{args[0]}'")
        if name == "FT.INFO":
            if args[1] not in self.indexes:
                raise Exception("Unknown index name")
            docs = sum(1 for k in self.hashes if k.startswith(self._prefix(args[1])))
            return [b"index_name", args[1].encode(), b"num_docs", str(docs).encode()]
        if name == "FT.CREATE":
            self.indexes[args[1]] = list(args)
            return b"OK"
        if name == "FT.SEARCH":
            return self._search(args)
        raise NotImplementedError(name)

    def _prefix(self, index: str) -> str:
        spec = self.indexes[index]
        return str(spec[spec.index("PREFIX") + 2])

    def _search(self, args: tuple[Any, ...]) -> list[Any]:
        index, query = args[1], args[2]
        import re

        namespace = re.search(r"@ns:\{([^}]*)\}", query).group(1)  # type: ignore[union-attr]
        k = int(re.search(r"KNN (\d+)", query).group(1))  # type: ignore[union-attr]
        vector = np.frombuffer(args[args.index("vec") + 1], dtype=np.float32)
        prefix = self._prefix(index)
        scored = []
        for key, fields in self.hashes.items():
            if not key.startswith(prefix) or fields["ns"] != namespace:
                continue
            stored = np.frombuffer(fields["embedding"], dtype=np.float32)
            scored.append((1.0 - float(stored @ vector), key, fields))
        scored.sort(key=lambda item: item[0])
        reply: list[Any] = [len(scored)]
        for dist, key, fields in scored[:k]:
            flat: list[bytes] = []
            for name, value in fields.items():
                if name == "embedding":
                    continue
                flat += [name.encode(), value.encode() if isinstance(value, str) else value]
            flat += [b"dist", repr(dist).encode()]
            reply += [key.encode(), flat]
        return reply

    async def get(self, key: str) -> bytes | None:
        self._maybe_fail()
        return self.strings.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._maybe_fail()
        self.strings[key] = value.encode()
        if ex:
            self.ttl[key] = ex
        return True

    async def hset(self, key: str, mapping: dict[str, Any]) -> int:
        self._maybe_fail()
        self.hashes[key] = dict(mapping)
        return len(mapping)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        self._maybe_fail()
        fields = self.hashes.get(key, {})
        return {k.encode(): (v.encode() if isinstance(v, str) else v) for k, v in fields.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        self.ttl[key] = seconds
        return True

    async def delete(self, key: str) -> int:
        removed = int(key in self.hashes or key in self.strings)
        self.hashes.pop(key, None)
        self.strings.pop(key, None)
        return removed

    async def scan_iter(self, match: str, count: int = 100) -> AsyncIterator[str]:
        prefix = match.rstrip("*")
        for key in list(self.hashes) + list(self.strings):
            if key.startswith(prefix):
                yield key

    async def ping(self) -> bool:
        self._maybe_fail()
        return True

    async def aclose(self) -> None:
        return None


# ----------------------------------------------------------------- the app


async def build_test_app(
    settings: Settings | None = None,
    *,
    provider: Any | None = None,
    embedder: Any | None = None,
    store: Any | None = None,
    sinks: list[Any] | None = None,
) -> tuple[Any, Any]:
    from app.components import build_components
    from app.main import create_app

    settings = settings or make_settings()
    components = await build_components(
        settings,
        embedder=embedder or HashEmbedder(),
        provider=provider if provider is not None else ScriptedProvider(),
        store=store,
        sinks=sinks if sinks is not None else [],
    )
    return create_app(settings, components), components


@pytest.fixture
async def client_and_components() -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    app, components = await build_test_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    ) as client:
        yield client, components
