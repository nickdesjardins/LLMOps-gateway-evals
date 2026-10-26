"""Request and response models (Pydantic v2). Every JSON body in and out is validated here.

The response models are the contract the playground, the project page, the eval runner and
the tests read. Money is a decimal string (``"0.0000412000"``): a float would round it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskHint = Literal["auto", "classify", "extract", "answer", "synthesize"]
CacheMode = Literal["use", "bypass", "refresh"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- ask


class AskRequest(_Strict):
    question: str = Field(min_length=1, max_length=8_000, description="Checked against QUESTION_MAX_CHARS.")
    task: TaskHint = Field(default="auto", description="classify and extract force the light tier.")
    cache: CacheMode = Field(
        default="use", description="bypass skips the cache; refresh re-answers and stores."
    )
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class BlockedNeighbor(BaseModel):
    question: str
    similarity: float
    reason: str


class CacheInfo(BaseModel):
    outcome: Literal["hit", "miss", "bypass", "off"]
    match: Literal["exact", "semantic"] | None = None
    similarity: float | None = None
    threshold: float | None = None
    nearest_question: str | None = None
    embed_ms: float | None = None
    search_ms: float | None = None
    total_ms: float | None = None
    backend: str | None = None
    cached_at: float | None = None
    origin_trace_id: str | None = None
    blocked: list[BlockedNeighbor] = Field(default_factory=list)


class RouteSignal(BaseModel):
    name: str
    weight: float


class RouteInfo(BaseModel):
    tier: Literal["light", "heavy"]
    score: float
    #: The score every request starts from; the signals' weights are added to it.
    base: float
    threshold: float
    decided_by: str
    signals: list[RouteSignal]


class AttemptInfo(BaseModel):
    model: str
    tier: Literal["light", "heavy"]
    outcome: str
    detail: str
    latency_ms: float


class Citation(BaseModel):
    label: str
    chunk_id: str
    title: str


class SourceInfo(BaseModel):
    label: str
    chunk_id: str
    doc_id: str
    title: str
    rank: int
    rrf: float
    dense: float
    bm25: float


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    estimated: bool = False


class CostInfo(BaseModel):
    actual_usd: str | None = Field(description="What the calls cost; null when a model has no known price.")
    reference_usd: str | None = Field(description="The same tokens at the tier's reference list price.")
    avoided_usd: str | None = Field(description="On a cache hit: the reference cost the hit avoided.")


class InjectionInfo(BaseModel):
    detected: bool
    signals: list[str]
    neutralized: int
    removed_characters: int


class AskResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[SourceInfo]
    cache: CacheInfo
    route: RouteInfo | None
    model: str
    tier: Literal["light", "heavy"]
    attempts: list[AttemptInfo]
    fallback: bool
    usage: UsageInfo
    cost: CostInfo
    validation: str | None = None
    injection: InjectionInfo
    prompt_version: str
    corpus_version: str
    latency_ms: float
    ttft_ms: float | None = None


# ---------------------------------------------------------------------------- chat


class ChatMessage(_Strict):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32_000)


class ChatCompletionRequest(BaseModel):
    """The Chat Completions request. Unknown OpenAI parameters are accepted and ignored."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="auto", max_length=200, description="auto, light, heavy, or a chain model id.")
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    max_tokens: int | None = Field(default=None, ge=1, le=100_000)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=100_000)
    stream: bool = False
    task: TaskHint = "auto"
    cache: CacheMode = "use"

    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class ChatChoiceMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatChoiceMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    x_gateway: dict[str, Any]


# ------------------------------------------------------------------- tools and ops


class RouteRequest(_Strict):
    text: str = Field(min_length=1, max_length=8_000)
    task: TaskHint = "auto"


class SimilarityRequest(_Strict):
    a: str = Field(min_length=1, max_length=500)
    b: str = Field(min_length=1, max_length=500)


class SimilarityResponse(BaseModel):
    similarity: float
    threshold: float
    #: Whether the contrast guard would refuse this pair as a hit, and why.
    guard: str | None
    would_hit: bool
    embedder: str
    normalized_a: str
    normalized_b: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    environment: str
    demo_mode: bool
    provider: dict[str, Any]
    cache: dict[str, Any]
    embedder: dict[str, Any]
    telemetry: dict[str, Any]
    corpus: dict[str, Any]
    budget: dict[str, Any]
