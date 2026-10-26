"""A provider that calls no model (``LLM_PROVIDER=offline``).

For a grounded question it answers with the first two sentences of the retrieved passage that
shares the most words with the question, cited, or with the refusal token when no passage
shares enough. For any other request it says that no model was called. Deterministic, instant
and free: it lets the whole gateway run on a laptop without a key, and the cache replay
(``evals/replay.py``) measure the cache without spending a request.

It is not a language model and is never presented as one: its model id is
``offline/extractive``, and the playground says so.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from app.config import OFFLINE_MODEL
from app.guard import estimate_messages_tokens, estimate_tokens
from app.pricing import Usage
from app.providers.base import ChatRequest, Completion, StreamEvent
from app.rag.prompts import REFUSAL_TOKEN, parse_question, parse_sources

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or the to what when "
    "where which who why with you your our we my".split()
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def extractive_answer(user_message: str) -> str:
    sources = parse_sources(user_message)
    question = _words(parse_question(user_message) or user_message)
    if not sources or not question:
        return REFUSAL_TOKEN
    scored = sorted(
        ((len(question & _words(text)), number, text) for number, text in sources),
        key=lambda item: (-item[0], item[1]),
    )
    overlap, number, text = scored[0]
    if overlap < max(2, len(question) // 3):
        return REFUSAL_TOKEN
    sentences = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    return " ".join(sentences[:2]) + f" [S{number}]"


class OfflineProvider:
    name = "offline"

    def _answer(self, request: ChatRequest) -> str:
        user = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
        if parse_sources(user):
            return extractive_answer(user)
        words = len(user.split())
        return f"Offline mode: no model was called. Your message had {words} words."

    async def complete(self, model: str, request: ChatRequest) -> Completion:
        text = self._answer(request)
        usage = Usage(
            input_tokens=estimate_messages_tokens(request.messages),
            output_tokens=estimate_tokens(text),
            estimated=True,
        )
        return Completion(model=OFFLINE_MODEL, text=text, finish_reason="stop", usage=usage)

    async def stream(self, model: str, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        completion = await self.complete(model, request)
        for piece in re.findall(r"\S+\s*", completion.text):
            yield StreamEvent(kind="delta", text=piece)
        yield StreamEvent(kind="done", completion=completion)

    async def aclose(self) -> None:
        return None
