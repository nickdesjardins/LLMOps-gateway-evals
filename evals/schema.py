"""The golden dataset and the records the evaluation stages hand to each other.

The suite is a pipeline of small steps, each writing one JSON file the next one reads, so a
step can be re-run alone and every number in a report can be traced back to its inputs:

=====================  ===================================  ===========================
step                   reads                                writes
=====================  ===================================  ===========================
retrieval_check        the golden set, the corpus           ``retrieval.json``
generate_answers       the golden set, the live pipeline    ``answers.jsonl``
score_deepeval         ``answers.jsonl``                    ``deepeval.json``
score_ragas            ``answers.jsonl``                    ``ragas.json``
gate                   all of the above, thresholds,        exit code and a summary
                       the accepted baseline
=====================  ===================================  ===========================

A golden case says what a correct answer contains (``reference_answer``), where it comes from
(``reference_context``, chunk ids of ``corpus/``), and whether the system should answer at all
(``expected_behavior``). Adversarial cases add ``must_not_contain``: strings whose presence in
an answer means the injection worked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.corpus import Corpus

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "golden" / "golden.jsonl"
OUT_DIR = ROOT / "evals" / "out"

Category = Literal["typical", "edge", "adversarial", "out_of_scope"]


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z]{3}-\d{2,3}$")
    category: Category
    question: str = Field(min_length=3, max_length=500)
    expected_behavior: Literal["answer", "refuse"]
    reference_answer: str | None = None
    reference_context: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> GoldenCase:
        if self.expected_behavior == "answer" and (not self.reference_answer or not self.reference_context):
            raise ValueError(f"{self.id}: an answerable case needs reference_answer and reference_context")
        if self.expected_behavior == "refuse" and self.reference_context:
            raise ValueError(f"{self.id}: a refusal case has no reference context")
        if self.category == "adversarial" and not self.must_not_contain:
            raise ValueError(f"{self.id}: an adversarial case lists what must not appear")
        return self


def load_golden(path: Path = GOLDEN, corpus: Corpus | None = None) -> list[GoldenCase]:
    cases = [
        GoldenCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden case ids")
    if corpus is not None:
        known = {c.chunk_id for c in corpus.chunks}
        for case in cases:
            missing = [cid for cid in case.reference_context if cid not in known]
            if missing:
                raise ValueError(f"{case.id}: reference_context names chunks that do not exist: {missing}")
    return cases


class AnswerRecord(BaseModel):
    """What the pipeline answered for one golden case (one line of ``answers.jsonl``)."""

    id: str
    category: Category
    expected_behavior: Literal["answer", "refuse"]
    question: str
    answer: str
    refused: bool
    retrieved_ids: list[str]
    retrieved_contexts: list[str]
    reference_answer: str | None
    reference_context_ids: list[str]
    reference_contexts: list[str]
    must_not_contain: list[str]
    model: str | None
    tier: str | None
    fallback: bool = False
    latency_ms: float | None = None
    cost_usd: str | None = None
    reference_cost_usd: str | None = None
    error: str | None = None


def read_answers(path: Path) -> list[AnswerRecord]:
    return [
        AnswerRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


# ----------------------------------------------------------- deterministic checks


def refusal_correct(record: AnswerRecord) -> bool:
    """Did the pipeline answer when it should, and decline when it should?"""
    return record.refused == (record.expected_behavior == "refuse")


def leaked(record: AnswerRecord) -> list[str]:
    """The forbidden strings an adversarial answer contains (empty: the injection failed)."""
    lowered = record.answer.lower()
    return [marker for marker in record.must_not_contain if marker.lower() in lowered]
