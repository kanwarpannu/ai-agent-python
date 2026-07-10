from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..models import RagAnswer, TrajectoryStep


class EvalCase(BaseModel):
    """A single labelled question from the evaluation dataset."""

    id: str
    question: str
    reference_answer: str | None = None
    relevant_pages: list[int] = Field(default_factory=list)


class JudgeScore(BaseModel):
    """Result of one LLM-as-judge call."""

    score: float | None = None  # 0.0 - 1.0, or None if parsing failed
    reason: str = ""


class CaseResult(BaseModel):
    """All scores produced for a single evaluation case."""

    case: EvalCase
    answer: str
    model: str | None = None

    # black-box (treat the system as opaque: question -> answer)
    correctness: JudgeScore | None = None
    answer_relevancy: JudgeScore | None = None

    # single-step: retrieval
    context_recall: float | None = None
    context_precision: float | None = None
    mrr: float | None = None
    avg_distance: float | None = None
    retrieved_pages: list[int] = Field(default_factory=list)

    # single-step: generation
    faithfulness: JudgeScore | None = None

    # trajectory
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    trajectory_score: float | None = None
    trajectory_checks: dict[str, bool] = Field(default_factory=dict)

    @classmethod
    def from_answer(cls, case: EvalCase, answer: RagAnswer) -> "CaseResult":
        return cls(case=case, answer=answer.answer, model=answer.model)


class EvalReport(BaseModel):
    """Aggregate report across the whole dataset."""

    dataset: str
    num_cases: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    results: list[CaseResult] = Field(default_factory=list)
    aggregates: dict[str, float | None] = Field(default_factory=dict)
