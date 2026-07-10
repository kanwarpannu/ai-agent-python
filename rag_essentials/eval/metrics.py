from __future__ import annotations

from ..models import AppState, RetrievalHit, TrajectoryStep
from .models import EvalCase

# Canonical transition-state paths through ``RAGService.ask``. The first ask on a
# fresh (IDLE) service includes a leading READY; subsequent asks start at RETRIEVING.
_CANONICAL_PATHS = (
    [AppState.READY, AppState.RETRIEVING, AppState.ANSWERING, AppState.READY],
    [AppState.RETRIEVING, AppState.ANSWERING, AppState.READY],
)


# --- single-step: retrieval ------------------------------------------------


def _hit_page(hit: RetrievalHit) -> int | None:
    page = hit.metadata.get("page")
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def retrieved_pages(hits: list[RetrievalHit]) -> list[int]:
    pages: list[int] = []
    for hit in hits:
        page = _hit_page(hit)
        if page is not None:
            pages.append(page)
    return pages


def context_recall(hits: list[RetrievalHit], relevant_pages: list[int]) -> float | None:
    """Fraction of ground-truth relevant pages that appear in the retrieved hits."""
    if not relevant_pages:
        return None
    found = set(retrieved_pages(hits)) & set(relevant_pages)
    return len(found) / len(set(relevant_pages))


def context_precision(hits: list[RetrievalHit], relevant_pages: list[int]) -> float | None:
    """Fraction of retrieved hits whose page is relevant."""
    if not relevant_pages or not hits:
        return None
    relevant = set(relevant_pages)
    matched = sum(1 for p in retrieved_pages(hits) if p in relevant)
    return matched / len(hits)


def mrr(hits: list[RetrievalHit], relevant_pages: list[int]) -> float | None:
    """Reciprocal rank of the first relevant hit (0.0 if none)."""
    if not relevant_pages:
        return None
    relevant = set(relevant_pages)
    for rank, hit in enumerate(hits, start=1):
        if _hit_page(hit) in relevant:
            return 1.0 / rank
    return 0.0


def avg_distance(hits: list[RetrievalHit]) -> float | None:
    distances = [h.distance for h in hits if h.distance is not None]
    if not distances:
        return None
    return sum(distances) / len(distances)


# --- trajectory ------------------------------------------------------------


def _transition_states(trajectory: list[TrajectoryStep]) -> list[AppState]:
    return [s.state for s in trajectory if s.step == "transition" and s.state is not None]


def trajectory_checks(trajectory: list[TrajectoryStep]) -> dict[str, bool]:
    states = _transition_states(trajectory)
    steps = [s.step for s in trajectory]

    # ordering: a RETRIEVING transition must precede ANSWERING, which precedes a final READY
    correct_order = False
    if AppState.RETRIEVING in states and AppState.ANSWERING in states:
        correct_order = (
            states.index(AppState.RETRIEVING) < states.index(AppState.ANSWERING)
            and states[-1] == AppState.READY
        )

    no_error_state = AppState.ERROR not in states

    # the retrieve step must have kept at least one chunk and occur before answering
    context_nonempty_before_answer = False
    if "retrieve" in steps and "answer" in steps and steps.index("retrieve") < steps.index("answer"):
        retrieve_step = trajectory[steps.index("retrieve")]
        context_nonempty_before_answer = retrieve_step.detail.get("kept", 0) > 0

    expected_path_match = states in _CANONICAL_PATHS

    return {
        "correct_order": correct_order,
        "no_error_state": no_error_state,
        "context_nonempty_before_answer": context_nonempty_before_answer,
        "expected_path_match": expected_path_match,
    }


def trajectory_score(checks: dict[str, bool]) -> float | None:
    if not checks:
        return None
    return sum(1 for v in checks.values() if v) / len(checks)


# --- aggregation -----------------------------------------------------------

# (metric name, attribute path). Judge metrics live under ``.score``.
_NUMERIC_FIELDS = ("context_recall", "context_precision", "mrr", "avg_distance", "trajectory_score")
_JUDGE_FIELDS = ("correctness", "answer_relevancy", "faithfulness")


def aggregate(results: list) -> dict[str, float | None]:
    """Mean of each metric across cases, ignoring cases where the metric is None."""
    summary: dict[str, float | None] = {}

    for field in _NUMERIC_FIELDS:
        values = [getattr(r, field) for r in results if getattr(r, field) is not None]
        summary[field] = sum(values) / len(values) if values else None

    for field in _JUDGE_FIELDS:
        values = [
            getattr(r, field).score
            for r in results
            if getattr(r, field) is not None and getattr(r, field).score is not None
        ]
        summary[field] = sum(values) / len(values) if values else None

    return summary


__all__ = [
    "retrieved_pages",
    "context_recall",
    "context_precision",
    "mrr",
    "avg_distance",
    "trajectory_checks",
    "trajectory_score",
    "aggregate",
]
