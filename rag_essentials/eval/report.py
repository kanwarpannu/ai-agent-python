from __future__ import annotations

from .models import CaseResult, EvalReport

# Display label -> CaseResult attribute. Judge metrics expose their value via `.score`.
_PER_CASE_COLUMNS = [
    ("correct", "correctness", True),
    ("relevancy", "answer_relevancy", True),
    ("recall", "context_recall", False),
    ("precision", "context_precision", False),
    ("mrr", "mrr", False),
    ("faithful", "faithfulness", True),
    ("traj", "trajectory_score", False),
]


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def _case_value(result: CaseResult, attr: str, is_judge: bool) -> float | None:
    raw = getattr(result, attr)
    if raw is None:
        return None
    return raw.score if is_judge else raw


def render_table(report: EvalReport) -> str:
    headers = ["case"] + [label for label, _, _ in _PER_CASE_COLUMNS]
    lines: list[str] = []
    lines.append(f"RAG Evaluation — {report.num_cases} cases — dataset: {report.dataset}")
    lines.append("")

    header_row = f"{'case':<12} " + " ".join(f"{h:>9}" for h in headers[1:])
    lines.append(header_row)
    lines.append("-" * len(header_row))

    for result in report.results:
        cells = [
            _fmt(_case_value(result, attr, is_judge))
            for _, attr, is_judge in _PER_CASE_COLUMNS
        ]
        lines.append(f"{result.case.id:<12} " + " ".join(f"{c:>9}" for c in cells))

    lines.append("-" * len(header_row))

    # aggregate row (means across cases)
    agg = report.aggregates
    agg_cells = []
    for _, attr, _ in _PER_CASE_COLUMNS:
        agg_cells.append(_fmt(agg.get(attr)))
    lines.append(f"{'MEAN':<12} " + " ".join(f"{c:>9}" for c in agg_cells))

    lines.append("")
    lines.append("Lenses: correct/relevancy = black-box · recall/precision/mrr = retrieval ·")
    lines.append("        faithful = generation (single-step) · traj = trajectory")
    if agg.get("avg_distance") is not None:
        lines.append(f"Mean retrieval distance: {agg['avg_distance']:.4f}")

    return "\n".join(lines)
