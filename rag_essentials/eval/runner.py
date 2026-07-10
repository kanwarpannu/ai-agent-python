from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings, get_settings
from ..exceptions import RAGError
from ..rag_service import RAGService
from . import metrics
from .dataset import load_dataset
from .judge import Judge
from .models import CaseResult, EvalReport

logger = logging.getLogger(__name__)


def _context_text(answer) -> str:
    return "\n\n---\n\n".join(hit.document for hit in answer.context_hits)


def run_eval(
    dataset_path: str | Path,
    settings: Settings | None = None,
    judge_model: str | None = None,
) -> EvalReport:
    """Run every case in the dataset through ``RAGService`` and score it.

    Scores across three lenses: black-box (correctness, answer relevancy),
    single-step (retrieval recall/precision/MRR + generation faithfulness), and
    trajectory (state-transition ordering checks).
    """
    settings = settings or get_settings()
    cases = load_dataset(dataset_path)

    # Constructing RAGService connects to Chroma; fail fast with guidance if it
    # (or the heartbeat) is unreachable, mirroring the `health` command.
    try:
        service = RAGService(settings)
        service.vector_store.heartbeat()
    except RAGError:
        raise
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RAGError(
            "Could not reach ChromaDB. Start it with `docker-compose up -d` and "
            f"ingest with `python -m rag_essentials.main ingest`. Underlying error: {exc}"
        ) from exc

    judge = Judge(model=judge_model or settings.llm.model)

    results: list[CaseResult] = []
    for case in cases:
        logger.info("Evaluating case %s", case.id)
        answer = service.ask(case.question)
        trajectory = list(service.context.trajectory)

        result = CaseResult.from_answer(case, answer)
        result.trajectory = trajectory

        # --- black-box ----------------------------------------------------
        if case.reference_answer:
            result.correctness = judge.correctness(
                case.question, answer.answer, case.reference_answer
            )
        result.answer_relevancy = judge.answer_relevancy(case.question, answer.answer)

        # --- single-step: retrieval --------------------------------------
        hits = answer.context_hits
        result.retrieved_pages = metrics.retrieved_pages(hits)
        result.context_recall = metrics.context_recall(hits, case.relevant_pages)
        result.context_precision = metrics.context_precision(hits, case.relevant_pages)
        result.mrr = metrics.mrr(hits, case.relevant_pages)
        result.avg_distance = metrics.avg_distance(hits)

        # --- single-step: generation -------------------------------------
        result.faithfulness = judge.faithfulness(answer.answer, _context_text(answer))

        # --- trajectory ---------------------------------------------------
        result.trajectory_checks = metrics.trajectory_checks(trajectory)
        result.trajectory_score = metrics.trajectory_score(result.trajectory_checks)

        results.append(result)

    return EvalReport(
        dataset=str(dataset_path),
        num_cases=len(results),
        results=results,
        aggregates=metrics.aggregate(results),
    )
