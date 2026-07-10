from __future__ import annotations

import json
import re

from lib.llm import LLM
from lib.messages import SystemMessage, UserMessage

from .models import JudgeScore

_JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of RAG question-answering systems. "
    "You always respond with a single JSON object and nothing else, of the form "
    '{"score": <float between 0 and 1>, "reason": "<short justification>"}. '
    "Do not wrap the JSON in markdown fences."
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class Judge:
    """LLM-as-judge built on the shared ``lib.llm.LLM`` client.

    ``lib.llm`` has no structured-output support, so we instruct the model to
    emit a JSON object and parse it defensively.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.llm = LLM(model=model, temperature=temperature)

    def _ask(self, instruction: str) -> JudgeScore:
        messages = [
            SystemMessage(content=_JUDGE_SYSTEM),
            UserMessage(content=instruction),
        ]
        response = self.llm.invoke(messages)
        raw = (response.content or "").strip()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> JudgeScore:
        candidate = raw
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            match = _JSON_BLOCK.search(raw)
            if not match:
                return JudgeScore(score=None, reason=f"unparseable judge output: {raw[:200]}")
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return JudgeScore(score=None, reason=f"unparseable judge output: {raw[:200]}")

        score = data.get("score")
        try:
            score = None if score is None else max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            score = None
        return JudgeScore(score=score, reason=str(data.get("reason", "")))

    # --- black-box metrics -------------------------------------------------

    def correctness(self, question: str, answer: str, reference: str) -> JudgeScore:
        return self._ask(
            "Compare the SYSTEM ANSWER to the REFERENCE ANSWER for the QUESTION. "
            "Score 1.0 if the system answer is factually consistent with and as complete "
            "as the reference, 0.0 if it is wrong or contradictory, partial credit otherwise.\n\n"
            f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nSYSTEM ANSWER:\n{answer}"
        )

    def answer_relevancy(self, question: str, answer: str) -> JudgeScore:
        return self._ask(
            "Judge how directly the ANSWER addresses the QUESTION, ignoring factual "
            "correctness. Score 1.0 if it fully and directly answers what was asked, "
            "0.0 if it is off-topic or evasive.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        )

    # --- single-step (generation) -----------------------------------------

    def faithfulness(self, answer: str, context: str) -> JudgeScore:
        return self._ask(
            "Judge whether every factual claim in the ANSWER is supported by the CONTEXT. "
            "Score 1.0 if all claims are grounded in the context, 0.0 if the answer "
            "introduces unsupported facts (hallucination). An answer that correctly states "
            "the information is not in the context should score 1.0.\n\n"
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
        )
