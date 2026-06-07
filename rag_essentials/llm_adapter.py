from __future__ import annotations

import logging
from textwrap import dedent

from lib.llm import LLM
from lib.messages import AIMessage, SystemMessage, UserMessage

from .config import LLMConfig
from .exceptions import LLMInvocationError
from .models import RetrievalHit

logger = logging.getLogger(__name__)


class LLMAdapter:
    """Adapter around the existing lib.llm.LLM wrapper.

    Notes:
    - The existing wrapper reads OPENAI_API_KEY and OPENAI_BASE_URL from env.
    - The current implementation of lib.llm.LLM.invoke returns `AIMessage,` (with a trailing comma),
      which means the runtime value is a 1-element tuple. We normalize that here instead of
      forcing a change in your existing shared library.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = LLM(model=config.model, temperature=config.temperature)

    def answer(self, question: str, hits: list[RetrievalHit]) -> tuple[str, str | None]:
        prompt = self._build_user_prompt(question, hits)
        messages = [
            SystemMessage(content=self.config.system_prompt),
            UserMessage(content=prompt),
        ]

        try:
            raw_response = self.client.invoke(messages)
        except Exception as exc:  # pragma: no cover
            raise LLMInvocationError(f"LLM invocation failed: {exc}") from exc

        message = self._normalize_ai_message(raw_response)
        return (message.content or "").strip(), self.config.model

    @staticmethod
    def _normalize_ai_message(raw_response) -> AIMessage:
        if isinstance(raw_response, tuple):
            if not raw_response:
                raise LLMInvocationError("LLM returned an empty tuple response")
            raw_response = raw_response[0]

        if not isinstance(raw_response, AIMessage):
            raise LLMInvocationError(f"Unexpected LLM response type: {type(raw_response)}")

        return raw_response

    @staticmethod
    def _build_user_prompt(question: str, hits: list[RetrievalHit]) -> str:
        context_blocks: list[str] = []
        for idx, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "unknown")
            page = hit.metadata.get("page", "?")
            context_blocks.append(
                f"[Chunk {idx} | source={source} | page={page}]\n{hit.document}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        return dedent(
            f"""
            Question:
            {question}

            Context:
            {context}
            """
        ).strip()
