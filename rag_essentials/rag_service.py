from __future__ import annotations

import logging

from .config import Settings
from .llm_adapter import LLMAdapter
from .models import AppState, RAGContext, RagAnswer
from .pdf_loader import PDFLoader
from .state_machine import StateMachine
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """Thin orchestration layer.

    Keeps business logic separate from transport concerns (CLI / future FastAPI / worker).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.context = RAGContext(
            pdf_path=settings.app.pdf_path,
            collection_name=settings.app.collection_name,
        )
        self.state_machine = StateMachine(self.context)
        self.loader = PDFLoader(
            chunk_size=settings.retrieval.chunk_size,
            chunk_overlap=settings.retrieval.chunk_overlap,
        )
        self.vector_store = ChromaVectorStore(
            config=settings.chroma,
            collection_name=settings.app.collection_name,
        )
        self.llm = LLMAdapter(settings.llm)

    def ingest(self) -> int:
        self.state_machine.transition(AppState.INGESTING)

        if self.settings.app.reset_collection_on_ingest:
            self.vector_store.reset_collection()

        chunks = self.loader.load(self.settings.app.pdf_path)
        count = self.vector_store.ingest_chunks(chunks)
        self.state_machine.transition(AppState.READY)
        return count

    def ask(self, question: str) -> RagAnswer:
        if self.context.state == AppState.IDLE:
            logger.info("Service is IDLE, assuming collection already exists and moving to READY")
            self.state_machine.transition(AppState.READY)

        self.context.last_query = question
        self.state_machine.transition(AppState.RETRIEVING)
        hits = self.vector_store.query(question, top_k=self.settings.retrieval.top_k)
        self.context.retrieved_hits = hits[: self.settings.retrieval.max_context_chunks]

        self.state_machine.transition(AppState.ANSWERING)
        answer_text, model_name = self.llm.answer(question, self.context.retrieved_hits)
        self.state_machine.transition(AppState.READY)

        return RagAnswer(
            question=question,
            answer=answer_text,
            context_hits=self.context.retrieved_hits,
            model=model_name,
        )
