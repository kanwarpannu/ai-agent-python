from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AppState(str, Enum):
    IDLE = "IDLE"
    INGESTING = "INGESTING"
    READY = "READY"
    RETRIEVING = "RETRIEVING"
    ANSWERING = "ANSWERING"
    ERROR = "ERROR"


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    id: str | None = None
    document: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float | None = None


class RagAnswer(BaseModel):
    question: str
    answer: str
    context_hits: list[RetrievalHit] = Field(default_factory=list)
    model: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RAGContext(BaseModel):
    state: AppState = AppState.IDLE
    pdf_path: str
    collection_name: str
    last_error: str | None = None
    last_query: str | None = None
    retrieved_hits: list[RetrievalHit] = Field(default_factory=list)
