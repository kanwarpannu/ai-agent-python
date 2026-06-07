from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from pypdf import PdfReader

from .exceptions import IngestionError
from .models import DocumentChunk

logger = logging.getLogger(__name__)


class PDFLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, pdf_path: str) -> list[DocumentChunk]:
        path = Path(pdf_path)
        if not path.exists():
            raise IngestionError(f"PDF not found: {pdf_path}")

        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # pragma: no cover - delegated library exceptions
            raise IngestionError(f"Failed to read PDF '{pdf_path}': {exc}") from exc

        logger.info("Loading PDF: %s (%s pages)", path.name, len(reader.pages))
        chunks: list[DocumentChunk] = []

        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            normalized = self._normalize_text(text)
            if not normalized:
                logger.warning("Skipping empty/extractless page %s", page_index)
                continue
            chunks.extend(self._chunk_page(normalized, page_index, path.name))

        if not chunks:
            raise IngestionError(
                "No text extracted from PDF. If the PDF is scanned/image-based, OCR may be required."
            )

        logger.info("Created %s chunks from %s", len(chunks), path.name)
        return chunks

    def _chunk_page(self, page_text: str, page_number: int, source_name: str) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(page_text):
            end = min(start + self.chunk_size, len(page_text))
            chunk_text = page_text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        metadata={
                            "source": source_name,
                            "page": page_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

            if end >= len(page_text):
                break
            start = max(0, end - self.chunk_overlap)

        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
