from __future__ import annotations

import logging

import chromadb

from .config import ChromaConfig
from .exceptions import IngestionError, RetrievalError
from .models import DocumentChunk, RetrievalHit

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, config: ChromaConfig, collection_name: str) -> None:
        self.config = config
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(
            host=config.host,
            port=config.port,
            ssl=config.ssl,
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=config.collection_metadata or None,
            configuration={"hnsw": {"space": config.hnsw_space}},
        )
        logger.info(
            "Connected to Chroma at %s:%s using collection=%s",
            config.host,
            config.port,
            collection_name,
        )

    def heartbeat(self) -> int:
        return self.client.heartbeat()

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted existing collection '%s'", self.collection_name)
        except Exception:
            logger.info("Collection '%s' did not exist before reset", self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=self.config.collection_metadata or None,
            configuration={"hnsw": {"space": self.config.hnsw_space}},
        )

    def ingest_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        try:
            self.collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
            )
        except Exception as exc:  # pragma: no cover
            raise IngestionError(f"Failed to add chunks to Chroma: {exc}") from exc

        logger.info("Inserted %s chunks into Chroma", len(chunks))
        return len(chunks)

    def query(self, question: str, top_k: int = 4) -> list[RetrievalHit]:
        try:
            result = self.collection.query(
                query_texts=[question],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover
            raise RetrievalError(f"Vector query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: list[RetrievalHit] = []
        for idx, doc in enumerate(documents):
            hits.append(
                RetrievalHit(
                    id=ids[idx] if idx < len(ids) else None,
                    document=doc,
                    metadata=metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {},
                    distance=distances[idx] if idx < len(distances) else None,
                )
            )

        logger.info("Retrieved %s vector hits", len(hits))
        return hits
