from __future__ import annotations

import logging
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

from app.config import Settings
from app.services.chunker import Chunk
from app.services.embedder import EMBEDDING_DIM

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 100


@dataclass(frozen=True)
class UpsertResult:
    index_name: str
    dimension: int
    upserted: int
    index_created: bool


class PineconeIndexer:
    """Idempotent builder: create the index if missing, embed chunks, upsert."""

    def __init__(self, settings: Settings) -> None:
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set")
        self._settings = settings
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    def _ensure_index(self) -> bool:
        """Create the index if absent. Return True iff we created it."""
        existing = {idx.name for idx in self._pc.list_indexes()}
        if self._settings.PINECONE_INDEX in existing:
            return False
        logger.info(
            "pinecone_creating_index name=%s dim=%d metric=cosine",
            self._settings.PINECONE_INDEX,
            EMBEDDING_DIM,
        )
        self._pc.create_index(
            name=self._settings.PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        return True

    def build(self, chunks: list[Chunk], vectors: list[list[float]]) -> UpsertResult:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks({len(chunks)}) != vectors({len(vectors)})"
            )
        if not chunks:
            raise ValueError("no chunks to index")

        index_created = self._ensure_index()
        index = self._pc.Index(self._settings.PINECONE_INDEX)

        total = 0
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch_chunks = chunks[i : i + _UPSERT_BATCH]
            batch_vectors = vectors[i : i + _UPSERT_BATCH]
            payload = [
                (
                    c.chunk_id,
                    v,
                    {
                        "source": c.source,
                        "title": c.title,
                        "page": c.page,
                        "text": c.text,
                    },
                )
                for c, v in zip(batch_chunks, batch_vectors, strict=True)
            ]
            index.upsert(vectors=payload)
            total += len(payload)
            logger.info(
                "pinecone_upserted batch=%d cumulative=%d total=%d",
                len(payload),
                total,
                len(chunks),
            )

        stats = index.describe_index_stats()
        logger.info(
            "pinecone_build_done name=%s vectors=%d dim=%d",
            self._settings.PINECONE_INDEX,
            stats.get("total_vector_count", total),
            EMBEDDING_DIM,
        )
        return UpsertResult(
            index_name=self._settings.PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            upserted=total,
            index_created=index_created,
        )