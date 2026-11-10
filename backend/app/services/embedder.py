from __future__ import annotations

import logging

from pinecone import Pinecone

from app.config import Settings

logger = logging.getLogger(__name__)

# Pinecone Inference model. 1024 dims, MTEB ~61, multilingual.
EMBEDDING_MODEL = "multilingual-e5-large"
EMBEDDING_DIM = 1024
MAX_BATCH = 96  # Pinecone Inference limit: 96 inputs per request


class Embedder:
    """Embeddings via Pinecone Inference — no local model, zero server RAM cost.

    Two entry points with intentionally distinct names because E5 models
    require different ``input_type`` values for documents vs queries:
    - embed_documents(texts) -> input_type="passage" (used at index time)
    - embed_query(text)      -> input_type="query"   (used at retrieval time)
    Mixing them up returns garbage retrieval results.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is required for Pinecone Inference")
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            batch = texts[i : i + MAX_BATCH]
            result = self._pc.inference.embed(
                model=EMBEDDING_MODEL,
                inputs=batch,
                parameters={"input_type": input_type},
            )
            for e in result:
                vectors.append(list(e.values))
            logger.info(
                "embedded_batch model=%s input_type=%s count=%d",
                EMBEDDING_MODEL, input_type, len(batch),
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "passage")

    def embed_query(self, text: str) -> list[float]:
        vecs = self._embed([text], "query")
        return vecs[0]

    @staticmethod
    def dimension() -> int:
        return EMBEDDING_DIM