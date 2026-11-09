from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from pinecone import Pinecone

from app.config import Settings
from app.services.embedder import Embedder

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class RetrievedChunk:
    """One piece of context retrieved for a query.

    source is "pdf" or "web". For pdf chunks, title and page are set.
    For web chunks, title is the site name and link is the URL.
    """
    source: str
    title: str
    content: str
    page: int | None = None
    link: str | None = None
    score: float | None = None
    chunk_id: str | None = None


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class PineconeRetriever:
    """Synchronous Pinecone retriever. Wrap in asyncio.to_thread for async."""

    def __init__(self, settings: Settings) -> None:
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set")
        self._settings = settings
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = self._pc.Index(settings.PINECONE_INDEX)
        self._embedder = Embedder()

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        vec = self._embedder.embed([query])[0]
        res = self._index.query(vector=vec, top_k=top_k, include_metadata=True)

        out: list[RetrievedChunk] = []
        for match in res["matches"]:
            md = match["metadata"]
            page_raw = md.get("page")
            # Pinecone returns numeric metadata as float; cast back to int
            page = int(page_raw) if isinstance(page_raw, (int, float)) else None
            out.append(
                RetrievedChunk(
                    source="pdf",
                    title=str(md.get("title", "unknown")),
                    content=str(md.get("text", "")),
                    page=page,
                    score=float(match["score"]),
                    chunk_id=match["id"],
                )
            )

        top_score = out[0].score if out and out[0].score is not None else None
        logger.info(
            "rag_retrieved query_len=%d hits=%d top_score=%s",
            len(query),
            len(out),
            f"{top_score:.3f}" if top_score is not None else "n/a",
        )
        return out


class RAGService:
    """PDF retrieval behind an async interface. Web fallback added in 4.5b."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._retriever: PineconeRetriever | None = (
            PineconeRetriever(settings) if settings.PINECONE_API_KEY else None
        )

    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievedChunk]:
        if self._retriever is None:
            return []
        # Pinecone SDK + torch embed are blocking; run in a thread to keep
        # the event loop free while other requests are in flight.
        return await asyncio.to_thread(self._retriever.retrieve, query, top_k)

    @staticmethod
    def assemble_context(chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for c in chunks:
            if c.source == "pdf":
                parts.append(
                    f"Source: PDF\nTitle: {c.title}\nPage: {c.page}\n{c.content}"
                )
            else:
                parts.append(
                    f"Source: Web Search\nTitle: {c.title}\nLink: {c.link}\n{c.content}"
                )
        return "\n\n".join(parts)