from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from pinecone import Pinecone

from app.config import Settings
from app.services.embedder import Embedder
from app.services.types import RetrievedChunk
from app.services.web_search import WebSearchService

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
MIN_PDF_HITS = 2

__all__ = [
    "RetrievedChunk",
    "Retriever",
    "PineconeRetriever",
    "RAGService",
    "DEFAULT_TOP_K",
    "MIN_PDF_HITS",
]


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
    """Combined retrieval: PDF first, web fallback when PDF is weak."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pdf: PineconeRetriever | None = (
            PineconeRetriever(settings) if settings.PINECONE_API_KEY else None
        )
        self._web: WebSearchService | None = (
            WebSearchService(settings)
            if settings.GOOGLE_CSE_API_KEY and settings.GOOGLE_CSE_ID
            else None
        )

    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        pdf_chunks: list[RetrievedChunk] = []
        if self._pdf is not None:
            try:
                pdf_chunks = await asyncio.to_thread(
                    self._pdf.retrieve, query, top_k
                )
            except Exception as exc:
                logger.warning(
                    "rag_pdf_retrieve_failed error_type=%s", type(exc).__name__
                )

        top_score = pdf_chunks[0].score if pdf_chunks else None
        pdf_is_strong = (
            len(pdf_chunks) >= MIN_PDF_HITS
            and top_score is not None
            and top_score >= self._settings.RAG_FALLBACK_SCORE_THRESHOLD
        )
        if pdf_is_strong:
            logger.info(
                "rag_pdf_strong hits=%d top_score=%.3f (no web fallback)",
                len(pdf_chunks),
                top_score,
            )
            return pdf_chunks

        web_chunks: list[RetrievedChunk] = []
        if self._web is not None:
            try:
                web_chunks = await self._web.search(
                    query, num_results=self._settings.RAG_WEB_RESULTS
                )
            except Exception as exc:
                logger.warning(
                    "rag_web_search_failed error_type=%s", type(exc).__name__
                )

        logger.info(
            "rag_combined pdf=%d web=%d (fallback fired)",
            len(pdf_chunks),
            len(web_chunks),
        )
        return pdf_chunks + web_chunks

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