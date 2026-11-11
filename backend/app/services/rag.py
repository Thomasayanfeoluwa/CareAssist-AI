from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from pinecone import Pinecone

from app.config import Settings
from app.services.embedder import Embedder
from app.services.types import RetrievedChunk
from app.services.web_search import WebSearchService

from app.services.relevance import (
    JUDGE_ANSWER_THRESHOLD,
    score_chunks_for_question,
)

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
        self._embedder = Embedder(settings)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        vec = self._embedder.embed_query(query)
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
    """Combined retrieval: PDF first, web fallback when PDF is weak.

    All external clients are built lazily on first retrieve() call, so
    importing this module (and therefore importing app.main) never touches
    the network. This matters for tests: conftest imports app.main, and a
    network call there would fail with fake credentials.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pdf: PineconeRetriever | None = None
        self._web: WebSearchService | None = None
        self._pdf_built = False
        self._web_built = False

    def _ensure_pdf(self) -> PineconeRetriever | None:
        if self._pdf_built:
            return self._pdf
        self._pdf_built = True
        if not self._settings.PINECONE_API_KEY:
            return None
        try:
            self._pdf = PineconeRetriever(self._settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rag_pdf_init_failed error_type=%s", type(exc).__name__
            )
            self._pdf = None
        return self._pdf

    def _ensure_web(self) -> WebSearchService | None:
        if self._web_built:
            return self._web
        self._web_built = True
        if not self._settings.TAVILY_API_KEY:
            return None
        try:
            self._web = WebSearchService(self._settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "rag_web_init_failed error_type=%s", type(exc).__name__
            )
            self._web = None
        return self._web

    def _llm_for_judge(self) -> "LLMService":
    # Lazy + cached. Constructing a new LLMService per request is
    # wasteful; the constructor sets module-level litellm flags.
        if not hasattr(self, "_judge_llm"):
            from app.services.llm import LLMService
            self._judge_llm = LLMService(self._settings)
        return self._judge_llm

    async def retrieve(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        pdf_chunks: list[RetrievedChunk] = []
        pdf = self._ensure_pdf()
        if pdf is not None:
            try:
                pdf_chunks = await asyncio.to_thread(pdf.retrieve, query, top_k)
            except Exception as exc:
                logger.warning(
                    "rag_pdf_retrieve_failed error_type=%s", type(exc).__name__
                )

        top_score = pdf_chunks[0].score if pdf_chunks else None

        # Numeric relevance judge (1-5) with threshold.
        # Fires Tavily when the judge says the chunks do not clearly answer.
        pdf_answers = False
        if len(pdf_chunks) >= MIN_PDF_HITS:
            judge_score = await score_chunks_for_question(
                query, pdf_chunks, self._llm_for_judge()
            )
            pdf_answers = judge_score >= JUDGE_ANSWER_THRESHOLD
            logger.info(
                "rag_relevance judge_score=%d threshold=%d answers=%s",
                judge_score, JUDGE_ANSWER_THRESHOLD, pdf_answers,
            )

        if pdf_answers:
            logger.info(
                "rag_pdf_answers hits=%d top_score=%.3f (no web fallback)",
                len(pdf_chunks),
                top_score if top_score is not None else -1.0,
            )
            return pdf_chunks

        web_chunks: list[RetrievedChunk] = []
        web = self._ensure_web()
        if web is not None:
            try:
                web_chunks = await web.search(
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