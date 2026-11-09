from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    """One piece of context retrieved for a query.

    source is "pdf" or "web". For pdf chunks, title and page should be set.
    For web chunks, title should be the site name and link the URL.
    """
    source: str
    title: str
    content: str
    page: int | None = None
    link: str | None = None


class RAGService:
    """Stage 4 wires Pinecone retrieval and Google CSE fallback here."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        return []

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