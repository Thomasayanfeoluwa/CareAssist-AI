from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    source: str  # "pdf" | "web"
    title: str
    content: str
    page: int | None = None
    link: str | None = None


class RAGService:
    """Stage 4 wires Pinecone + Google CSE here."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        return []

    def assemble_context(self, chunks: list[RetrievedChunk]) -> str:
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