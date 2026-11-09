from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """One piece of context retrieved for a query.

    source is "pdf" or "web".
    - pdf chunks: title + page + content; link/score optional
    - web chunks: title + link + content; page/score optional
    """
    source: str
    title: str
    content: str
    page: int | None = None
    link: str | None = None
    score: float | None = None
    chunk_id: str | None = None