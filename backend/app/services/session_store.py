from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings


@dataclass
class HistoryEntry:
    role: str
    content: str


class SessionStore:
    """Stage 3 replaces with Redis-backed per-session store."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def get(self, session_id: str) -> list[HistoryEntry]:
        return []

    async def append(self, session_id: str, role: str, content: str) -> None:
        return None