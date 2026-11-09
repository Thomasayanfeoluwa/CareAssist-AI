from __future__ import annotations

from app.config import Settings


class WebSearchService:
    """Stage 4 wires Google Custom Search here."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(self, query: str, num_results: int = 5) -> list[dict[str, str]]:
        return []