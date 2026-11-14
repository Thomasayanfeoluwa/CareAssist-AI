from __future__ import annotations

import logging
from typing import Any

from tavily import AsyncTavilyClient, TavilyClient

from app.config import Settings
from app.services.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_NUM_RESULTS = 5
_MAX_RESULTS = 20
_REQUEST_TIMEOUT = 15.0


class WebSearchService:
    """Tavily web search wrapper. Failures return [] — never raise.

    Basic search depth costs 1 credit. Free tier: 1,000 credits/month.
    We use the native async client for the route path, and the sync client
    for tests and for any synchronous callers.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        key = settings.TAVILY_API_KEY
        self._sync: TavilyClient | None = TavilyClient(api_key=key) if key else None
        self._async: AsyncTavilyClient | None = (
            AsyncTavilyClient(api_key=key) if key else None
        )

    @staticmethod
    def _clamp(n: int) -> int:
        return min(max(int(n), 1), _MAX_RESULTS)

    @staticmethod
    def _parse(resp: dict[str, Any]) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for item in (resp.get("results") or []):
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").strip()
            title = (item.get("title") or "Unknown source").strip() or "Unknown source"
            if not url or not content:
                continue
            raw_score = item.get("score")
            score = float(raw_score) if isinstance(raw_score, int | float) else None
            out.append(
                RetrievedChunk(
                    source="web",
                    title=title,
                    content=content,
                    link=url,
                    score=score,
                )
            )
        return out

    def search_sync(
        self, query: str, num_results: int = DEFAULT_NUM_RESULTS
    ) -> list[RetrievedChunk]:
        if not query.strip() or self._sync is None:
            return []
        try:
            resp = self._sync.search(
                query=query,
                max_results=self._clamp(num_results),
                search_depth="basic",
                timeout=_REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "web_search_failed error_type=%s", type(exc).__name__
            )
            return []
        chunks = self._parse(resp)
        logger.info(
            "web_search_ok provider=tavily query_len=%d results=%d",
            len(query),
            len(chunks),
        )
        return chunks

    async def search(
        self, query: str, num_results: int = DEFAULT_NUM_RESULTS
    ) -> list[RetrievedChunk]:
        if not query.strip() or self._async is None:
            return []
        try:
            resp = await self._async.search(
                query=query,
                max_results=self._clamp(num_results),
                search_depth="basic",
                timeout=_REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "web_search_failed error_type=%s", type(exc).__name__
            )
            return []
        chunks = self._parse(resp)
        logger.info(
            "web_search_ok provider=tavily query_len=%d results=%d",
            len(query),
            len(chunks),
        )
        return chunks