from __future__ import annotations

import asyncio
import logging
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import Settings
from app.services.types import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_NUM_RESULTS = 5


class WebSearchService:
    """Google Custom Search wrapper. Failures return [] — never raise."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._service: Any | None = None
        self._init_attempted = False

    def _get_service(self) -> Any | None:
        if self._init_attempted:
            return self._service
        self._init_attempted = True
        if not (self._settings.GOOGLE_CSE_API_KEY and self._settings.GOOGLE_CSE_ID):
            return None
        try:
            self._service = build(
                "customsearch",
                "v1",
                developerKey=self._settings.GOOGLE_CSE_API_KEY,
                cache_discovery=False,
            )
        except Exception as exc:
            logger.warning(
                "web_search_init_failed error_type=%s", type(exc).__name__
            )
            return None
        return self._service

    def search_sync(
        self, query: str, num_results: int = DEFAULT_NUM_RESULTS
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        service = self._get_service()
        if service is None:
            return []
        try:
            res = (
                service.cse()
                .list(
                    q=query,
                    cx=self._settings.GOOGLE_CSE_ID,
                    num=min(max(num_results, 1), 10),
                )
                .execute()
            )
        except HttpError as exc:
            logger.warning("web_search_http_error status=%s", exc.status_code)
            return []
        except Exception as exc:
            logger.warning(
                "web_search_failed error_type=%s", type(exc).__name__
            )
            return []

        items = res.get("items", []) or []
        out: list[RetrievedChunk] = []
        for item in items:
            link = item.get("link", "").strip()
            title = item.get("title", "Unknown source").strip() or "Unknown source"
            snippet = item.get("snippet", "").strip()
            if not link or not snippet:
                continue
            out.append(
                RetrievedChunk(
                    source="web",
                    title=title,
                    content=snippet,
                    link=link,
                )
            )
        logger.info("web_search_ok query_len=%d results=%d", len(query), len(out))
        return out

    async def search(
        self, query: str, num_results: int = DEFAULT_NUM_RESULTS
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.search_sync, query, num_results)