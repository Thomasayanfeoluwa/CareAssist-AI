from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import litellm

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    fallback_used: bool
    latency_ms: int


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        litellm.suppress_debug_info = True
        litellm.drop_params = True

    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.perf_counter()
        fallback = False
        try:
            resp = await self._call(self._settings.LITELLM_PRIMARY_MODEL, messages)
            model = self._settings.LITELLM_PRIMARY_MODEL
        except Exception as exc:  # noqa: BLE001 — any failure triggers fallback
            logger.warning(
                "llm_primary_failed model=%s error_type=%s",
                self._settings.LITELLM_PRIMARY_MODEL,
                type(exc).__name__,
            )
            resp = await self._call(self._settings.LITELLM_FALLBACK_MODEL, messages)
            model = self._settings.LITELLM_FALLBACK_MODEL
            fallback = True
        latency_ms = int((time.perf_counter() - start) * 1000)
        content: str = resp.choices[0].message.content or ""
        # NOTE: do not log message content — see privacy requirements.
        logger.info(
            "llm_call model=%s fallback=%s latency_ms=%d",
            model, fallback, latency_ms,
        )
        return LLMResult(
            content=content,
            model=model,
            fallback_used=fallback,
            latency_ms=latency_ms,
        )

    @staticmethod
    async def _call(model: str, messages: list[dict[str, str]]) -> Any:
        return await litellm.acompletion(
            model=model, messages=messages, temperature=0.3
        )