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
    """LiteLLM wrapper: Groq primary, OpenRouter fallback.
    
    Privacy contract: this module never logs user prompt or model output
    content. Only model name, fallback flag, latency, and error *type*.
    """

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
        except Exception as exc:
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
    
    async def _call(self, model: str, messages: list[dict[str, str]]) -> Any:
        return await litellm.acompletion(
            model=model, messages=messages, temperature=self._settings.LLM_TEMPERATURE,
        )