from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)

MAX_EXCHANGES = 5
MAX_ENTRIES = MAX_EXCHANGES * 2  # each exchange = user + assistant


@dataclass
class HistoryEntry:
    role: str  # "user" or "assistant"
    content: str


class SessionStore(Protocol):
    async def get(self, session_id: str) -> list[HistoryEntry]: ...
    async def append(self, session_id: str, role: str, content: str) -> None: ...
    async def clear(self, session_id: str) -> None: ...


def _cap(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    if len(entries) <= MAX_ENTRIES:
        return entries
    return entries[-MAX_ENTRIES:]


@dataclass
class _Bucket:
    entries: list[HistoryEntry] = field(default_factory=list)
    expires_at: float = 0.0


class InMemorySessionStore:
    """In-process store for dev and single-worker tests."""

    def __init__(self, settings: Settings) -> None:
        self._ttl = settings.SESSION_TTL_SECONDS
        self._lock = asyncio.Lock()
        self._buckets: dict[str, _Bucket] = {}

    async def get(self, session_id: str) -> list[HistoryEntry]:
        async with self._lock:
            bucket = self._buckets.get(session_id)
            if bucket is None:
                return []
            if bucket.expires_at <= time.monotonic():
                del self._buckets[session_id]
                return []
            return list(bucket.entries)

    async def append(self, session_id: str, role: str, content: str) -> None:
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.get(session_id)
            if bucket is None or bucket.expires_at <= now:
                bucket = _Bucket()
                self._buckets[session_id] = bucket
            bucket.entries.append(HistoryEntry(role=role, content=content))
            bucket.entries = _cap(bucket.entries)
            bucket.expires_at = now + self._ttl

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._buckets.pop(session_id, None)


class RedisSessionStore:
    """Redis-backed store for production. Session isolation is per Redis key."""

    _PREFIX = "careassist:session:"
    _SUFFIX = ":history"

    def __init__(self, settings: Settings) -> None:
        # Imported lazily so this module stays importable in environments
        # that do not install `redis` (e.g. minimal test containers).
        import redis.asyncio as redis_asyncio
        self._ttl = settings.SESSION_TTL_SECONDS
        self._client = redis_asyncio.from_url(
            settings.REDIS_URL, decode_responses=True
        )

    def _key(self, session_id: str) -> str:
        return f"{self._PREFIX}{session_id}{self._SUFFIX}"

    async def get(self, session_id: str) -> list[HistoryEntry]:
        raw = await self._client.lrange(self._key(session_id), 0, -1)
        out: list[HistoryEntry] = []
        for item in raw:
            try:
                data = json.loads(item)
                role = data["role"]
                content = data["content"]
                if role in ("user", "assistant") and isinstance(content, str):
                    out.append(HistoryEntry(role=role, content=content))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "session_entry_parse_failed error_type=%s", type(exc).__name__
                )
        return out

    async def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        payload = json.dumps({"role": role, "content": content})
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -MAX_ENTRIES, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._client.delete(self._key(session_id))


def build_session_store(settings: Settings) -> SessionStore:
    """Pick the store implementation based on settings.REDIS_URL."""
    if settings.REDIS_URL:
        logger.info("session_store_backend redis")
        return RedisSessionStore(settings)
    logger.info("session_store_backend memory")
    return InMemorySessionStore(settings)