from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.services.session_store import (
    MAX_ENTRIES,
    HistoryEntry,
    InMemorySessionStore,
)


def _settings(monkeypatch, ttl: int = 1800) -> Settings:
    monkeypatch.setenv("SESSION_TTL_SECONDS", str(ttl))
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_empty_session_returns_empty_list(client) -> None:
    store = client.app.state.session_store
    assert asyncio.run(store.get("sess-empty")) == []


def test_append_and_get_roundtrip_preserves_order(client) -> None:
    store = client.app.state.session_store
    asyncio.run(store.append("sess-A", "user", "hello"))
    asyncio.run(store.append("sess-A", "assistant", "hi there"))
    entries = asyncio.run(store.get("sess-A"))
    assert entries == [
        HistoryEntry(role="user", content="hello"),
        HistoryEntry(role="assistant", content="hi there"),
    ]


def test_sessions_are_isolated(client) -> None:
    store = client.app.state.session_store
    asyncio.run(store.append("sess-A", "user", "A-content"))
    asyncio.run(store.append("sess-B", "user", "B-content"))
    assert [e.content for e in asyncio.run(store.get("sess-A"))] == ["A-content"]
    assert [e.content for e in asyncio.run(store.get("sess-B"))] == ["B-content"]


def test_cap_trims_to_last_ten(client) -> None:
    store = client.app.state.session_store
    for i in range(6):
        asyncio.run(store.append("sess-A", "user", f"u{i}"))
        asyncio.run(store.append("sess-A", "assistant", f"a{i}"))
    entries = asyncio.run(store.get("sess-A"))
    assert len(entries) == MAX_ENTRIES
    assert entries[0].content == "u1"
    assert entries[-1].content == "a5"


def test_clear_removes_history(client) -> None:
    store = client.app.state.session_store
    asyncio.run(store.append("sess-A", "user", "hi"))
    assert len(asyncio.run(store.get("sess-A"))) == 1
    asyncio.run(store.clear("sess-A"))
    assert asyncio.run(store.get("sess-A")) == []


def test_ttl_expiry(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_TTL_SECONDS", "1")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    store = InMemorySessionStore(s)

    async def scenario() -> None:
        await store.append("sess-A", "user", "hi")
        assert len(await store.get("sess-A")) == 1
        await asyncio.sleep(1.1)
        assert await store.get("sess-A") == []

    asyncio.run(scenario())


def test_max_entries_constant_is_five_exchanges() -> None:
    assert MAX_ENTRIES == 10