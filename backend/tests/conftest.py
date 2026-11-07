from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

_REQUIRED = {
    "GROQ_API_KEY": "test-groq",
    "OPENROUTER_API_KEY": "test-openrouter",
    "PINECONE_API_KEY": "test-pinecone",
    "GOOGLE_CSE_API_KEY": "test-cse-key",
    "GOOGLE_CSE_ID": "test-cse-id",
    "SESSION_SECRET": "x" * 32,
    "PILOT_AUTH_TOKEN": "test-pilot-token-1234",  # noqa: S105
    "LLM_MOCK": "true",
}


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    from app.ratelimit import limiter
    limiter.reset()


@pytest.fixture
def client(monkeypatch) -> TestClient:
    for k, v in _REQUIRED.items():
        monkeypatch.setenv(k, v)
    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-pilot-token-1234"}