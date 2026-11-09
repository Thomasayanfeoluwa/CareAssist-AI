from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Baseline env for all tests. Individual tests may delete keys to exercise
# fail-fast behaviour; the autouse fixture below re-applies this each test.
_REQUIRED = {
    "GROQ_API_KEY": "test-groq",
    "OPENROUTER_API_KEY": "test-openrouter",
    "PINECONE_API_KEY": "test-pinecone",
    "GOOGLE_CSE_API_KEY": "test-cse-key",
    "GOOGLE_CSE_ID": "test-cse-id",
    "SESSION_SECRET": "x" * 32,
    "PILOT_AUTH_TOKEN": "pilot-token-XYZ12",
    "LLM_MOCK": "true",
}


@pytest.fixture(autouse=True)
def _set_env(monkeypatch) -> None:
    for k, v in _REQUIRED.items():
        monkeypatch.setenv(k, v)
    # Clear the lru_cache so each test sees the fresh env
    from app.config import get_settings
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    from app.ratelimit import limiter
    limiter.reset()


@pytest.fixture
def client() -> TestClient:
    from app.config import get_settings
    from app.main import create_app
    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer pilot-token-XYZ12"}