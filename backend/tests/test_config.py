from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = [
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "PINECONE_API_KEY",
    "TAVILY_API_KEY",
    "SESSION_SECRET",
    "PILOT_AUTH_TOKEN",
]


def _all_required() -> dict[str, str]:
    return {
        "GROQ_API_KEY": "x",
        "OPENROUTER_API_KEY": "x",
        "PINECONE_API_KEY": "x",
        "TAVILY_API_KEY": "x",
        "SESSION_SECRET": "y" * 32,
        "PILOT_AUTH_TOKEN": "pilot-token-XYZ12",
    }


@pytest.mark.parametrize("missing", REQUIRED)
def test_missing_required_var_raises_with_name(monkeypatch, missing: str) -> None:
    # Clear every required var, then set all except the one under test
    for var in REQUIRED:
        monkeypatch.delenv(var, raising=False)
    for var, val in _all_required().items():
        if var != missing:
            monkeypatch.setenv(var, val)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]

    msg = str(excinfo.value)
    assert missing in msg, (
        f"ValidationError did not name {missing!r}; got: {msg}"
    )


def test_all_required_present_instantiates(monkeypatch) -> None:
    for var, val in _all_required().items():
        monkeypatch.setenv(var, val)
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    # Index name is deployment-configurable; only assert it is a non-empty string.
    assert isinstance(s.PINECONE_INDEX, str)
    assert len(s.PINECONE_INDEX) > 0
    # Model IDs are deployment-configurable; only assert shape, not exact value.
    assert s.LITELLM_PRIMARY_MODEL.startswith(("groq/", "openrouter/"))
    assert "/" in s.LITELLM_PRIMARY_MODEL
    assert s.LLM_TEMPERATURE == 0.3


def test_session_secret_min_length(monkeypatch) -> None:
    for var, val in _all_required().items():
        monkeypatch.setenv(var, val)
    monkeypatch.setenv("SESSION_SECRET", "short")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "SESSION_SECRET" in str(excinfo.value)


def test_pilot_auth_token_min_length(monkeypatch) -> None:
    for var, val in _all_required().items():
        monkeypatch.setenv(var, val)
    monkeypatch.setenv("PILOT_AUTH_TOKEN", "short")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "PILOT_AUTH_TOKEN" in str(excinfo.value)


def test_log_level_validated(monkeypatch) -> None:
    for var, val in _all_required().items():
        monkeypatch.setenv(var, val)
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "LOG_LEVEL" in str(excinfo.value)


def test_cors_origins_whitespace_stripped(monkeypatch) -> None:
    for var, val in _all_required().items():
        monkeypatch.setenv(var, val)
    monkeypatch.setenv("CORS_ORIGINS", "  http://a.example  ,  http://b.example  ")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.cors_origin_list == ["http://a.example", "http://b.example"]


def test_llm_temperature_override(monkeypatch) -> None:
    for var, val in _all_required().items():
        monkeypatch.setenv(var, val)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.LLM_TEMPERATURE == 0.0