from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = [
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "PINECONE_API_KEY",
    "GOOGLE_CSE_API_KEY",
    "GOOGLE_CSE_ID",
    "SESSION_SECRET",
    "PILOT_AUTH_TOKEN",
]


@pytest.mark.parametrize("missing", REQUIRED)
def test_missing_required_var_raises_with_name(monkeypatch, missing) -> None:
    for var in REQUIRED:
        monkeypatch.delenv(var, raising=False)
    for var in REQUIRED:
        if var != missing:
            monkeypatch.setenv(var, "x" * 32)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert missing in str(excinfo.value)