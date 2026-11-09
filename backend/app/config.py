from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Required, no defaults (fail-fast if missing) ---
    GROQ_API_KEY: Annotated[str, Field(min_length=1)]
    OPENROUTER_API_KEY: Annotated[str, Field(min_length=1)]
    PINECONE_API_KEY: Annotated[str, Field(min_length=1)]
    TAVILY_API_KEY: Annotated[str, Field(min_length=1)]
    SESSION_SECRET: Annotated[str, Field(min_length=32)]
    PILOT_AUTH_TOKEN: Annotated[str, Field(min_length=16)]

    # --- Required, with defaults ---
    PINECONE_INDEX: str = "carebot"
    LITELLM_PRIMARY_MODEL: str = "groq/llama-3.3-70b-versatile"
    LITELLM_FALLBACK_MODEL: str = "openrouter/meta-llama/llama-3.3-70b-instruct"
    # REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_URL: str = ""   # empty -> in-memory store (dev/tests); set in prod (Render)
    CORS_ORIGINS: str = "http://localhost:5173"
    

    # --- Ops ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LLM_MOCK: bool = False
    APP_VERSION: str = "0.1.0"
    REQUEST_MAX_BYTES: int = 16 * 1024
    RATE_LIMIT_PER_MINUTE: int = 30
    LLM_TEMPERATURE: float = 0.3
    RAG_FALLBACK_SCORE_THRESHOLD: float = 0.4
    RAG_WEB_RESULTS: int = 5

    # --- Session ---
    SESSION_TTL_SECONDS: int = 1800          # 30 min, matches cookie Max-Age
    SESSION_COOKIE_SECURE: bool = False      # set True on Render (HTTPS only)

    @field_validator("CORS_ORIGINS")
    @classmethod
    def _normalize_origins(cls, v: str) -> str:
        return ",".join(o.strip() for o in v.split(",") if o.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o for o in self.CORS_ORIGINS.split(",") if o]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()