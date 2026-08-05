"""Runtime config loaded from .env (see .env.example)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Providers
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_provider: str = "gemini"  # gemini | groq

    # Retrieval
    embed_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6

    # Reliability under free-tier rate limits
    llm_max_retries: int = 5
    llm_backoff_base: float = 2.0

    # Parallelism (finals tie-break is submission time → wall-clock matters).
    # Modest default so concurrent LLM calls don't trip free-tier rate limits.
    max_workers: int = 4


settings = Settings()
