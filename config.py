"""
Configuration management for LLM Inference Server.
All settings are configurable via environment variables with the LLM_ prefix.
Example: LLM_MAX_BATCH_SIZE=16 overrides the default of 8.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Model Configuration ─────────────────────────────────────────────────
    model_name: str = Field(
        default="sshleifer/tiny-gpt2",
        description="HuggingFace model identifier. Use a small model for CPU."
    )
    max_new_tokens: int = Field(
        default=50,
        description="Maximum number of new tokens to generate per request."
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature. 0.0 = greedy (deterministic, cacheable)."
    )

    # ── Batching Configuration ───────────────────────────────────────────────
    max_batch_size: int = Field(
        default=8,
        description="Maximum number of requests to process in a single batch."
    )
    batch_timeout_ms: float = Field(
        default=50.0,
        description="Max milliseconds to wait for a batch to fill before processing."
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum total token length. Longer prompts are truncated."
    )

    # ── Caching Configuration ────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for cached responses in seconds."
    )
    cache_max_entries: int = Field(
        default=10000,
        description="Maximum number of entries in the in-process cache (LRU eviction)."
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable or disable response caching entirely."
    )

    # ── Server Configuration ─────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Server bind host.")
    port: int = Field(default=8000, description="Server bind port.")
    log_level: str = Field(default="info", description="Uvicorn log level.")

    model_config = {"env_prefix": "LLM_", "case_sensitive": False}


# Singleton settings instance used across the application
settings = Settings()
