"""Application configuration loaded from environment variables / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pokemontcg_api_key: str | None = Field(default=None, alias="POKEMONTCG_API_KEY")
    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'data' / 'prices.db'}",
        alias="DATABASE_URL",
    )
    model_dir: Path = Field(default=PROJECT_ROOT / "models", alias="MODEL_DIR")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()
