"""Application configuration — loaded from environment variables and/or a .env file.

All runtime settings live here in a single ``Settings`` object so every other
module imports *one* thing (``from pokemon_forecaster.config import settings``)
instead of scattering ``os.getenv`` calls throughout the codebase.

How it works
------------
``pydantic-settings`` reads values in this priority order (highest wins):

1. Real environment variables  (e.g. ``DATABASE_URL=postgresql://…``)
2. A ``.env`` file at the project root               (local dev convenience)
3. The ``Field(default=…)`` fallback                 (sensible defaults)

Each field has an ``alias`` that is the *environment variable name* — this
lets you use the conventional ALL_CAPS names in shell / Docker / Kubernetes
without the Python attribute having to match.

Example .env (copy from .env.example):

    POKEMONTCG_API_KEY=abc123
    DATABASE_URL=sqlite:///data/prices.db
    MODEL_DIR=models
    API_PORT=8000
    LOG_LEVEL=INFO

Kubernetes / Docker
-------------------
Inject secrets via ``secretKeyRef`` in your pod spec (see k8s/deployment.yaml
and k8s/cronjob-ingest.yaml).  The ``extra="ignore"`` setting means any env
vars that aren't declared below are silently skipped — useful in environments
that inject many unrelated variables.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved absolute path to the repository root regardless of where
# Python is invoked from.  Used as the anchor for relative default paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All application settings, validated at import time by Pydantic.

    Validation happens once when the module is first imported.  If a required
    field is missing or has the wrong type, Pydantic raises ``ValidationError``
    immediately — you find out at startup, not mid-request.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",  # load from .env if present
        env_file_encoding="utf-8",
        extra="ignore",  # silently drop unknown env vars
    )

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------

    pokemontcg_api_key: str | None = Field(
        default=None,
        alias="POKEMONTCG_API_KEY",
        description=(
            "Optional API key for pokemontcg.io.  Without it you're limited to "
            "~1 000 requests/day; with a free key that rises to ~20 000."
        ),
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'data' / 'prices.db'}",
        alias="DATABASE_URL",
        description=(
            "SQLAlchemy connection string.  Defaults to a local SQLite file for "
            "development.  In production, use Postgres: "
            "'postgresql://user:pass@host:5432/forecaster'."
        ),
    )

    # ------------------------------------------------------------------
    # Model artefacts
    # ------------------------------------------------------------------

    model_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        alias="MODEL_DIR",
        description=(
            "Directory where trained .joblib model files are stored.  "
            "In Kubernetes, mount a PersistentVolumeClaim here so the API "
            "pod and the training pod share the same artefacts."
        ),
    )

    # ------------------------------------------------------------------
    # API server
    # ------------------------------------------------------------------

    api_host: str = Field(
        default="0.0.0.0",
        alias="API_HOST",
        description="Interface the Uvicorn server binds to.  0.0.0.0 = all interfaces.",
    )

    api_port: int = Field(
        default=8000,
        alias="API_PORT",
        description="TCP port the FastAPI app listens on.",
    )

    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )


# Module-level singleton — import this in every other module.
# Pydantic validates and coerces all fields on first access.
settings = Settings()
