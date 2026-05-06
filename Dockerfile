# syntax=docker/dockerfile:1.7
# Multi-stage build using uv for fast, reproducible installs.

############################
# Stage 1: builder
############################
FROM python:3.11-slim AS builder

# Install uv from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System deps Prophet/cmdstan need at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy lockfiles first so the install layer caches across code changes
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install only runtime deps — no dev/frontend
RUN uv sync --frozen --no-dev --no-editable --extra frontend || uv sync --no-dev --no-editable --extra frontend

############################
# Stage 2: runtime
############################
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=app:app src ./src

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "pokemon_forecaster.api.main"]
