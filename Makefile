.PHONY: install dev fmt lint test ingest train api ui docker up down clean

install:
	uv sync

dev:
	uv sync --extra dev --extra frontend

fmt:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

test:
	uv run pytest

ingest:
	uv run pokemon-ingest --query 'set.id:swsh1' --batch-size 50 --limit 250

train:
	uv run pokemon-train --min-history 30

api:
	uv run pokemon-api

ui:
	uv run --extra frontend streamlit run src/pokemon_forecaster/frontend/streamlit_app.py

docker:
	docker build -t pokemon-forecaster:local .

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
