# Architecture & Technical Reference

> **Pokémon TCG Card Price Forecaster** — a full-stack ML system that ingests daily price data from the Pokémon TCG API, builds per-card time-series forecasting models, serves predictions via a REST API, and visualises them in a Streamlit web UI.  Containerised with Docker, deployed to Kubernetes, with CI/CD via GitHub Actions.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Map](#2-component-map)
3. [Data Pipeline](#3-data-pipeline)
4. [Database Schema](#4-database-schema)
5. [Feature Engineering](#5-feature-engineering)
6. [Forecasting Models](#6-forecasting-models)
7. [Model Validation & Evaluation](#7-model-validation--evaluation)
8. [REST API Contract](#8-rest-api-contract)
9. [Configuration & Secrets](#9-configuration--secrets)
10. [Docker & Docker Compose](#10-docker--docker-compose)
11. [Kubernetes Deployment](#11-kubernetes-deployment)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [Local Development Guide](#13-local-development-guide)
14. [Design Decisions & Trade-offs](#14-design-decisions--trade-offs)
15. [Interview Q&A Reference](#15-interview-qa-reference)

---

## 1. System Overview

The system solves a data-availability problem: the Pokémon TCG API ([pokemontcg.io](https://pokemontcg.io)) only exposes *current* prices — there is no historical endpoint.  The only way to produce a price forecast is to run a scheduled ingest job that snapshots prices every day and accumulates a time series over weeks and months.

**Core loop:**

```
[pokemontcg.io API]
       │  HTTP GET (daily, 06:00 UTC)
       ▼
[Ingest Job (CronJob)]  ──▶  [SQLite / Postgres DB]
                                      │
                              [Batch Trainer Job]
                                      │  .joblib artefacts
                                      ▼
                              [Model Store (disk/PVC)]
                                      │
                              [FastAPI Service]  ◀──  [Streamlit Frontend]
                                      │
                              [POST /predict]  ──▶  JSON forecast response
```

---

## 2. Component Map

| Component | Location | Purpose |
|---|---|---|
| `config.py` | `src/pokemon_forecaster/config.py` | Pydantic settings — reads env vars / `.env` |
| `data/client.py` | `src/.../data/client.py` | HTTP client for pokemontcg.io (httpx + tenacity retries) |
| `data/storage.py` | `src/.../data/storage.py` | SQLAlchemy ORM models + DAO (`PriceStore`) |
| `features/engineering.py` | `src/.../features/engineering.py` | Lag, rolling, calendar, set-release features |
| `models/forecaster.py` | `src/.../models/forecaster.py` | `Forecaster` ABC + `ProphetForecaster` |
| `api/main.py` | `src/.../api/main.py` | FastAPI app — `/health`, `/predict` endpoints |
| `api/schemas.py` | `src/.../api/schemas.py` | Pydantic request/response models |
| `cli/ingest.py` | `src/.../cli/ingest.py` | Daily ingest CLI entry point |
| `cli/train.py` | `src/.../cli/train.py` | Batch model training CLI |
| `frontend/streamlit_app.py` | `src/.../frontend/streamlit_app.py` | Streamlit UI |
| `Dockerfile` | `/Dockerfile` | Multi-stage Docker build |
| `docker-compose.yml` | `/docker-compose.yml` | Local multi-container stack |
| `k8s/` | `/k8s/` | Kubernetes manifests |
| `.github/workflows/ci-cd.yml` | `/.github/workflows/` | GitHub Actions CI/CD |

---

## 3. Data Pipeline

### 3.1 Source — pokemontcg.io API

The Pokémon TCG API is a free, public REST API.  Key endpoint:

```
GET https://api.pokemontcg.io/v2/cards?q=set.id:swsh1&page=1&pageSize=250
```

Each card object includes a `tcgplayer.prices` block:

```json
{
  "id": "swsh1-1",
  "name": "Caterpie",
  "tcgplayer": {
    "prices": {
      "normal":    { "market": 0.12, "low": 0.05, "mid": 0.10, "high": 0.25 },
      "holofoil":  { "market": 8.50, "low": 6.00, "mid": 8.00, "high": 12.00 }
    }
  }
}
```

### 3.2 Ingest Job

**Entry point:** `pokemon_forecaster.cli.ingest:main`  
**CLI:** `uv run pokemon-ingest --query 'set.id:swsh1' --batch-size 50`  
**Kubernetes:** `k8s/cronjob-ingest.yaml` runs at `0 6 * * *` (06:00 UTC daily)

The ingest job:
1. Pages through all cards matching the query (250 cards per HTTP request).
2. Upserts each card's metadata into the `cards` table.
3. Inserts one `price_snapshot` row per (card_id, variant) with today's prices.
4. Commits every `--batch-size` cards (default 50) to bound memory usage.

**Idempotency:** A `UNIQUE` constraint on `(card_id, variant, snapshot_date)` prevents duplicate rows if the job runs twice in one day.

### 3.3 Building History

The API only gives current prices.  History accumulates one row at a time:

| Day | swsh1-1 holofoil market |
|-----|------------------------|
| 2024-01-01 | $8.50 |
| 2024-01-02 | $8.75 |
| 2024-01-03 | $9.10 |
| … | … |

After **30 days** you have enough to run Prophet.  After **60 days** the seasonality estimates become reliable.  After **90 days** XGBoost with lag features becomes viable.

**Bootstrapping options:**
- Forward-fill (run daily ingest for 1–2 months — honest but slow).
- Backfill from [PriceCharting](https://www.pricecharting.com/) historical exports.

---

## 4. Database Schema

SQLite in development; Postgres in production.  Change `DATABASE_URL` — no code changes needed (SQLAlchemy is DB-agnostic).

### `cards` table

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | e.g. `swsh1-1` — pokemontcg.io canonical ID |
| `name` | TEXT | Card name, e.g. `"Caterpie"` |
| `set_id` | TEXT (indexed) | Set identifier, e.g. `"swsh1"` |
| `set_name` | TEXT | Human-readable set name |
| `rarity` | TEXT nullable | `"Common"`, `"Rare Holo"`, etc. |
| `number` | TEXT nullable | Collector number within the set |
| `release_date` | DATE nullable | Set release date — used for `days_since_release` feature |

### `price_snapshots` table

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK (autoincrement) | |
| `card_id` | TEXT FK → cards.id (indexed) | |
| `variant` | TEXT (indexed) | `"holofoil"`, `"normal"`, `"reverseHolofoil"`, … |
| `snapshot_date` | DATE (indexed) | Date this price was captured — becomes `ds` in the model |
| `market` | FLOAT nullable | TCGPlayer market price in USD (volume-weighted average of recent sales) |
| `low` | FLOAT nullable | Lowest listed price |
| `mid` | FLOAT nullable | Mid price |
| `high` | FLOAT nullable | Highest listed price |
| `direct_low` | FLOAT nullable | TCGPlayer Direct lowest price |
| `captured_at` | DATETIME | Wall-clock UTC timestamp of insertion |

**Unique constraint:** `(card_id, variant, snapshot_date)` — prevents duplicate daily snapshots.

**Why long format?** One row per (card, variant, date) — the shape Prophet, pandas, and most forecasting libraries expect.  Filtering by `(card_id, variant)` gives you a ready-to-use univariate time series.

---

## 5. Feature Engineering

> **Note:** Feature engineering is used by future XGBoost/LSTM backends.  Prophet (the current baseline) learns seasonality internally from raw `(ds, y)` data and does not require a feature matrix.

All functions live in `features/engineering.py`.  Call `build_feature_frame(df)` to run the full pipeline.

### 5.1 Lag Features (`add_lag_features`)

| Feature | Formula | Captures |
|---|---|---|
| `market_lag_1` | price at t−1 | Yesterday's price — strongest single predictor |
| `market_lag_7` | price at t−7 | Same day last week (weekly cycle) |
| `market_lag_14` | price at t−14 | Two weeks ago |
| `market_lag_30` | price at t−30 | Same day last month |

Implemented with `pd.Series.shift(n)` — no look-ahead bias.

### 5.2 Rolling Statistics (`add_rolling_features`)

| Feature | Formula | Captures |
|---|---|---|
| `market_rollmean_7` | mean(t−8 … t−1) | 7-day local trend |
| `market_rollstd_7` | std(t−8 … t−1) | 7-day volatility |
| `market_rollmean_14` | mean(t−15 … t−1) | 2-week trend |
| `market_rollstd_14` | std(t−15 … t−1) | 2-week volatility |
| `market_rollmean_30` | mean(t−31 … t−1) | Monthly trend |
| `market_rollstd_30` | std(t−31 … t−1) | Monthly volatility |

All rolling windows use `shift(1)` before rolling to prevent the current day's price from leaking into its own features.

### 5.3 Calendar Features (`add_calendar_features`)

| Feature | Values | Captures |
|---|---|---|
| `dow` | 0–6 (Mon–Sun) | Weekly cycle — new reveals often hit Fridays |
| `month` | 1–12 | Monthly / seasonal patterns |
| `is_weekend` | 0 / 1 | Weekend trading behaviour |

### 5.4 Domain Feature: Days Since Release (`add_set_features`)

```
days_since_release = snapshot_date - set.release_date
```

New cards consistently trade at a premium for 2–4 weeks post-release ("hype premium"), then revert toward a stable value as supply catches up with demand.  This feature is typically the top-3 most important in XGBoost feature importances.

### 5.5 Data Leakage Prevention

Every feature that involves the target column uses a **1-day shift** before computation.  This ensures the feature value for day `t` is computed only from data available *strictly before* day `t`.  Forgetting this causes "perfect" training metrics that collapse to random on live data.

---

## 6. Forecasting Models

### 6.1 `Forecaster` Abstract Base Class

```python
class Forecaster(ABC):
    def fit(self, history: pd.DataFrame) -> None: ...
    def predict(self, horizon_days: int) -> pd.DataFrame: ...
    def save(self, path: Path) -> None: ...
    def load(cls, path: Path) -> Forecaster: ...
```

Every backend implements this interface.  The FastAPI endpoint calls only `fit` and `predict`, so adding a new model backend requires zero changes to the API.

### 6.2 `ProphetForecaster` (current baseline)

**Library:** [Facebook Prophet](https://facebook.github.io/prophet/)  
**Model form:** `y(t) = trend(t) + seasonality(t) + ε`

Prophet fits:
- A piecewise-linear trend with automatic changepoint detection
- Weekly Fourier seasonality (8 terms)
- Yearly Fourier seasonality when enough data exists (auto-enabled)

**Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| `daily_seasonality` | `False` | Card prices don't vary hour-by-hour |
| `weekly_seasonality` | `True` | Weekend/Friday patterns are real |
| `yearly_seasonality` | `"auto"` | Enabled when 2+ years of data exist |
| `changepoint_prior_scale` | `0.05` | Conservative — avoids chasing short-term volatility spikes |

**Persistence:** `joblib.dump(model, path)` — serialises the entire fitted Prophet object (including Stan-generated posteriors) to a `.joblib` binary.

### 6.3 Planned Backends

| Backend | When to use | Notes |
|---|---|---|
| `XGBoostForecaster` | 90+ days history | Uses `build_feature_frame`; rolling-origin CV required |
| `LSTMForecaster` | 200+ days history | PyTorch sliding-window; GPU recommended for training |

---

## 7. Model Validation & Evaluation

### 7.1 Why Standard Train/Test Split is Wrong for Time Series

A random 80/20 train/test split would let the model train on *future* data and validate on *past* data — the model would look excellent on paper but fail completely in production.  Time-series models must respect temporal ordering.

### 7.2 Rolling-Origin Cross-Validation

The correct approach for time-series backtesting:

```
Fold 1:  train [day 1 … 60]        → validate [day 61 … 75]
Fold 2:  train [day 1 … 75]        → validate [day 76 … 90]
Fold 3:  train [day 1 … 90]        → validate [day 91 … 105]
...
```

Each fold trains on all history up to a cutoff and validates on the next N days.  The training window *always* ends before the validation window starts.

**Prophet's built-in CV:**

```python
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(
    model,
    initial="60 days",   # minimum training window
    period="30 days",    # how far to advance the cutoff each fold
    horizon="14 days",   # how far ahead to evaluate
)
df_perf = performance_metrics(df_cv)
# df_perf contains MAE, RMSE, MAPE, coverage per forecast horizon
```

### 7.3 Key Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **MAE** | mean(\|y − ŷ\|) | Average absolute dollar error |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more than MAE |
| **MAPE** | mean(\|y − ŷ\| / y) × 100 | Scale-free — % error; undefined if y = 0 |
| **Coverage** | % of actuals inside \[yhat_lower, yhat_upper\] | Should be ~80% for Prophet's 80% intervals |

### 7.4 Current Baseline Performance

> *To be populated once the ingest job has 60+ days of history.*  
> Run `prophet.diagnostics.cross_validation` and add the results here.

A well-calibrated Prophet model on liquid collectibles typically achieves:
- MAPE: 5–15% at a 7-day horizon
- Coverage: 75–85% (close to the nominal 80%)

---

## 8. REST API Contract

**Base URL (local):** `http://localhost:8000`  
**OpenAPI docs:** `http://localhost:8000/docs`

### `GET /health`

Liveness probe — used by Kubernetes `livenessProbe` and `readinessProbe`.

**Response 200:**
```json
{ "status": "ok", "version": "0.1.0" }
```

### `POST /predict`

Forecast prices for a card.

**Request body:**
```json
{
  "card_id":      "swsh1-1",
  "variant":      "holofoil",
  "horizon_days": 14
}
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `card_id` | string | required | pokemontcg.io card ID |
| `variant` | string | `"holofoil"` | any variant present in DB |
| `horizon_days` | int | 14 | 1 ≤ n ≤ 90 |

**Response 200:**
```json
{
  "card_id":      "swsh1-1",
  "variant":      "holofoil",
  "model":        "prophet",
  "horizon_days": 14,
  "forecast": [
    {
      "ds":          "2024-02-01",
      "yhat":        9.42,
      "yhat_lower":  7.80,
      "yhat_upper":  11.10
    },
    ...
  ]
}
```

**Response 400:**
```json
{
  "detail": "Not enough history for swsh1-1/holofoil (5 snapshots, need >=30). Run more ingests."
}
```

**Response 422:** FastAPI validation error (wrong field type, out-of-range `horizon_days`, etc.)

---

## 9. Configuration & Secrets

All configuration is in `config.py` via `pydantic-settings`.  Priority (highest first): environment variable → `.env` file → default.

| Variable | Default | Description |
|---|---|---|
| `POKEMONTCG_API_KEY` | `None` | Free key from dev.pokemontcg.io — increases rate limit from ~1k to ~20k req/day |
| `DATABASE_URL` | `sqlite:///data/prices.db` | SQLAlchemy URL.  Use `postgresql://user:pass@host/db` in production |
| `MODEL_DIR` | `./models` | Directory for `.joblib` model artefacts |
| `API_HOST` | `0.0.0.0` | Interface the API binds to |
| `API_PORT` | `8000` | TCP port |
| `LOG_LEVEL` | `INFO` | Python log level |

**Kubernetes secrets:**

```bash
kubectl create secret generic forecaster-secrets \
  --from-literal=database_url='postgresql://user:pass@host:5432/forecaster' \
  --from-literal=pokemontcg_api_key='YOUR_KEY'
```

Both the API `Deployment` and the ingest `CronJob` pull these values via `secretKeyRef`.

---

## 10. Docker & Docker Compose

### Dockerfile — Multi-Stage Build

```
Stage 1 (builder):  python:3.11-slim + uv
  - Install build-essential (needed by Prophet/cmdstan)
  - Copy pyproject.toml + uv.lock + src/
  - `uv sync --frozen --no-dev` → installs deps to /opt/venv

Stage 2 (runtime):  python:3.11-slim
  - Copy /opt/venv from builder (no build tools → smaller image)
  - Create non-root `app` user
  - EXPOSE 8000
  - HEALTHCHECK via httpx
  - CMD: python -m pokemon_forecaster.api.main
```

**Why multi-stage?**  Build tools (gcc, build-essential) are needed to compile Prophet's C extensions at install time but are unnecessary at runtime.  The two-stage build keeps the final image lean (typically 40–60% smaller).

**Why non-root user?**  Running as `root` in a container is a common security anti-pattern.  The `app` user has no elevated privileges.

### Docker Compose

Runs two services on the same bridge network:

| Service | Port | Command |
|---|---|---|
| `api` | 8000 | `python -m pokemon_forecaster.api.main` |
| `frontend` | 8501 | `streamlit run …` |

The `frontend` service sets `API_URL=http://api:8000` so requests resolve within the Docker network.  A `depends_on: condition: service_healthy` ensures the frontend doesn't start before the API passes its healthcheck.

Data volumes are bind-mounted: `./data:/app/data` and `./models:/app/models` so the SQLite DB and model artefacts persist on the host.

---

## 11. Kubernetes Deployment

### Files

| File | Kind | Purpose |
|---|---|---|
| `k8s/deployment.yaml` | Deployment | Runs 2 API replicas; liveness + readiness probes |
| `k8s/service.yaml` | Service | ClusterIP — exposes port 80 → 8000 inside the cluster |
| `k8s/cronjob-ingest.yaml` | CronJob | Nightly ingest at 06:00 UTC |

### Deploy to a cluster

```bash
# 1. Build and push the image
docker build -t ghcr.io/YOUR_GITHUB_USER/pokemon-price-forecaster:latest .
docker push ghcr.io/YOUR_GITHUB_USER/pokemon-price-forecaster:latest

# 2. Create the secret
kubectl create secret generic forecaster-secrets \
  --from-literal=database_url='postgresql://...' \
  --from-literal=pokemontcg_api_key='...'

# 3. Apply all manifests
kubectl apply -f k8s/
```

### CronJob — `cronjob-ingest.yaml`

```
Schedule:          0 6 * * *  (06:00 UTC daily)
concurrencyPolicy: Forbid     (skip if previous run still active)
backoffLimit:      2          (retry up to 2× on failure)
restartPolicy:     OnFailure
--batch-size:      50         (commit every 50 cards)
```

`concurrencyPolicy: Forbid` prevents overlapping runs.  Without it, a slow network day could cause two ingest jobs to run simultaneously, potentially inserting conflicting rows.

### Scaling Notes

- The `Deployment` runs 2 replicas for availability.  Both replicas read from the same DB and model store.  With SQLite (single-file), this requires a ReadWriteMany PVC or switching to Postgres.
- With Postgres, scale replicas freely — the DB handles concurrent reads, and the unique constraint handles concurrent writes.
- Models are loaded from a shared `PersistentVolumeClaim` (replace the `MODEL_DIR` default with a PVC mount path in production).

---

## 12. CI/CD Pipeline

**File:** `.github/workflows/ci-cd.yml`

### Triggers

| Trigger | Jobs run |
|---|---|
| PR against `main` | `test` (lint + pytest) |
| Push to `main` | `test` → `build-and-push` |
| Tag `v*.*.*` | `test` → `build-and-push` (+ optional `deploy`) |

### Jobs

**`test`**
1. Install `uv`, pin Python 3.11
2. `uv sync --extra dev` — install all dev dependencies
3. `ruff check .` — lint (style, imports, unused vars)
4. `ruff format --check .` — format compliance
5. `pytest -q` — run the test suite with coverage

**`build-and-push`** (on `main` / tags only)
1. Log in to GitHub Container Registry (GHCR) using `GITHUB_TOKEN` — no manual secret needed
2. Extract image tags: branch name, PR number, semver, short SHA
3. Multi-platform build with BuildKit layer caching (GHA cache)
4. Push to `ghcr.io/YOUR_GITHUB_USER/pokemon-price-forecaster`

**`deploy`** (commented out — wire up once you have a cluster)
1. Decode `KUBECONFIG` secret from base64
2. `kubectl apply -f k8s/`

### Image Tags

| Event | Tags |
|---|---|
| Push to `main` | `main`, `sha-abc1234` |
| Tag `v1.2.3` | `1.2.3`, `1.2`, `1`, `latest` |
| PR #42 | `pr-42` |

---

## 13. Local Development Guide

### Prerequisites

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
git clone https://github.com/YOUR_GITHUB_USER/pokemon-price-forecaster
cd pokemon-price-forecaster

uv python install 3.11
make dev          # installs dev + frontend extras
cp .env.example .env
# Optionally add your POKEMONTCG_API_KEY to .env
```

### Running things

```bash
make ingest       # pull swsh1 cards (limit 50 — quick smoke test)
make train        # train models for cards with 30+ days of history
make api          # start FastAPI at http://localhost:8000
make ui           # start Streamlit at http://localhost:8501

make up           # both API + frontend via Docker Compose
make test         # pytest
make lint         # ruff + mypy
```

### Manual ingest options

```bash
# Ingest a specific set
uv run pokemon-ingest --query 'set.id:swsh1'

# Ingest with a specific batch size
uv run pokemon-ingest --query 'set.id:swsh1' --batch-size 50

# Only process 100 cards (dev/smoke test)
uv run pokemon-ingest --query 'set.id:swsh1' --limit 100

# Train only cards with 60+ days of history
uv run pokemon-train --min-history 60 --variant holofoil
```

---

## 14. Design Decisions & Trade-offs

### SQLite → Postgres

**Decision:** Start with SQLite; switch to Postgres for production.  
**Rationale:** SQLite requires zero infrastructure for local development and testing.  The `PriceStore` DAO abstracts the DB entirely — changing `DATABASE_URL` is the only change needed.  SQLite is unsuitable for multi-replica Kubernetes deployments because multiple writers on a network-mounted file will corrupt data.

### Prophet as Baseline

**Decision:** Start with Prophet before adding XGBoost or LSTM.  
**Rationale:** Prophet requires no feature engineering, handles seasonality automatically, produces uncertainty intervals, and works with 30+ rows.  It gives you a working end-to-end pipeline quickly.  XGBoost will likely outperform Prophet once you have 90+ days of data, but it requires more infrastructure (rolling CV, feature pipeline, hyperparameter tuning).

### Lazy vs Batch Training

**Decision:** Lazy training as fallback, nightly `pokemon-train` job as primary.  
**Rationale:** The lazy path (`/predict` trains on first request) is a convenient fallback for cards that don't have a cached model.  The `pokemon-train` batch job pre-trains all qualifying cards overnight so the API always serves from fast cached artefacts during the day.  The two paths use the same `ProphetForecaster` class — no duplication.

### Forecaster ABC

**Decision:** Abstract base class for all forecasting backends.  
**Rationale:** The API endpoint code calls only `model.fit()` and `model.predict()` — it doesn't know which backend it's talking to.  Adding XGBoost or LSTM later requires zero changes to the API.  This is the Open/Closed Principle in practice.

### `uv` for Dependency Management

**Decision:** `uv` instead of pip/Poetry/conda.  
**Rationale:** `uv` is significantly faster than pip for large dependency trees (Prophet + Prophet's deps + scikit-learn + pandas is a heavy install).  `uv.lock` provides fully reproducible installs — the Docker image builds from the same lockfile as local development.

---

## 15. Interview Q&A Reference

### "Walk me through the system architecture."

> The system is a time-series forecasting pipeline.  It has three main components: (1) a daily ingest job that pulls price snapshots from the Pokémon TCG API and stores them in a relational DB; (2) a nightly batch training job that fits a Prophet model for each card and saves the artefact to disk; (3) a FastAPI service that loads these artefacts and serves forecasts.  A Streamlit frontend provides a no-code interface.  Everything is containerised with Docker and deployed to Kubernetes, with CI/CD via GitHub Actions.

### "How do you handle missing data in the time series?"

> The ingest job captures `market` as `NULL` when a card has no TCGPlayer listing on a given day (common for newly released cards with zero sales).  When building the training DataFrame, we filter out rows where `market IS NULL`.  A more sophisticated approach would interpolate short gaps (e.g. linear fill for 1–2 consecutive missing days) rather than dropping them — that's on the roadmap.

### "Why can't you just backfill historical data?"

> The pokemontcg.io API only exposes current prices — there's no historical endpoint.  To build a time series you have to run the ingest on a schedule and accumulate rows one day at a time.  The only way to backfill is to use a third-party data source like PriceCharting, which has historical exports going back several years.  For this project I chose the "forward-fill" approach: start the daily ingest, wait for history to accumulate, and train serious models once there are 60+ rows per card.

### "Why Prophet and not LSTM or XGBoost?"

> Prophet is the right baseline because it handles trend and seasonality automatically, produces calibrated uncertainty intervals out of the box, and only needs 30 rows to produce something useful.  XGBoost and LSTM are on the roadmap — XGBoost with engineered lag/rolling features will almost certainly outperform Prophet once there are 90+ days of data, but it requires more careful implementation (rolling-origin cross-validation, feature pipeline, hyperparameter tuning).  Prophet lets me validate the end-to-end pipeline (API → DB → model → forecast → UI) before adding model complexity.

### "How do you validate the model? What metrics do you use?"

> Time-series models can't use a random train/test split because that would allow the model to train on future data.  Instead I use rolling-origin cross-validation: fix a minimum training window, advance the cutoff date by a step size, evaluate on the next N days, repeat.  Prophet has this built-in via `prophet.diagnostics.cross_validation`.  The key metrics are MAE (average dollar error), MAPE (scale-free percentage error), and coverage (what fraction of actuals fall inside the 80% uncertainty interval — should be ~80% for a well-calibrated model).

### "How does the API handle on-demand training? Is that a problem in production?"

> The `/predict` endpoint falls back to training on demand if no cached `.joblib` exists.  Prophet training takes 2–10 seconds per card — acceptable for a low-traffic demo, a problem at scale.  In production, the `pokemon-train` batch job runs every night after the ingest and pre-trains all qualifying cards.  The API then only loads pre-trained artefacts and responds in milliseconds.  The lazy fallback is kept for cards that are too new to have been trained yet.

### "How do you prevent data leakage in the feature engineering?"

> Every feature that involves the target column (market price) uses `pd.Series.shift(1)` before computing lags or rolling statistics.  This means the feature value for day `t` is computed only from prices at `t-1` and earlier — the model never "sees" today's price when predicting today's price.  Forgetting this shift is one of the most common bugs in time-series ML.

### "How would you scale this to 20,000 cards in production?"

> Three things: (1) Replace SQLite with Postgres — concurrent writes from multiple replicas won't corrupt data, and Postgres handles millions of rows efficiently. (2) Mount a `PersistentVolumeClaim` for the model artefacts and share it across API replicas. (3) Run the ingest without `--limit` to pull all cards nightly — at 250 cards per API request and ~20k cards total, that's ~80 paginated HTTP calls, which completes in under a minute.

### "What would you add next?"

> Top priorities: (1) XGBoostForecaster with rolling-origin CV and the engineered feature set. (2) Postgres in the Kubernetes deployment (PostgreSQL helm chart or Cloud SQL). (3) Rate limiting + API key auth on the FastAPI layer (slowapi). (4) A card-name search in the Streamlit UI so users don't need to know the raw card ID.  Stretch goals: tournament meta-share features from limitless.gg, reprint flags, and an LSTM model.

---

*Last updated: May 2026 · Author: John C. Hood II*
