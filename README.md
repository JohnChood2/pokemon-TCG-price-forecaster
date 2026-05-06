# Pokémon TCG Card Price Forecaster

Forecasts Pokémon TCG card prices 1–4 weeks out using historical TCGPlayer data.
Served as a FastAPI app with a Streamlit frontend, containerized with Docker,
deployed to Kubernetes via GitHub Actions.

## Stack

- **Python 3.11**, managed end-to-end with [`uv`](https://docs.astral.sh/uv/)
- **Data**: [pokemontcg.io](https://pokemontcg.io) → SQLite (or Postgres in prod)
- **Modeling**: Prophet (baseline) → XGBoost (with engineered features) → LSTM (stretch)
- **Serving**: FastAPI + Uvicorn
- **Frontend**: Streamlit + Plotly
- **MLOps**: Docker → GitHub Actions → GHCR → Kubernetes (minikube/GKE)

---

## Quickstart

### Prerequisites
Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
git clone <your-repo>
cd pokemon-price-forecaster

# Pin Python and install everything (dev + frontend extras)
uv python install 3.11
make dev

cp .env.example .env
# (optional) add a free POKEMONTCG_API_KEY for higher rate limits
```

### Run things

```bash
# Pull a small slice of cards + current prices into the DB
make ingest

# Train models for cards with enough history
make train

# Start the API on http://localhost:8000
make api

# In another terminal, start the Streamlit UI on http://localhost:8501
make ui

# Or run both with Docker Compose
make up
```

API docs are auto-generated at <http://localhost:8000/docs>.

---

## Project layout

```
src/pokemon_forecaster/
  config.py            # Pydantic settings (env vars)
  data/
    client.py          # pokemontcg.io HTTP client (httpx + tenacity retries)
    storage.py         # SQLAlchemy models + DAO
  features/
    engineering.py     # lag, rolling, calendar, set features
  models/
    forecaster.py      # Forecaster ABC + ProphetForecaster
  api/
    main.py            # FastAPI app
    schemas.py         # Pydantic request/response
  frontend/
    streamlit_app.py   # UI
  cli/
    ingest.py          # CronJob entry — run daily
    train.py           # Batch trainer — run after ingest

k8s/                   # Kubernetes manifests
.github/workflows/     # CI/CD
```

---

## The phased build plan

The scaffold here is **phase 1**. To finish the project:

1. ✅ **Scaffold** — uv project, module layout, Dockerfile, CI skeleton.
2. 🟡 **Data pipeline** — client + storage are written. Now run ingest **daily for ~30+ days** (or backfill historical data) so you have a real time series, not one snapshot. The Pokémon TCG API gives you *current* prices only — building history is a function of running ingest on a schedule.
3. ⬜ **Feature engineering** — basic features stubbed. Add: tournament meta-share (limitless.gg has APIs), set rotation distance, reprint flags.
4. ⬜ **Modeling** — Prophet works out of the box. Implement `XGBoostForecaster` using `features.build_feature_frame`. Backtest with rolling-origin CV — *not* a single train/test split, since this is time series.
5. ⬜ **API hardening** — add rate limiting (slowapi), API keys, request logging.
6. ⬜ **Frontend polish** — card search by name (call the API client), historical-vs-forecast plot.
7. ⬜ **Docker** — works as-is. Test locally with `make up`.
8. ⬜ **CI/CD** — `.github/workflows/ci-cd.yml` lints, tests, builds, pushes to GHCR. To push to your repo, just commit and tag.
9. ⬜ **Kubernetes** —
   - Local: `minikube start && kubectl apply -f k8s/`
   - Cloud: GKE Autopilot has a generous free tier. Replace `YOUR_GITHUB_USER` in the manifests, create the `forecaster-secrets` Secret, then `kubectl apply -f k8s/`.

---

## Bootstrapping data history

The TCGPlayer prices in pokemontcg.io are **current snapshots only**. Two ways
to get a real time series:

1. **Forward-fill** — run the daily ingest CronJob for a month or two before
   training serious models. Cheap, slow, accurate.
2. **Backfill from a third source** — projects like
   [pkmn.cards](https://pkmn.cards) and the
   [PriceCharting](https://www.pricecharting.com/) export include historical
   prices. You'd add a one-off backfill script that pulls historical CSVs and
   inserts them into `price_snapshots`.

For a portfolio project, option 1 is the honest answer and gives you something
real to talk about ("the system has been running for X weeks, here's the
prediction error over time").

---

## Testing

```bash
make test         # pytest
make lint         # ruff + mypy
```

CI runs both on every PR.

## License

MIT
