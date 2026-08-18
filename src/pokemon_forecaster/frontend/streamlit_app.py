"""Streamlit web frontend for the Pokémon TCG Price Forecaster.

This is a thin UI wrapper around the FastAPI ``/predict`` endpoint.  It lets
non-technical users explore price forecasts by entering a card ID, choosing a
variant, and selecting a forecast horizon — no API client or curl knowledge needed.

How it works
------------
1. The user fills in the sidebar form and clicks "Forecast".
2. Streamlit calls ``httpx.post`` against the FastAPI service (``API_URL``).
3. The API returns a JSON forecast; we parse it into a pandas DataFrame.
4. Plotly renders an interactive line chart with the 80% confidence band.

Architecture note
-----------------
The Streamlit app is a *separate service* from the FastAPI backend.  In Docker
Compose (``docker-compose.yml``) they run in two containers on the same network:
``api`` (port 8000) and ``frontend`` (port 8501).  The ``API_URL`` environment
variable wires them together.

In Kubernetes you'd deploy the frontend as its own Deployment + Service, with
``API_URL`` pointing to the ClusterIP service of the API deployment.

Run locally
-----------
    uv run --extra frontend streamlit run src/pokemon_forecaster/frontend/streamlit_app.py

Or via Docker Compose:
    make up
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# The API URL is injected via environment variable in Docker/Kubernetes.
# Falls back to localhost for local development.
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pokémon Price Forecaster",
    page_icon="🃏",
    layout="wide",
)
st.title("🃏 Pokémon TCG Price Forecaster")

# ── Sidebar — forecast parameters ────────────────────────────────────────────
with st.sidebar:
    st.header("Forecast settings")

    card_id = st.text_input(
        "Card ID",
        value="swsh1-1",
        help=(
            "pokemontcg.io card identifier.  Format: {set_id}-{number}, "
            "e.g. swsh1-1 (Sword & Shield base set, card #1) or base1-4 (Charizard)."
        ),
    )

    variant = st.selectbox(
        "Variant",
        ["holofoil", "normal", "reverseHolofoil", "1stEditionHolofoil"],
        index=0,
        help="TCGPlayer price tier.  'holofoil' is the most-traded variant for rare cards.",
    )

    horizon = st.slider(
        "Forecast horizon (days)",
        min_value=1,
        max_value=90,
        value=14,
        help="How many days into the future to forecast.  Wider horizons = wider uncertainty.",
    )

    submit = st.button("Forecast", type="primary")

# ── Main area — chart + data table ───────────────────────────────────────────
if submit:
    with st.spinner("Calling forecaster…"):
        try:
            r = httpx.post(
                f"{API_URL}/predict",
                json={
                    "card_id": card_id,
                    "variant": variant,
                    "horizon_days": horizon,
                },
                timeout=60.0,  # Prophet training can take a few seconds
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            # Surface the API's error detail (e.g. "not enough history") to the user.
            detail = e.response.json().get("detail", str(e))
            st.error(f"API returned {e.response.status_code}: {detail}")
            st.stop()
        except httpx.HTTPError as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
            st.stop()

    # Parse forecast JSON into a DataFrame.
    df = pd.DataFrame(data["forecast"])
    df["ds"] = pd.to_datetime(df["ds"])

    col1, col2 = st.columns([2, 1])

    with col1:
        # ── Plotly chart with confidence band ──
        fig = go.Figure()

        # Point forecast line
        fig.add_trace(
            go.Scatter(
                x=df["ds"],
                y=df["yhat"],
                mode="lines+markers",
                name="Forecast (yhat)",
                line={"color": "royalblue", "width": 2},
            )
        )

        # 80% confidence band — drawn as a filled area using the "toself" fill
        # trick: upper bound forward, lower bound reversed = a closed polygon.
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["ds"], df["ds"][::-1]]),
                y=pd.concat([df["yhat_upper"], df["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(65, 105, 225, 0.15)",
                line={"color": "rgba(255,255,255,0)"},
                name="80% interval",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            title=f"{card_id} ({variant}) — {horizon}-day price forecast",
            xaxis_title="Date",
            yaxis_title="Market price (USD)",
            legend={"orientation": "h", "y": -0.2},
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ── Summary metrics + raw data table ──
        st.metric("Model", data["model"])
        st.metric("Horizon (days)", data["horizon_days"])
        st.metric(
            "Forecast range (USD)",
            f"${df['yhat'].min():.2f} - ${df['yhat'].max():.2f}",
        )
        st.dataframe(
            df[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
                columns={
                    "ds": "Date",
                    "yhat": "Price",
                    "yhat_lower": "Lower 80%",
                    "yhat_upper": "Upper 80%",
                }
            ),
            use_container_width=True,
        )

else:
    # Placeholder instructions shown before the user submits a forecast.
    st.info(
        "Enter a **Card ID** and click **Forecast** to see a price prediction.\n\n"
        "**Need card IDs?**  Find them at [pokemontcg.io](https://pokemontcg.io) "
        "or try ``swsh1-1``, ``swsh1-2``, ``base1-4``."
    )
