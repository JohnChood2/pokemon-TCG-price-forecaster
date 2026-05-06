"""Streamlit frontend for the forecaster.

Run: `uv run --extra frontend streamlit run src/pokemon_forecaster/frontend/streamlit_app.py`
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Pokémon Price Forecaster", page_icon="🃏", layout="wide")
st.title("🃏 Pokémon TCG Price Forecaster")

with st.sidebar:
    st.header("Forecast settings")
    card_id = st.text_input("Card ID", value="swsh1-1", help="e.g. swsh1-1, base1-4")
    variant = st.selectbox(
        "Variant",
        ["holofoil", "normal", "reverseHolofoil", "1stEditionHolofoil"],
        index=0,
    )
    horizon = st.slider("Horizon (days)", min_value=1, max_value=90, value=14)
    submit = st.button("Forecast", type="primary")

if submit:
    with st.spinner("Calling forecaster..."):
        try:
            r = httpx.post(
                f"{API_URL}/predict",
                json={"card_id": card_id, "variant": variant, "horizon_days": horizon},
                timeout=60.0,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    df = pd.DataFrame(data["forecast"])
    df["ds"] = pd.to_datetime(df["ds"])

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["ds"], y=df["yhat"], mode="lines+markers", name="Forecast"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["ds"], df["ds"][::-1]]),
                y=pd.concat([df["yhat_upper"], df["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,100,200,0.2)",
                line={"color": "rgba(255,255,255,0)"},
                name="80% interval",
            )
        )
        fig.update_layout(
            title=f"{card_id} ({variant}) — next {horizon} days",
            xaxis_title="Date",
            yaxis_title="Market price (USD)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Model", data["model"])
        st.metric("Horizon (days)", data["horizon_days"])
        st.dataframe(df, use_container_width=True)
else:
    st.info("Enter a card ID and click **Forecast**.")
