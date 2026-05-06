"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd

from pokemon_forecaster.features import build_feature_frame


def test_build_feature_frame_drops_nans_and_keeps_columns():
    df = pd.DataFrame(
        {
            "snapshot_date": pd.date_range("2024-01-01", periods=90, freq="D"),
            "market": [10 + i * 0.05 for i in range(90)],
            "release_date": ["2023-11-01"] * 90,
        }
    )
    out = build_feature_frame(df)
    assert "market_lag_1" in out.columns
    assert "market_rollmean_7" in out.columns
    assert "dow" in out.columns
    assert "days_since_release" in out.columns
    assert out["market_lag_1"].isna().sum() == 0
    # Largest lag is 30 — we should keep at least 90 - 30 = 60 rows.
    assert len(out) >= 60
