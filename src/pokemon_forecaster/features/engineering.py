"""Feature engineering for price forecasting.

The classic time-series feature set:
  - Lag features (price t-1, t-7, t-30)
  - Rolling stats (mean/std over 7d, 30d windows)
  - Calendar features (day-of-week, month)
  - Domain features (days since set release, days to next set, rarity one-hot)

The Prophet path skips most of this (it learns seasonality internally), but
XGBoost and LSTMs need it.
"""

from __future__ import annotations

import pandas as pd

LAG_DAYS = (1, 7, 14, 30)
ROLLING_WINDOWS = (7, 14, 30)


def add_lag_features(df: pd.DataFrame, target_col: str = "market") -> pd.DataFrame:
    """Add lag columns. Assumes df is sorted by date and single-card."""
    out = df.copy()
    for lag in LAG_DAYS:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str = "market") -> pd.DataFrame:
    out = df.copy()
    for w in ROLLING_WINDOWS:
        roll = out[target_col].shift(1).rolling(window=w, min_periods=max(2, w // 2))
        out[f"{target_col}_rollmean_{w}"] = roll.mean()
        out[f"{target_col}_rollstd_{w}"] = roll.std()
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "snapshot_date") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["dow"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def add_set_features(
    df: pd.DataFrame,
    release_date_col: str = "release_date",
    date_col: str = "snapshot_date",
) -> pd.DataFrame:
    """`days_since_release` is a strong predictor — new cards have hype premiums."""
    out = df.copy()
    snap = pd.to_datetime(out[date_col])
    rel = pd.to_datetime(out[release_date_col])
    out["days_since_release"] = (snap - rel).dt.days
    return out


def build_feature_frame(df: pd.DataFrame, target_col: str = "market") -> pd.DataFrame:
    """Run the full pipeline on a single-card-variant frame."""
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_calendar_features(df)
    if "release_date" in df.columns:
        df = add_set_features(df)
    return df.dropna().reset_index(drop=True)
