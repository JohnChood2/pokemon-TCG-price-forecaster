"""Feature engineering for price forecasting models.

Why features?
-------------
Prophet learns seasonality internally from raw (ds, y) data, so it doesn't
need a feature matrix.  But XGBoost and LSTM models need explicit features
to capture patterns that the raw price series doesn't encode directly.

This module builds the classic time-series feature set used in financial
and e-commerce price prediction:

1. **Lag features**  — price at t-1, t-7, t-14, t-30.
   Captures autocorrelation: today's price is strongly correlated with
   yesterday's price (t-1), the same day last week (t-7), etc.

2. **Rolling statistics** — mean and standard deviation over 7, 14, 30-day
   windows.  Mean captures local trend; std captures recent volatility.
   Both are shifted by 1 day so they never include ``t`` itself (no leakage).

3. **Calendar features** — day of week, month, weekend flag.
   Pokémon card prices have a real weekly cycle: new set reveals often happen
   on Fridays, tournament events land on weekends, etc.

4. **Domain feature: days_since_release** — the number of days between a
   card's set release date and the snapshot date.
   New cards carry a "hype premium" that decays over the first few weeks as
   supply enters the market.  This single feature is often the strongest
   predictor in a gradient-boosted model.

Usage
-----
Typical call in the XGBoostForecaster training loop::

    df = store.get_history_as_frame(card_id="swsh1-1", variant="holofoil")
    # df has columns: snapshot_date, market, release_date
    feature_df = build_feature_frame(df, target_col="market")
    X = feature_df.drop(columns=["market", "snapshot_date"])
    y = feature_df["market"]

Note on data leakage
--------------------
All rolling and lag operations shift the series *forward* by 1 day
(``shift(1)``) before computing.  This guarantees that the feature for day ``t``
only uses information available *before* day ``t``.  Forgetting to do this is
one of the most common bugs in time-series ML — the model would look perfect
on training data and fail completely on live predictions.
"""

from __future__ import annotations

import pandas as pd

# Days to look back for lag features.  1 day = yesterday, 7 = last week, etc.
LAG_DAYS = (1, 7, 14, 30)

# Window sizes for rolling mean/std.  Shift(1) ensures no data leakage.
ROLLING_WINDOWS = (7, 14, 30)


def add_lag_features(df: pd.DataFrame, target_col: str = "market") -> pd.DataFrame:
    """Add lag columns for each value in ``LAG_DAYS``.

    Parameters
    ----------
    df:
        Single-card, single-variant DataFrame sorted by date ascending.
        Must contain a column named ``target_col``.
    target_col:
        The price column to lag (default: ``"market"``).

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus new columns ``{target_col}_lag_{n}``
        for each lag n.  The first ``max(LAG_DAYS)`` rows will have NaN
        in the lag columns — drop them with ``build_feature_frame`` or
        ``df.dropna()``.

    Notes
    -----
    ``pd.Series.shift(n)`` moves all values down by ``n`` rows, leaving
    the first ``n`` rows as NaN.  This is the standard pandas way to
    create lagged features without risk of look-ahead bias.
    """
    out = df.copy()
    for lag in LAG_DAYS:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str = "market") -> pd.DataFrame:
    """Add rolling mean and standard deviation features.

    Both statistics are computed on ``shift(1)`` of the target, meaning the
    rolling window for day ``t`` ends at day ``t-1``.  This is critical for
    preventing data leakage — the model must not "see" day ``t``'s price when
    predicting it.

    Parameters
    ----------
    df:
        Single-card, single-variant DataFrame sorted by date ascending.
    target_col:
        The price column to roll over.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus columns:
        - ``{target_col}_rollmean_{w}`` — rolling mean over w days
        - ``{target_col}_rollstd_{w}``  — rolling std over w days

    Notes
    -----
    ``min_periods=max(2, w // 2)`` allows the rolling window to start
    computing once at least half the window is filled.  This reduces the
    number of NaN rows dropped during ``build_feature_frame`` — important
    when you have only 30-60 rows of history.
    """
    out = df.copy()
    for w in ROLLING_WINDOWS:
        # shift(1) = don't include today's price in the window
        roll = out[target_col].shift(1).rolling(window=w, min_periods=max(2, w // 2))
        out[f"{target_col}_rollmean_{w}"] = roll.mean()
        out[f"{target_col}_rollstd_{w}"] = roll.std()
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "snapshot_date") -> pd.DataFrame:
    """Encode calendar information as numeric columns.

    Parameters
    ----------
    df:
        DataFrame with a date or datetime column named ``date_col``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus:
        - ``dow``        — day of week (0 = Monday, 6 = Sunday)
        - ``month``      — month (1-12)
        - ``is_weekend`` — 1 if Saturday or Sunday, else 0

    Notes
    -----
    These features capture weekly and monthly seasonality.  They are redundant
    for Prophet (which models seasonality directly) but are valuable for
    XGBoost, which treats every row independently and needs explicit encodings.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["dow"] = dt.dt.dayofweek  # 0 = Mon, 6 = Sun
    out["month"] = dt.dt.month
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def add_set_features(
    df: pd.DataFrame,
    release_date_col: str = "release_date",
    date_col: str = "snapshot_date",
) -> pd.DataFrame:
    """Add domain-specific features based on the card's set release date.

    ``days_since_release`` is consistently one of the top feature importances
    in trained XGBoost models.  Newly released cards trade at a significant
    premium for the first 2-4 weeks as pack-opening hype drives demand.
    After that, supply normalises and prices stabilise or decline.

    Parameters
    ----------
    df:
        DataFrame with ``release_date_col`` (the set's release date) and
        ``date_col`` (the observation date).  Both can be strings or datetimes
        — ``pd.to_datetime`` handles either.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus ``days_since_release`` (integer).

    Future extensions
    -----------------
    - ``days_to_rotation`` — days until the set rotates out of the Standard
      format.  Rotation typically causes a sharp price drop.
    - ``is_reprint`` — whether this card has been reprinted (binary flag).
      Reprints usually cause a permanent price correction downward.
    """
    out = df.copy()
    snap = pd.to_datetime(out[date_col])
    rel = pd.to_datetime(out[release_date_col])
    out["days_since_release"] = (snap - rel).dt.days
    return out


def build_feature_frame(df: pd.DataFrame, target_col: str = "market") -> pd.DataFrame:
    """Run the full feature-engineering pipeline on a single-card DataFrame.

    This is the entry point for building the feature matrix before fitting
    XGBoost or LSTM models.  It runs all four feature groups in order,
    then drops any rows that still have NaN values (introduced by the lag
    and rolling operations on the early rows of the series).

    Parameters
    ----------
    df:
        Raw price history for a single (card_id, variant).  Must contain:
        - ``snapshot_date`` — date of each observation.
        - ``market`` (or whatever ``target_col`` is) — the price to predict.
        - ``release_date`` (optional) — if present, ``days_since_release`` is added.

    Returns
    -------
    pd.DataFrame
        Feature matrix with all engineered columns, sorted by date, NaN-free.
        The target column (``market``) is included — the caller should split
        it out into ``X`` and ``y`` before fitting.

    Notes on row count after ``dropna``
    -------------------------------------
    The largest lag is 30 days, so you lose ~30 rows from the start of the
    series.  With 60 input rows you'll have ~30 usable rows — enough to train
    a basic model, but more data is always better.

    Example
    -------
    ::

        df = store.get_history_as_frame("swsh1-1", "holofoil")
        feature_df = build_feature_frame(df)
        X = feature_df.drop(columns=["market", "snapshot_date", "release_date"])
        y = feature_df["market"]
        model.fit(X, y)
    """
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = add_calendar_features(df)
    if "release_date" in df.columns:
        df = add_set_features(df)
    return df.dropna().reset_index(drop=True)
