"""Price forecasting models.

Architecture
------------
All forecasting backends implement the ``Forecaster`` abstract base class,
which exposes exactly two methods: ``fit(history)`` and ``predict(horizon_days)``.
The FastAPI endpoint calls only these two methods, so adding a new backend
(XGBoost, LSTM, …) requires zero changes to the API code.

The pluggable backend pattern also makes A/B testing trivial: load two model
files, call ``predict`` on both, and return whichever has lower recent MAE.

Model persistence
-----------------
``Forecaster.save()`` and ``Forecaster.load()`` use ``joblib``, which
serialises Python objects to a compressed binary file.  This means the entire
fitted model (Prophet's Stan-generated posterior, sklearn pipelines, etc.)
is captured in a single ``.joblib`` artefact.

File naming convention:  ``{card_id}__{variant}.joblib``
  e.g.  ``models/swsh1-1__holofoil.joblib``

Current backend
---------------
``ProphetForecaster`` wraps Facebook Prophet.  Prophet is a good baseline
because it:

- Handles trend changes, weekly seasonality, and yearly seasonality out of the
  box with no feature engineering required.
- Produces uncertainty intervals (``yhat_lower`` / ``yhat_upper``) natively.
- Works with as few as ~30 observations.
- Is interpretable — you can plot the components (trend, seasonality) to
  explain the forecast to a non-technical audience.

Hyperparameters
---------------
``changepoint_prior_scale=0.05`` (default 0.05) controls how flexible the
trend line is.  Lower = smoother trend, less risk of over-fitting short-term
spikes.  For volatile collectibles you might want to experiment with 0.1–0.3
if the model consistently under-reacts to price shocks.

Planned backends
----------------
- ``XGBoostForecaster``  — uses the engineered features from
  ``features.engineering.build_feature_frame``.  Expected to beat Prophet
  once you have 90+ days of history and can incorporate lag/rolling features.
- ``LSTMForecaster``     — PyTorch sliding-window model.  Highest ceiling but
  needs the most data (200+ days) and is the hardest to explain in an
  interview.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class Forecaster(ABC):
    """Abstract interface that every forecasting backend must implement.

    The API layer depends *only* on this interface, not on any concrete
    implementation, which keeps the codebase open for extension and
    closed for modification (Open/Closed Principle).
    """

    name: str = "base"

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> None:
        """Train the model on historical price data.

        Parameters
        ----------
        history:
            Long-format DataFrame with exactly two columns:

            - ``ds`` (datetime-like)  — the date of each observation.
            - ``y``  (float)          — the market price on that date.

            Rows must be sorted by ``ds`` ascending and contain no NaNs in ``y``.
            30 rows is the bare minimum; 60+ is recommended.
        """

    @abstractmethod
    def predict(self, horizon_days: int) -> pd.DataFrame:
        """Generate a price forecast for the next *horizon_days* days.

        Parameters
        ----------
        horizon_days:
            How many future days to forecast (1–90 per the API contract).

        Returns
        -------
        pd.DataFrame
            One row per forecast day with columns:

            - ``ds``          — the future date.
            - ``yhat``        — the point estimate (expected market price).
            - ``yhat_lower``  — lower bound of the 80% uncertainty interval.
            - ``yhat_upper``  — upper bound of the 80% uncertainty interval.
        """

    def save(self, path: Path) -> None:
        """Serialise the fitted model to a ``.joblib`` file.

        Creates parent directories automatically so callers don't have to.
        The file can be loaded on any machine with the same Python + package
        versions — be mindful of version pinning when upgrading Prophet.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved %s -> %s", self.name, path)

    @classmethod
    def load(cls, path: Path) -> Forecaster:
        """Deserialise a previously saved model from disk.

        Note: ``joblib.load`` returns whatever object was pickled, so the
        return type annotation is ``Forecaster`` rather than ``cls``.  This
        is intentional — you might save a ``ProphetForecaster`` and load it
        through the base class interface.
        """
        return joblib.load(path)  # type: ignore[no-any-return]


class ProphetForecaster(Forecaster):
    """Price forecaster backed by Facebook Prophet.

    Prophet is a decomposable time-series model of the form::

        y(t) = trend(t) + seasonality(t) + holidays(t) + ε

    It fits the trend using piece-wise linear or logistic growth, and
    captures seasonality with Fourier series.

    Configuration choices
    ----------------------
    - ``daily_seasonality=False``   Pokémon prices don't meaningfully vary
                                    hour-by-hour within a day, and daily
                                    seasonality would overfit with sparse data.
    - ``weekly_seasonality=True``   Weekend trading activity is real — prices
                                    sometimes move on Friday/Saturday.
    - ``yearly_seasonality="auto"`` Enabled when there's enough data (typically
                                    2+ years); disabled otherwise.
    - ``changepoint_prior_scale=0.05``  Conservative trend flexibility.  Avoids
                                    chasing short-term volatility spikes.

    Lazy import
    -----------
    Prophet's ``cmdstan`` dependency is large and slow to import.  The import
    is deferred to ``__init__`` so that modules importing this file don't pay
    the startup cost unless they actually instantiate the class.
    """

    name = "prophet"

    def __init__(self, **prophet_kwargs: object) -> None:
        """Instantiate a Prophet model with sensible defaults.

        Parameters
        ----------
        **prophet_kwargs:
            Any keyword argument accepted by ``prophet.Prophet`` — forwarded
            directly.  Useful for hyperparameter tuning:

                ProphetForecaster(changepoint_prior_scale=0.3)
        """
        # Deferred import keeps module load fast for callers that don't use Prophet.
        from prophet import Prophet  # type: ignore[import-untyped]

        self._model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality="auto",
            changepoint_prior_scale=0.05,
            **prophet_kwargs,  # type: ignore[arg-type]
        )
        self._fitted = False

    def fit(self, history: pd.DataFrame) -> None:
        """Fit Prophet to a (ds, y) history frame.

        Parameters
        ----------
        history:
            DataFrame with columns ``ds`` and ``y``.  Prophet is strict about
            these names — any other columns are ignored.

        Raises
        ------
        ValueError
            If ``ds`` or ``y`` columns are missing.
        """
        if not {"ds", "y"}.issubset(history.columns):
            raise ValueError("history must have columns 'ds' and 'y'")
        self._model.fit(history[["ds", "y"]])
        self._fitted = True

    def predict(self, horizon_days: int) -> pd.DataFrame:
        """Generate a forward forecast.

        Prophet's ``make_future_dataframe`` appends ``horizon_days`` rows to
        the training date range.  We call ``predict()`` on the full frame
        (history + future) and then slice off only the future rows using
        ``.tail(horizon_days)``.

        Parameters
        ----------
        horizon_days:
            Number of future days to forecast.

        Returns
        -------
        pd.DataFrame
            Columns: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
            Rows: exactly ``horizon_days`` future dates.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        # make_future_dataframe creates a date range that includes the
        # training period plus `horizon_days` future rows.
        future = self._model.make_future_dataframe(periods=horizon_days)
        forecast = self._model.predict(future)

        # Return only the future rows (not the in-sample fitted values).
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)


# ---------------------------------------------------------------------------
# TODO: future backends
# ---------------------------------------------------------------------------

# class XGBoostForecaster(Forecaster):
#     """Gradient-boosted tree model using engineered lag/rolling/calendar features.
#
#     Better than Prophet once you have 90+ days of history.  Uses
#     ``features.engineering.build_feature_frame`` to produce the feature matrix.
#     Train with rolling-origin cross-validation (not a single train/test split)
#     since this is a time series — future data must never leak into training.
#     """
#
# class LSTMForecaster(Forecaster):
#     """PyTorch LSTM with a sliding-window input of length 30 days.
#
#     Highest modelling ceiling but requires 200+ days of data, GPU for training,
#     and is the hardest to explain.  Good stretch goal once the Prophet baseline
#     is validated.
#     """
