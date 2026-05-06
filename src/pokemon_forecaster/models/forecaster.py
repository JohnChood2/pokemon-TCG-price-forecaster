"""Forecasting models.

We expose a single `Forecaster` protocol so the API doesn't care which
backend is in use. Start with Prophet (zero feature engineering required),
then add XGBoost/LSTM behind the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class Forecaster(ABC):
    """Common interface for all forecasting backends."""

    name: str = "base"

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> None:
        """Fit to a long-format frame with columns ['ds', 'y']."""

    @abstractmethod
    def predict(self, horizon_days: int) -> pd.DataFrame:
        """Return a DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']."""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved %s -> %s", self.name, path)

    @classmethod
    def load(cls, path: Path) -> Forecaster:
        return joblib.load(path)  # type: ignore[no-any-return]


class ProphetForecaster(Forecaster):
    """Facebook Prophet — handles trend + weekly/yearly seasonality automatically.

    Best when you have >60 days of history. Below that, fall back to a naive
    or exponential-smoothing model (TODO).
    """

    name = "prophet"

    def __init__(self, **prophet_kwargs: object) -> None:
        # Lazy import — prophet pulls in cmdstan, slow at module load.
        from prophet import Prophet

        self._model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality="auto",
            changepoint_prior_scale=0.05,
            **prophet_kwargs,  # type: ignore[arg-type]
        )
        self._fitted = False

    def fit(self, history: pd.DataFrame) -> None:
        if not {"ds", "y"}.issubset(history.columns):
            raise ValueError("history must have columns 'ds' and 'y'")
        self._model.fit(history[["ds", "y"]])
        self._fitted = True

    def predict(self, horizon_days: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        future = self._model.make_future_dataframe(periods=horizon_days)
        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)


# TODO: class XGBoostForecaster(Forecaster) — uses features.build_feature_frame
# TODO: class LSTMForecaster(Forecaster) — pytorch, sliding window
