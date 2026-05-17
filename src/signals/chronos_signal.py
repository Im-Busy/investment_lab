"""Chronos foundation model as signal source.

Zero-shot forecasting -> directional signal -> trade entry.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChronosSignalGenerator:
    """Generate trading signals from Chronos zero-shot forecasts.

    Forecasts next N days of returns -> compute expected direction -> signal.
    """

    def __init__(
        self,
        model_size: str = "chronos-2",
        device: str = "cpu",
        forecast_horizon: int = 5,
        context_length: int = 252,
        signal_threshold: float = 0.005,
    ):
        self.model_size = model_size
        self.device = device
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.signal_threshold = signal_threshold
        self._forecaster = None

    def _get_forecaster(self):
        if self._forecaster is None:
            from src.ml.models.chronos import ChronosForecaster

            self._forecaster = ChronosForecaster(
                model_size=self.model_size,
                device=self.device,
                prediction_length=self.forecast_horizon,
                context_length=self.context_length,
            )
        return self._forecaster

    def generate_signal(self, prices: pd.Series) -> float:
        """Generate a trading signal from Chronos forecast.

        Args:
            prices: Historical OHLC close prices (needs at least context_length bars)

        Returns:
            Signal in [-1, 1]: positive = bullish, negative = bearish, 0 = neutral
        """
        if len(prices) < self.context_length:
            return 0.0

        recent = prices.iloc[-self.context_length :]

        try:
            forecaster = self._get_forecaster()
            result = forecaster.predict(recent, num_samples=50)
        except Exception as e:
            logger.warning(f"Chronos prediction failed: {e}")
            return 0.0

        forecast_mean = result["mean"]
        current_price = float(recent.iloc[-1])
        expected_return = (float(forecast_mean.iloc[-1]) - current_price) / current_price

        signal = float(np.clip(expected_return / self.signal_threshold, -1, 1))
        return signal


class ChronosMultiHorizonSignal:
    """Multi-horizon Chronos signal with ensemble across horizons."""

    HORIZONS = [1, 5, 21]  # 1-day, 1-week, 1-month

    def __init__(
        self,
        model_size: str = "chronos-2",
        device: str = "cpu",
        weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
    ):
        self.generators = {
            h: ChronosSignalGenerator(
                model_size=model_size,
                device=device,
                forecast_horizon=h,
            )
            for h in self.HORIZONS
        }
        self.weights = weights

    def generate_signal(self, prices: pd.Series) -> float:
        """Weighted ensemble across forecast horizons."""
        signals = []
        for h, gen in self.generators.items():
            sig = gen.generate_signal(prices)
            signals.append(sig)
        weighted = sum(w * s for w, s in zip(self.weights, signals))
        return float(np.clip(weighted, -1, 1))
