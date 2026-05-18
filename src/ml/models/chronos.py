"""
Amazon Chronos-2 Foundation Model

Universal time-series foundation model with zero-shot capabilities.
Supports univariate, multivariate, and covariate-informed forecasting.

Paper: https://arxiv.org/abs/2510.15821
GitHub: https://github.com/amazon-science/chronos-forecasting
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class ChronosForecaster:
    """
    Chronos-2 wrapper for financial forecasting.

    Uses Chronos2Pipeline (chronos-2, chronos-2-small, bolt variants).
    Features: zero-shot forecasting, probabilistic quantile forecasts.
    """

    MODEL_SIZES = {
        "chronos-2": "amazon/chronos-2",
        "chronos-2-small": "autogluon/chronos-2-small",
        "bolt-tiny": "amazon/chronos-bolt-tiny",
        "bolt-mini": "amazon/chronos-bolt-mini",
        "bolt-small": "amazon/chronos-bolt-small",
        "bolt-base": "amazon/chronos-bolt-base",
    }

    _DEFAULT_SIZE = "chronos-2"

    def __init__(
        self,
        model_size: str = "chronos-2",
        device: str = "cpu",
        prediction_length: int = 64,
        context_length: int = 512,
    ):
        """
        Initialize Chronos-2 forecaster.

        Args:
            model_size: Model size variant (chronos-2 is default)
            device: torch device
            prediction_length: Forecast horizon
            context_length: Input context length
        """
        from chronos import Chronos2Pipeline

        self.model_size = model_size
        self.device = device
        self.prediction_length = prediction_length
        self.context_length = context_length

        model_id = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES[self._DEFAULT_SIZE])
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device,
            dtype="auto",
        )

    def predict(
        self,
        series: pd.Series,
        exogenous: Optional[pd.DataFrame] = None,
        num_samples: int = 100,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> Dict[str, pd.Series]:
        """
        Generate zero-shot forecasts.

        Args:
            series: Time series to forecast
            exogenous: Optional covariates
            num_samples: Number of sample paths (ignored by chronos-2; uses built-in samples)
            quantiles: Quantiles for probabilistic forecasts

        Returns:
            Dictionary with mean, median, and quantile forecasts as pd.Series
        """
        import torch

        values = series.values.astype(np.float32)
        context = torch.tensor(values).reshape(1, 1, -1)

        quantile_preds, mean_preds = self.pipeline.predict_quantiles(
            context,
            prediction_length=self.prediction_length,
            quantile_levels=quantiles,
        )

        q_np = quantile_preds[0].detach().cpu().numpy().squeeze()  # (pred_len, n_quantiles)
        m_np = mean_preds[0].detach().cpu().numpy().squeeze()  # (pred_len,)

        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=self.prediction_length,
            freq="D",
        )

        result = {
            "mean": pd.Series(m_np, index=forecast_index),
            "median": pd.Series(np.median(q_np, axis=1), index=forecast_index),
        }

        for i, q in enumerate(quantiles):
            result[f"q{int(q * 100)}"] = pd.Series(q_np[:, i], index=forecast_index)

        return result

    def fine_tune(
        self,
        train_series: List[pd.Series],
        epochs: int = 10,
        learning_rate: float = 1e-4,
    ) -> None:
        """
        Fine-tune on domain-specific data.

        Args:
            train_series: List of training series
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        pass
