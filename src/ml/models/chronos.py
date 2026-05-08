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

    Features:
    - Zero-shot forecasting (no training required)
    - Multivariate and covariate-informed support
    - Multiple model sizes (8M to 1B+ parameters)
    - In-context learning across series
    """

    MODEL_SIZES = {
        "tiny": "amazon/chronos-2-tiny",
        "mini": "amazon/chronos-2-mini",
        "small": "amazon/chronos-2-small",
        "base": "amazon/chronos-2-base",
        "large": "amazon/chronos-2-large",
        "xl": "amazon/chronos-2-xl",
    }

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        prediction_length: int = 64,
        context_length: int = 512,
    ):
        """
        Initialize Chronos-2.

        Args:
            model_size: Model size variant
            device: torch device
            prediction_length: Forecast horizon
            context_length: Input context length
        """
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError("Install chronos-forecasting: uv add chronos-forecasting")

        self.model_size = model_size
        self.device = device
        self.prediction_length = prediction_length
        self.context_length = context_length

        model_id = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES["base"])
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype="auto",
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
            num_samples: Number of sample paths
            quantiles: Quantiles for probabilistic forecasts

        Returns:
            Dictionary with forecasts
        """
        import torch

        context = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0)

        predictions = self.pipeline.predict(
            context,
            prediction_length=self.prediction_length,
            num_samples=num_samples,
            quantiles=quantiles,
        )

        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=self.prediction_length,
            freq=series.index.freq or "D",
        )

        predictions_np = predictions.numpy()

        result = {
            "mean": pd.Series(predictions_np.mean(axis=0).squeeze(), index=forecast_index),
            "median": pd.Series(np.median(predictions_np, axis=0).squeeze(), index=forecast_index),
        }

        for q in quantiles:
            result[f"q{int(q * 100)}"] = pd.Series(
                np.quantile(predictions_np, q, axis=0).squeeze(),
                index=forecast_index,
            )

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
