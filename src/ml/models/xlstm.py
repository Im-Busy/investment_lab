"""
xLSTM - Extended Long Short-Term Memory

Modern LSTM variant with exponential gating and matrix memory.
NeurIPS 2024 spotlight paper.

Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html
GitHub: https://github.com/NX-AI/xlstm
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class xLSTMForecaster:
    """
    xLSTM wrapper for long-term time series forecasting.

    Two variants:
    - sLSTM: Scalar memory with exponential gating
    - mLSTM: Matrix memory, fully parallelizable

    Use cases:
    - Sequential pattern recognition
    - Long-term dependency modeling
    - Alternative to Transformers when memory is constrained
    """

    def __init__(
        self,
        xlstm_type: str = "slstm",
        hidden_size: int = 128,
        num_layers: int = 4,
        sequence_length: int = 256,
        forecast_horizon: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        """
        Initialize xLSTM.

        Args:
            xlstm_type: Architecture variant
            hidden_size: Hidden layer dimension
            num_layers: Number of xLSTM layers
            sequence_length: Input sequence length
            forecast_horizon: Prediction horizon
            learning_rate: Learning rate
            device: torch device
        """
        self.xlstm_type = xlstm_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate
        self.device = device

        self.model = None
        self._available = False

        try:
            import torch

            self._available = True
        except ImportError:
            pass

    def _build_model(self, input_size: int) -> None:
        """Build xLSTM architecture."""
        import torch

        try:
            from xlstm import xLSTMTime
        except ImportError:
            raise ImportError(
                "Install xLSTM: pip install xlstm or check https://github.com/NX-AI/xlstm"
            )

        self.model = xLSTMTime(
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=self.forecast_horizon,
            xlstm_type=self.xlstm_type,
            num_layers=self.num_layers,
        ).to(self.device)

    def _create_sequences(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences."""
        X, y = [], []

        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i : i + self.sequence_length])
            y.append(
                data[i + self.sequence_length : i + self.sequence_length + self.forecast_horizon]
            )

        return np.array(X), np.array(y)

    def fit(
        self,
        series: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """
        Train xLSTM on time series.

        Args:
            series: Time series to learn from
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio

        Returns:
            Training history
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if not self._available:
            raise ImportError("xLSTM requires torch. Install with: uv add torch")

        X, y = self._create_sequences(series.values)

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        self._build_model(input_size=X_train.shape[-1])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_output = self.model(torch.FloatTensor(X_val).to(self.device))
                val_loss = criterion(
                    val_output,
                    torch.FloatTensor(y_val).to(self.device),
                ).item()

            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss)

        return history

    def predict(self, series: pd.Series) -> pd.Series:
        """Generate forecasts."""
        import torch

        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        context = series.values[-self.sequence_length :]
        X = torch.FloatTensor(context).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            forecast = self.model(X).squeeze().cpu().numpy()

        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=self.forecast_horizon,
            freq=series.index.freq or "D",
        )

        return pd.Series(forecast, index=forecast_index)
