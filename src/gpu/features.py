"""
GPU-Accelerated Feature Engineering

Applies GPU-friendly principles to technical indicator computation:
1. Data parallelism: All bars computed simultaneously via tensor operations
2. Minimal branching: torch.where() replaces conditional assignments
3. Coalesced memory: Features stacked as [N, F] contiguous tensor
4. Batched transfers: Single CPU->GPU transfer; all features computed on-device
5. Tensor cores: unfold + dot product for rolling and exponential statistics
6. Maximized occupancy: Native PyTorch primitives handle thread scheduling

Indicator equivalence:
- SMA: padding + unfold + mean (O(n) parallel)
- EMA: truncated exponential convolution (O(n*T) parallel, zero Python loops)
- ATR: True Range + Wilder smoothing via exponential convolution
- RSI: Gains/losses + Wilder smoothing via exponential convolution
- ADX: +DM/-DM + Wilder smoothing via exponential convolution
- Bollinger Bands: SMA + rolling std (unfold)
- Donchian: rolling max/min via unfold
- Volume Profile: parallel histogram per price level
- Returns: simple/log differencing (element-wise)
- Normalization: z-score, min-max, robust (all element-wise)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .smooth import adx_parallel, atr_parallel, ema_parallel, rsi_parallel
from .utils import get_device, to_numpy, to_tensor


class GPUFeatureEngineer:
    """
    GPU-accelerated feature engineering for OHLCV data.

    Computes all technical indicators and derived features on GPU
    in a single pass — no per-bar Python loops.

    Usage:
        engineer = GPUFeatureEngineer(device)
        engineer.load_data(df)
        features = engineer.compute_all()
        # features: torch.Tensor [N, F]
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self._ohlcv: Optional[torch.Tensor] = None
        self._n_bars: int = 0
        self._feature_names: List[str] = []

    def load_data(self, df: pd.DataFrame) -> None:
        """Transfer OHLCV data to GPU once (Principle 4)."""
        ohlcv_np = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
        self._ohlcv = to_tensor(ohlcv_np, device=self.device, dtype=torch.float32)
        self._n_bars = self._ohlcv.shape[0]
        self._feature_names = []

    @property
    def ohlcv(self) -> torch.Tensor:
        if self._ohlcv is None:
            raise RuntimeError("load_data() must be called before feature computation")
        return self._ohlcv

    @property
    def close(self) -> torch.Tensor:
        return self.ohlcv[:, 3]

    @property
    def high(self) -> torch.Tensor:
        return self.ohlcv[:, 1]

    @property
    def low(self) -> torch.Tensor:
        return self.ohlcv[:, 2]

    @property
    def volume(self) -> torch.Tensor:
        return self.ohlcv[:, 4]

    @property
    def typical_price(self) -> torch.Tensor:
        return (self.high + self.low + self.close) / 3.0

    # ------------------------------------------------------------------
    # Parallel Rolling Window Operations
    # ------------------------------------------------------------------

    def rolling_mean(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """
        GPU-parallel rolling mean using unfold.

        Principle 1 (Data Parallelism): All windows computed simultaneously.
        No Python for-loop — PyTorch unfold handles batching internally.

        Equivalent to: pandas.Series.rolling(window).mean()
        """
        if window <= 1:
            return x.clone()
        x_pad = F.pad(x.view(1, 1, -1), (window - 1, 0), mode="replicate")
        return x_pad.unfold(2, window, 1).mean(dim=3).view(-1)

    def rolling_std(self, x: torch.Tensor, window: int, ddof: int = 1) -> torch.Tensor:
        """GPU-parallel rolling standard deviation using unfold."""
        if window < 2:
            return torch.zeros_like(x)
        x_pad = F.pad(x.view(1, 1, -1), (window - 1, 0), mode="replicate")
        windows = x_pad.unfold(2, window, 1).squeeze(0).squeeze(0)  # [N, window]
        return windows.std(dim=1, correction=ddof)

    def rolling_min(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-parallel rolling minimum."""
        if window <= 1:
            return x.clone()
        x_pad = F.pad(x.view(1, 1, -1), (window - 1, 0), mode="replicate")
        return x_pad.unfold(2, window, 1).min(dim=3).values.view(-1)

    def rolling_max(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-parallel rolling maximum."""
        if window <= 1:
            return x.clone()
        x_pad = F.pad(x.view(1, 1, -1), (window - 1, 0), mode="replicate")
        return x_pad.unfold(2, window, 1).max(dim=3).values.view(-1)

    def rolling_sum(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """GPU-parallel rolling sum."""
        if window <= 1:
            return x.clone()
        x_pad = F.pad(x.view(1, 1, -1), (window - 1, 0), mode="replicate")
        return x_pad.unfold(2, window, 1).sum(dim=3).view(-1)

    # ------------------------------------------------------------------
    # Sequential Indicators (inherently recursive, vectorized on GPU)
    # ------------------------------------------------------------------

    def ema(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """
        GPU EMA via truncated exponential convolution.

        Principle 1 (Data Parallelism): All N bars computed simultaneously
        via unfold + kernel dot product. Zero Python for-loops.

        Principle 5 (Tensor Cores): unfold + sum is matmul-friendly.
        """
        return ema_parallel(x, period)

    def rsi(self, period: int = 14) -> torch.Tensor:
        """
        GPU RSI — parallel Wilder smoothing via exponential convolution.

        Principle 1 (Data Parallelism): Gains/losses separated element-wise,
        then Wilder-smoothed in parallel via truncated convolution.
        Zero Python for-loops.

        Equivalent to: src/indicators/technical_numba.py rsi_numba()
        """
        return rsi_parallel(self.close, period)

    def atr(self, period: int = 14) -> torch.Tensor:
        """
        GPU ATR — parallel Wilder smoothing via exponential convolution.

        Principle 1 (Data Parallelism): True Range computed element-wise,
        then Wilder-smoothed in parallel. Zero Python for-loops.

        Equivalent to: src/indicators/technical_numba.py atr_numba()
        """
        return atr_parallel(self.high, self.low, self.close, period)

    def adx(self, period: int = 14) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU ADX — parallel Wilder smoothing via exponential convolution.

        Principle 1 (Data Parallelism): +DM, -DM, TR computed element-wise,
        then all three Wilder-smoothed in parallel. DX computed element-wise,
        then Wilder-smoothed again. Zero Python for-loops.

        Returns: (adx, plus_di, minus_di)

        Equivalent to: src/indicators/technical_numba.py adx_numba()
        """
        return adx_parallel(self.high, self.low, self.close, period)

    # ------------------------------------------------------------------
    # Derived Features
    # ------------------------------------------------------------------

    def returns(self, log_returns: bool = False) -> torch.Tensor:
        """Simple or log returns (element-wise, fully parallel)."""
        c = self.close
        if log_returns:
            return torch.log(c / torch.roll(c, 1, 0))
        return (c - torch.roll(c, 1, 0)) / torch.roll(c, 1, 0)

    def volatility(self, period: int = 20) -> torch.Tensor:
        """Annualized rolling volatility."""
        ret = self.returns(log_returns=True)
        ret = torch.nan_to_num(ret, nan=0.0)
        vol = self.rolling_std(ret, period) * np.sqrt(252)
        vol[:period] = float("nan")
        return vol

    def volume_ratio(self, period: int = 20) -> torch.Tensor:
        """Current volume / rolling average volume."""
        avg_vol = self.rolling_mean(self.volume, period)
        return self.volume / (avg_vol + 1e-9)

    def price_position(self, period: int = 20) -> torch.Tensor:
        """
        Price position within rolling range: (close - low) / (high - low).
        Fully parallel via rolling min/max.
        """
        roll_high = self.rolling_max(self.high, period)
        roll_low = self.rolling_min(self.low, period)
        roll_range = roll_high - roll_low
        pos = torch.where(
            roll_range > 0,
            (self.close - roll_low) / roll_range,
            torch.tensor(0.5, device=self.device),
        )
        pos[:period] = float("nan")
        return pos

    def bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bollinger Bands: middle, upper, lower."""
        middle = self.rolling_mean(self.close, period)
        std = self.rolling_std(self.close, period)
        upper = middle + num_std * std
        lower = middle - num_std * std
        return middle, upper, lower

    def bb_position(self, period: int = 20) -> torch.Tensor:
        """Bollinger Band percentile position."""
        _, upper, lower = self.bollinger_bands(period)
        bandwidth = upper - lower
        return torch.where(
            bandwidth > 0,
            (self.close - lower) / bandwidth,
            torch.tensor(0.5, device=self.device),
        )

    def donchian(self, period: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Donchian channel: upper, lower, middle."""
        upper = self.rolling_max(self.high, period)
        lower = self.rolling_min(self.low, period)
        middle = (upper + lower) / 2.0
        return upper, lower, middle

    def donchian_position(self, period: int = 20) -> torch.Tensor:
        """Donchian channel position: (close - lower) / (upper - lower)."""
        upper, lower, _ = self.donchian(period)
        channel_width = upper - lower
        return torch.where(
            channel_width > 0,
            (self.close - lower) / channel_width,
            torch.tensor(0.5, device=self.device),
        )

    def z_score(self, period: int = 20) -> torch.Tensor:
        """Rolling z-score of close price."""
        mean = self.rolling_mean(self.close, period)
        std = self.rolling_std(self.close, period)
        return (self.close - mean) / (std + 1e-9)

    # ------------------------------------------------------------------
    # Feature Assembly
    # ------------------------------------------------------------------

    def compute_all(self) -> torch.Tensor:
        """
        Compute all GPU features and return as [N, F] tensor.

        Features (F=18):
        0:  SMA_20        - 20-bar simple moving average of close
        1:  EMA_14        - 14-bar exponential moving average
        2:  RSI_14        - 14-bar RSI
        3:  ATR_14        - 14-bar Average True Range
        4:  ADX_14        - 14-bar ADX
        5:  RET_1         - 1-bar returns
        6:  RET_5         - 5-bar returns
        7:  VOL_20        - 20-bar rolling volatility (annualized)
        8:  VOL_RATIO_20  - volume / 20-bar avg volume
        9:  PRICE_POS_20  - price position in 20-bar range
        10: BB_MIDDLE_20  - Bollinger middle band
        11: BB_WIDTH_20   - Bollinger band width
        12: DONCHIAN_POS  - Donchian channel position
        13: Z_SCORE_20    - rolling z-score
        14: PRICE_DIFF    - close - SMA_20
        15: HIGH_LOW      - daily range
        16: BODY_RATIO    - |close-open| / range
        17: CLOSE_CHANGE  - close - close[1]

        Returns:
            [N, 18] torch.Tensor on device
        """
        features: List[torch.Tensor] = []
        self._feature_names = [
            "SMA_20", "EMA_14", "RSI_14", "ATR_14", "ADX_14",
            "RET_1", "RET_5", "VOL_20", "VOL_RATIO_20", "PRICE_POS_20",
            "BB_MIDDLE_20", "BB_WIDTH_20", "DONCHIAN_POS", "Z_SCORE_20",
            "PRICE_DIFF", "HIGH_LOW", "BODY_RATIO", "CLOSE_CHANGE",
        ]

        features.append(self.rolling_mean(self.close, 20))  # SMA_20
        features.append(self.ema(self.close, 14))  # EMA_14
        features.append(self.rsi(14))  # RSI_14
        features.append(self.atr(14))  # ATR_14
        adx_val, _, _ = self.adx(14)
        features.append(adx_val)  # ADX_14

        ret_1 = self.returns()
        features.append(torch.nan_to_num(ret_1, nan=0.0))  # RET_1

        ret_5 = (self.close - torch.roll(self.close, 5, 0)) / (torch.roll(self.close, 5, 0) + 1e-9)
        features.append(torch.nan_to_num(ret_5, nan=0.0))  # RET_5

        features.append(self.volatility(20))  # VOL_20
        features.append(self.volume_ratio(20))  # VOL_RATIO_20
        features.append(self.price_position(20))  # PRICE_POS_20

        bb_mid, bb_up, bb_low = self.bollinger_bands(20)
        features.append(bb_mid)  # BB_MIDDLE_20
        features.append((bb_up - bb_low) / (bb_mid + 1e-9))  # BB_WIDTH_20

        features.append(self.donchian_position(20))  # DONCHIAN_POS
        features.append(self.z_score(20))  # Z_SCORE_20
        features.append(self.close - self.rolling_mean(self.close, 20))  # PRICE_DIFF

        daily_range = self.high - self.low
        features.append(daily_range)  # HIGH_LOW

        body = torch.abs(self.close - self.open)
        body_ratio = torch.where(daily_range > 0, body / daily_range, torch.tensor(0.0, device=self.device))
        features.append(body_ratio)  # BODY_RATIO

        features.append(self.close - torch.roll(self.close, 1, 0))  # CLOSE_CHANGE

        feature_tensor = torch.stack(features, dim=1)
        feature_tensor = torch.nan_to_num(feature_tensor, nan=0.0)

        self._feature_names = self._feature_names
        return feature_tensor

    def to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Export computed features to pandas DataFrame."""
        features = self.compute_all()
        features_np = to_numpy(features)
        return pd.DataFrame(features_np, index=df.index, columns=self._feature_names)

    def get_feature_names(self) -> List[str]:
        return self._feature_names
