"""
Drawdown / MAE as Primary ML Target (FS3)

Predicts worst-case drawdown (Maximum Adverse Excursion) within a trade horizon to
feed dynamic position sizing and adaptive stop-loss placement. Replaces fixed ATR-multiple
stops with ML-predicted maximum adverse excursion.

Key concepts:
- MAE (Maximum Adverse Excursion): worst-case % drawdown from entry within N bars
- MFE (Maximum Favorable Excursion): best-case % runup from entry within N bars
- MAE ratio: MAE / MFE — captures trade quality (lower is better)

Integration:
- Uses existing `forward_return_N` and `max_drawdown_N` columns from FeatureExtractor
- Generates regression targets (continuous MAE %) or classification labels (MAE buckets)
- Feeds into pattern classifier, position sizing, and dynamic stop placement

Example:
    >>> from src.ml.drawdown_target import DrawdownTargetGenerator
    >>> gen = DrawdownTargetGenerator(horizon=20)
    >>> labels = gen.generate_labels(df)
    >>> df["mae_target"] = labels["mae_pct"]
    >>> stops = gen.suggest_stops(df, predicted_mae=0.05)  # 5% predicted MAE
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.ml.feature_engineering import FeatureExtractor


class DrawdownTargetGenerator:
    """
    Generate drawdown/MAE-based targets for ML training.

    Computes forward Maximum Adverse Excursion (MAE) from each bar as a
    regression or classification target. Used to train models that predict
    worst-case drawdown before a trade becomes profitable.

    Args:
        horizon: Forward horizon in bars for drawdown calculation
        horizons: Multiple horizons for multi-horizon MAE computation
        atr_period: ATR period for ATR-normalized MAE
        mae_bins: Bin edges for categorical MAE classification (percentiles if None)
        use_atr_normalized: If True, return MAE in ATR multiples instead of %

    Example:
        >>> gen = DrawdownTargetGenerator(horizon=20)
        >>> labels = gen.generate_labels(df)
        >>> labels["mae_pct"]        # % drawdown
        >>> labels["mae_bucket"]      # Low/Medium/High
        >>> labels["mfe_pct"]         # % runup
        >>> labels["mae_ratio"]       # MAE / MFE
    """

    MAE_BINS = [-0.001, 0.02, 0.05, 0.10, 0.20, float("inf")]
    MAE_LABELS = ["Tight", "Low", "Medium", "High", "Extreme"]

    def __init__(
        self,
        horizon: int = 20,
        horizons: Optional[List[int]] = None,
        atr_period: int = 20,
        mae_bins: Optional[List[float]] = None,
        use_atr_normalized: bool = False,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self.horizon = horizon
        self.horizons = horizons or [5, 10, 20]
        self.atr_period = atr_period
        self.mae_bins = mae_bins or self.MAE_BINS
        self.mae_labels = self.MAE_LABELS[: len(self.mae_bins) - 1]
        self.use_atr_normalized = use_atr_normalized
        self._extractor = FeatureExtractor()

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MAE/MFE labels from OHLCV data.

        Args:
            df: OHLCV DataFrame with Open, High, Low, Close, Volume columns

        Returns:
            DataFrame with columns: mae_pct, mae_bucket, mfe_pct, mae_ratio,
            mae_atr, and per-horizon variants

        Raises:
            ValueError: If df has insufficient rows or missing columns
        """
        self._validate_input(df)

        labels = pd.DataFrame(index=df.index)
        close = df["Close"].values
        low = df["Low"].values
        high = df["High"].values

        for h in self.horizons:
            mae_arr = np.full(len(df), np.nan)
            mfe_arr = np.full(len(df), np.nan)

            for i in range(len(df) - h):
                future_low = low[i + 1 : i + 1 + h].min()
                future_high = high[i + 1 : i + 1 + h].max()
                entry = close[i]

                mae_arr[i] = max(0.0, (entry - future_low) / entry)
                mfe_arr[i] = max(0.0, (future_high - entry) / entry)

            labels[f"mae_pct_{h}"] = mae_arr
            labels[f"mfe_pct_{h}"] = mfe_arr
            labels[f"mae_ratio_{h}"] = np.where(
                mfe_arr > 0,
                np.divide(mae_arr, mfe_arr, out=np.full_like(mae_arr, np.nan), where=mfe_arr > 0),
                np.nan,
            )

        labels["mae_pct"] = labels[f"mae_pct_{self.horizon}"]
        labels["mfe_pct"] = labels[f"mfe_pct_{self.horizon}"]
        labels["mae_ratio"] = labels[f"mae_ratio_{self.horizon}"]

        labels["mae_bucket"] = pd.cut(
            labels["mae_pct"],
            bins=self.mae_bins,
            labels=self.mae_labels,
            include_lowest=True,
        )

        labels["mae_bucket_int"] = pd.cut(
            labels["mae_pct"],
            bins=self.mae_bins,
            labels=False,
            include_lowest=True,
        )

        if self.use_atr_normalized:
            atr = self._compute_atr(df, self.atr_period)
            labels["atr"] = atr.values
            labels["mae_atr"] = (labels["mae_pct"] * close) / (atr.values + 1e-10)

        return labels

    def generate_targets_at_signals(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Generate MAE targets at specific signal timestamps.

        Args:
            df: OHLCV DataFrame
            signals: Signal DataFrame with timestamp column
            timestamp_col: Name of timestamp column in signals

        Returns:
            Signals DataFrame with added MAE target columns
        """
        labels = self.generate_labels(df)
        all_targets = signals.copy()

        # Initialize target columns with proper dtype (object for mixed float/string)
        for col in labels.columns:
            if "bucket" in col:
                all_targets[col] = None
            else:
                all_targets[col] = np.nan

        for idx, row in signals.iterrows():
            ts = row[timestamp_col]
            if ts in labels.index:
                aligned = labels.loc[ts]
                if isinstance(aligned, pd.DataFrame):
                    aligned = aligned.iloc[0]
                for col in labels.columns:
                    all_targets.at[idx, col] = aligned[col]

        return all_targets

    def suggest_stops(
        self,
        df: pd.DataFrame,
        predicted_mae: float = 0.05,
        multiplier: float = 1.5,
        method: str = "mae_multiplier",
    ) -> pd.Series:
        """
        Suggest dynamic stop levels based on predicted MAE.

        Args:
            df: OHLCV DataFrame
            predicted_mae: ML-predicted MAE as decimal (e.g., 0.05 = 5%)
            multiplier: Safety margin multiplier on predicted MAE
            method: "mae_multiplier" or "atr_multiplier"

        Returns:
            Series of stop-loss prices aligned to df index

        Raises:
            ValueError: If method is unknown
        """
        close = df["Close"]

        if method == "mae_multiplier":
            stop_distance = predicted_mae * multiplier
            return close * (1 - stop_distance)

        elif method == "atr_multiplier":
            atr = self._compute_atr(df, self.atr_period)
            stop_distance = atr * multiplier
            return close - stop_distance

        else:
            raise ValueError(f"Unknown stop method: {method}")

    def generate_trade_level_targets(
        self,
        df: pd.DataFrame,
        signal_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        price_col: str = "entry_price",
    ) -> pd.DataFrame:
        """
        Generate MAE/MFE targets at the trade level, using entry prices.

        For each trade, computes forward MAE and MFE from the entry price
        within the horizon.

        Args:
            df: OHLCV DataFrame with full price history
            signal_df: Signal DataFrame with timestamps and entry prices
            timestamp_col: Column with signal timestamps
            price_col: Column with entry prices

        Returns:
            Signal DataFrame with trade-level MAE/MFE columns
        """
        close = df["Close"]
        low = df["Low"]
        high = df["High"]

        trades = signal_df.copy()
        trades["mae_pct"] = np.nan
        trades["mfe_pct"] = np.nan
        trades["mae_bucket"] = np.nan
        trades["mae_price"] = np.nan
        trades["mfe_price"] = np.nan
        trades["trade_return"] = np.nan

        for idx, row in trades.iterrows():
            ts = row[timestamp_col]
            entry_price = row[price_col]

            if ts not in df.index:
                continue

            pos = int(df.index.get_loc(ts))
            end_pos = min(pos + self.horizon + 1, len(df))

            if end_pos <= pos + 1:
                continue

            future_low = low.iloc[pos + 1 : end_pos].min()
            future_high = high.iloc[pos + 1 : end_pos].max()
            future_close = close.iloc[end_pos - 1]

            mae_pct = max(0.0, (entry_price - future_low) / entry_price)
            mfe_pct = max(0.0, (future_high - entry_price) / entry_price)

            trades.at[idx, "mae_pct"] = mae_pct
            trades.at[idx, "mfe_pct"] = mfe_pct
            trades.at[idx, "mae_price"] = entry_price * (1 - mae_pct)
            trades.at[idx, "mfe_price"] = entry_price * (1 + mfe_pct)
            trades.at[idx, "trade_return"] = (future_close - entry_price) / entry_price

        mae_data = trades["mae_pct"].dropna()
        if len(mae_data) > 0:
            trades["mae_bucket"] = pd.cut(
                trades["mae_pct"],
                bins=self.mae_bins,
                labels=self.mae_labels,
                include_lowest=True,
            )

        return trades

    def get_mae_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get MAE distribution statistics for model calibration.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with MAE quantiles and bucket frequencies per horizon
        """
        labels = self.generate_labels(df)
        stats = {}

        for h in self.horizons:
            col = f"mae_pct_{h}"
            data = labels[col].dropna()

            stats[f"horizon_{h}"] = {
                "count": len(data),
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "q25": data.quantile(0.25),
                "q75": data.quantile(0.75),
                "q90": data.quantile(0.90),
                "q95": data.quantile(0.95),
                "q99": data.quantile(0.99),
                "max": data.max(),
            }

        return pd.DataFrame(stats).T

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if len(df) < self.horizon + 1:
            raise ValueError(
                f"Insufficient rows: {len(df)} < {self.horizon + 1} (horizon + 1 required)"
            )

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()
