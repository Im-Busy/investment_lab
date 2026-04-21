"""
Feature Store — Pre-computed feature cache as parquet.

Computes features once from OHLCV data for multiple tickers and stores them
as parquet files. Serves features to any model without recomputation,
ensuring consistent point-in-time features across all experiments.

Directory structure:
    experiments/features/
    ├── feature_store.parquet    # ticker x date x features (flattened)
    ├── metadata.json             # Feature names, tickers, date range, version
    └── labels.parquet           # Forward returns for each ticker/date

Usage:
    from src.ml.feature_store import FeatureStore

    store = FeatureStore()
    # Build from raw OHLCV data
    store.build(ohlcv_dict={'SPY': df, 'QQQ': df2})
    # Load cached features
    store.load()
    # Get features for specific ticker/date range
    X = store.get_features(tickers=['SPY'], start='2020-01-01', end='2024-12-31')
    y = store.get_labels(label_horizon=5)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.ml.features import FeatureEngineer

logger = logging.getLogger(__name__)

FEATURES_DIR = Path("experiments") / "features"


class FeatureStore:
    """Pre-computed feature cache with parquet storage.

    Computes features once, stores as parquet, and serves consistent
    point-in-time features to all downstream models.

    Directory structure: experiments/features/feature_store.parquet

    Attributes:
        store_dir: Base directory for feature store files.
        feature_path: Path to the parquet feature file.
        metadata_path: Path to the metadata JSON file.
        labels_path: Path to the labels parquet file.
    """

    def __init__(
        self,
        store_dir: Path | str = FEATURES_DIR,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.feature_path = self.store_dir / "feature_store.parquet"
        self.metadata_path = self.store_dir / "metadata.json"
        self.labels_path = self.store_dir / "labels.parquet"
        self._features: pd.DataFrame | None = None
        self._metadata: dict[str, Any] = {}
        self._engineer = feature_engineer or FeatureEngineer()
        self._labels: pd.DataFrame | None = None

    def build(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        compute_labels: bool = True,
        label_horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute and cache features from OHLCV data for multiple tickers.

        Args:
            ohlcv_data: Dict of ticker -> OHLCV DataFrame with DatetimeIndex.
            compute_labels: Whether to compute forward returns as labels.
            label_horizons: Forward return horizons (e.g. [1, 5, 20]).

        Returns:
            DataFrame with columns: ticker, date, feature1, feature2, ...
        """
        start = time.time()
        all_rows = []
        label_rows = []
        label_horizons = label_horizons or [1, 5, 20]

        for ticker, df in ohlcv_data.items():
            features = self._engineer.generate_features(df)
            features = features.dropna()

            if len(features) == 0:
                logger.warning(f"No valid features for {ticker}, skipping")
                continue

            features = features.copy()
            features.insert(0, "ticker", ticker)
            features.insert(1, "date", features.index.astype(str))
            all_rows.append(features.reset_index(drop=True))

            if compute_labels:
                for h in label_horizons:
                    fwd_returns = df["Close"].pct_change(h).shift(-h)
                    fwd_returns = fwd_returns.dropna()
                    fwd_returns = fwd_returns.rename(f"forward_return_{h}d")
                    fwd_returns.index.name = "date"
                    lr = pd.DataFrame(
                        {
                            "ticker": ticker,
                            "date": [d.isoformat()[:10] for d in fwd_returns.index],
                            f"forward_return_{h}d": fwd_returns.values,
                        }
                    )
                    label_rows.append(lr)

        if not all_rows:
            raise ValueError("No valid features generated from any ticker")

        self._features = pd.concat(all_rows, ignore_index=True)

        if label_rows:
            self._labels = pd.concat(label_rows, ignore_index=True)

        self._metadata = {
            "n_tickers": len(ohlcv_data),
            "tickers": sorted(ohlcv_data.keys()),
            "n_features": len(self._features.columns) - 2,
            "feature_names": [c for c in self._features.columns if c not in ("ticker", "date")],
            "n_samples": len(self._features),
            "date_range": {
                "start": str(self._features["date"].min()),
                "end": str(self._features["date"].max()),
            }
            if "date" in self._features.columns
            else {},
            "label_horizons": label_horizons if compute_labels else [],
            "engineer_params": {
                "price_windows": self._engineer.price_windows,
                "momentum_windows": self._engineer.momentum_windows,
                "volatility_windows": self._engineer.volatility_windows,
            },
        }
        self._save()

        elapsed = time.time() - start
        logger.info(
            f"Built feature store: {self._metadata['n_tickers']} tickers, "
            f"{self._metadata['n_features']} features, {self._metadata['n_samples']} samples "
            f"in {elapsed:.2f}s"
        )
        return self._features

    def load(self) -> pd.DataFrame:
        """Load cached features from parquet.

        Returns:
            DataFrame with all cached features.

        Raises:
            FileNotFoundError: If feature store not found.
        """
        if not self.feature_path.exists():
            raise FileNotFoundError(f"Feature store not found at {self.feature_path}")

        self._features = pd.read_parquet(self.feature_path)

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                self._metadata = json.load(f)

        if self.labels_path.exists():
            self._labels = pd.read_parquet(self.labels_path)

        logger.info(f"Loaded feature store: {len(self._features)} samples")
        return self._features

    def get_features(
        self,
        tickers: Sequence[str] | None = None,
        start: str = "",
        end: str = "",
        exclude_features: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Get features with optional filtering.

        Args:
            tickers: Filter to these tickers. None = all.
            start: Start date filter (YYYY-MM-DD).
            end: End date filter (YYYY-MM-DD).
            exclude_features: Feature column names to exclude.

        Returns:
            Filtered feature DataFrame.
        """
        if self._features is None:
            self.load()

        df = self._features.copy()

        if tickers:
            df = df[df["ticker"].isin(tickers)]
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        if exclude_features:
            cols = [c for c in df.columns if c not in exclude_features]
            df = df[cols]

        return df

    def get_labels(
        self,
        tickers: Sequence[str] | None = None,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """Get forward return labels.

        Args:
            tickers: Filter to these tickers. None = all.
            horizon: Forward return horizon in days.

        Returns:
            DataFrame with ticker, date, and forward_return_Xd column.
        """
        if self._labels is None:
            raise ValueError("Labels not computed. Build with compute_labels=True")

        df = self._labels.copy()
        if tickers:
            df = df[df["ticker"].isin(tickers)]

        col = f"forward_return_{horizon}d"
        if col not in df.columns:
            raise ValueError(
                f"Horizon {horizon}d not found. Available: {[c for c in df.columns if c.startswith('forward_return_')]}"
            )

        return df[["ticker", "date", col]]

    def get_feature_names(self) -> list[str]:
        """Get list of feature column names."""
        if self._metadata:
            return self._metadata.get("feature_names", [])
        if self._features is not None:
            return [c for c in self._features.columns if c not in ("ticker", "date")]
        return []

    def get_tickers(self) -> list[str]:
        """Get list of tickers in the store."""
        if self._metadata:
            return self._metadata.get("tickers", [])
        if self._features is not None:
            return sorted(self._features["ticker"].unique().tolist())
        return []

    def is_loaded(self) -> bool:
        """Check if features are loaded."""
        return self._features is not None

    def _save(self) -> None:
        """Save features, labels, and metadata to disk."""
        self.store_dir.mkdir(parents=True, exist_ok=True)

        if self._features is not None:
            self._features.to_parquet(self.feature_path, index=False)
        if self._labels is not None:
            self._labels.to_parquet(self.labels_path, index=False)
        if self._metadata:
            with open(self.metadata_path, "w") as f:
                json.dump(self._metadata, f, indent=2)
