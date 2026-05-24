"""
Volatility Forecasting with Gradient Boosting.

Predicts realized volatility over a forward horizon of N bars, enabling
dynamic position sizing based on expected risk. Uses CatBoost regression
with financial time-series features including rolling volatility windows,
ATR, Bollinger Band width, volume, and return-based predictors.

The target is annualized realized volatility computed as:
    realized_vol = std(log_returns[t:t+horizon]) * sqrt(252)

Usage:
    from src.ml.volatility_forecaster import VolatilityForecaster

    forecaster = VolatilityForecaster(horizon=10)
    forecaster.fit(X, returns)
    vol_pred = forecaster.predict(X_test)
    position_size = forecaster.adjust_position_size(equity, entry_price, vol_pred)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ANNUALIZATION = np.sqrt(252)


@dataclass
class VolForecastResult:
    """Results from volatility forecasting training.

    Attributes:
        horizon: Forecast horizon in bars.
        rmse: Root mean squared error.
        mae: Mean absolute error.
        r2: R-squared score.
        direction_accuracy: Fraction where predicted direction matches actual.
        feature_importance: Dict of feature name -> importance score.
        n_train: Number of training samples.
    """

    horizon: int
    rmse: float
    mae: float
    r2: float
    direction_accuracy: float
    feature_importance: Dict[str, float]
    n_train: int
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "r2": round(self.r2, 4),
            "direction_accuracy": round(self.direction_accuracy, 4),
            "n_train": self.n_train,
        }

    def __repr__(self) -> str:
        return (
            f"VolForecastResult(h={self.horizon}, rmse={self.rmse:.4f}, "
            f"r2={self.r2:.4f}, dir_acc={self.direction_accuracy:.3f})"
        )


class VolatilityForecaster:
    """Predict realized volatility using CatBoost regression.

    Generates features from price/volume data and trains a gradient boosting
    model to predict forward realized volatility.

    Core features generated automatically:
      - Rolling volatility (5, 10, 20, 50, 100 bars)
      - ATR and ATR relative to close
      - Bollinger Band width (volatility proxy)
      - Volume surge indicators
      - Return skewness/kurtosis
      - High-low range

    Example:
        >>> forecaster = VolatilityForecaster(horizon=20)
        >>> result = forecaster.fit(df)
        >>> vol_pred = forecaster.predict(df_test)
        >>> shares = forecaster.adjust_position_size(100000, 150.0, vol_pred)
    """

    def __init__(
        self,
        horizon: int = 10,
        n_estimators: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        verbose: bool = False,
    ):
        """Initialize volatility forecaster.

        Args:
            horizon: Forward bars to forecast volatility over.
            n_estimators: Number of CatBoost trees.
            depth: Tree depth.
            learning_rate: Learning rate.
            l2_leaf_reg: L2 regularization.
            random_seed: Random seed.
            verbose: Whether to print training progress.
        """
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose

        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    @staticmethod
    def compute_target(
        returns: pd.Series,
        horizon: int,
        annualize: bool = True,
    ) -> pd.Series:
        """Compute realized volatility target.

        Args:
            returns: Log or simple returns series.
            horizon: Forward window for volatility calculation.
            annualize: Whether to annualize (multiply by sqrt(252)).

        Returns:
            Series of realized volatility, aligned to original index.
        """
        vol = returns.rolling(horizon).std().shift(-horizon + 1)
        if annualize:
            vol = vol * ANNUALIZATION
        return vol

    @staticmethod
    def generate_features(
        df: pd.DataFrame,
        returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Generate volatility prediction features from OHLCV data.

        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume.
            returns: Optional pre-computed returns (computed from Close if None).

        Returns:
            DataFrame of features, indexed same as df.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df.get("Volume", pd.Series(1.0, index=df.index))

        if returns is None:
            returns = close.pct_change()

        log_ret = np.log(close / close.shift(1))

        features = pd.DataFrame(index=df.index)

        # Rolling realized volatility at multiple windows
        for w in [5, 10, 20, 50, 100]:
            features[f"vol_{w}"] = returns.rolling(w).std() * ANNUALIZATION
            features[f"vol_log_{w}"] = log_ret.rolling(w).std() * ANNUALIZATION

        # ATR-based volatility
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for w in [14, 20]:
            atr_val = tr.rolling(w).mean()
            features[f"atr_{w}"] = atr_val
            features[f"atr_pct_{w}"] = atr_val / close

        # Bollinger Band width (volatility proxy)
        for w in [20, 50]:
            ma = close.rolling(w).mean()
            std = close.rolling(w).std()
            features[f"bb_width_{w}"] = (std * 2) / ma

        # Volume features
        vol_ma_20 = volume.rolling(20).mean()
        features["volume_ratio"] = volume / vol_ma_20
        features["volume_std"] = volume.rolling(20).std() / vol_ma_20

        # High-low range
        features["hl_range"] = (high - low) / close
        features["hl_range_ma5"] = ((high - low) / close).rolling(5).mean()

        # Return distribution features
        features["ret_skew_20"] = returns.rolling(20).skew()
        features["ret_kurt_20"] = returns.rolling(20).kurt()
        features["ret_mean_5"] = returns.rolling(5).mean()
        features["ret_mean_20"] = returns.rolling(20).mean()

        # Volatility regime: short-term vs long-term
        features["vol_regime_10_50"] = features["vol_10"] / features["vol_50"].replace(0, np.nan)
        features["vol_regime_20_100"] = features["vol_20"] / features["vol_100"].replace(0, np.nan)

        # Volatility momentum (change in vol)
        for w in [10, 20]:
            features[f"vol_chg_{w}"] = features["vol_10"].diff(w)

        # Close relative to moving averages
        for w in [20, 50]:
            features[f"close_ma_{w}"] = close / close.rolling(w).mean()

        # Log-vol (more normally distributed)
        for w in [10, 20, 50]:
            vol_col = features[f"vol_{w}"]
            mask = vol_col > 0
            features.loc[mask, f"log_vol_{w}"] = np.log(vol_col.loc[mask])

        return features

    def fit(
        self,
        df: pd.DataFrame,
        returns: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
    ) -> VolatilityForecaster:
        """Train the volatility forecasting model.

        Automatically generates features and computes the target (realized
        volatility over self.horizon bars).

        Args:
            df: DataFrame with OHLCV columns.
            returns: Optional pre-computed returns.
            cat_features: Categorical feature names.

        Returns:
            Self for chaining.
        """
        if returns is None:
            returns = df["Close"].pct_change()

        X_raw = self.generate_features(df, returns)
        y_raw = self.compute_target(returns, self.horizon, annualize=True)

        mask = X_raw.notna().all(axis=1) & y_raw.notna()
        X = X_raw.loc[mask].copy()
        y = y_raw.loc[mask].copy()

        self._feature_names = list(X.columns)
        self._target_mean = float(y.mean())
        self._target_std = float(y.std())

        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            loss_function="Huber:delta=1.0",
            task_type="CPU",
        )
        self._model.fit(X.values, y.values, cat_features=cat_features or [])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict forward realized volatility.

        Args:
            df: DataFrame with OHLCV columns.

        Returns:
            Array of predicted annualized volatility.
        """
        self._check_fitted()
        X = self.generate_features(df)
        mask = X.notna().all(axis=1)
        preds = np.full(len(df), np.nan, dtype=np.float64)
        preds[mask.values] = self._model.predict(X.loc[mask].values)
        return preds

    def predict_single(self, features: pd.DataFrame) -> float:
        """Predict volatility for a single feature row.

        Args:
            features: DataFrame with one row of features.

        Returns:
            Predicted annualized volatility.
        """
        self._check_fitted()
        return float(self._model.predict(features.values)[0])

    def adjust_position_size(
        self,
        equity: float,
        entry_price: float,
        predicted_vol: float,
        target_vol: float = 0.01,
        max_position_pct: float = 0.20,
    ) -> float:
        """Adjust position size based on predicted volatility.

        Scales position inversely with expected volatility to achieve a
        target portfolio volatility contribution.

        Args:
            equity: Current portfolio equity.
            entry_price: Entry price per share.
            predicted_vol: Predicted annualized volatility (from predict()).
            target_vol: Target annualized volatility contribution (default 1%).
            max_position_pct: Maximum position as fraction of equity.

        Returns:
            Number of shares to trade.
        """
        if predicted_vol <= 0 or np.isnan(predicted_vol):
            predicted_vol = 0.20

        vol_scalar = target_vol / predicted_vol
        vol_scalar = np.clip(vol_scalar, 0.01, 2.0)

        notional = equity * vol_scalar
        notional = min(notional, equity * max_position_pct)

        shares = notional / entry_price
        return float(np.floor(shares))

    def evaluate(
        self,
        df: pd.DataFrame,
        returns: Optional[pd.Series] = None,
    ) -> VolForecastResult:
        """Evaluate the volatility forecaster on data.

        Args:
            df: DataFrame with OHLCV columns.
            returns: Optional pre-computed returns.

        Returns:
            VolForecastResult with metrics.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        self._check_fitted()

        if returns is None:
            returns = df["Close"].pct_change()

        y_true = self.compute_target(returns, self.horizon, annualize=True)
        y_pred = self.predict(df)

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_t = y_true.values[mask]
        y_p = y_pred[mask]

        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        mae = float(mean_absolute_error(y_t, y_p))
        r2 = float(r2_score(y_t, y_p))

        # Direction accuracy
        if len(y_t) > 1:
            actual_dir = np.sign(np.diff(y_t))
            pred_dir = np.sign(np.diff(y_p))
            dir_acc = float(np.mean(actual_dir == pred_dir))
        else:
            dir_acc = float("nan")

        feat_imp = self._get_feature_importance()

        return VolForecastResult(
            horizon=self.horizon,
            rmse=rmse,
            mae=mae,
            r2=r2,
            direction_accuracy=dir_acc,
            feature_importance=feat_imp,
            n_train=len(y_t),
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        pct_embargo: float = 0.05,
        returns: Optional[pd.Series] = None,
    ) -> VolForecastResult:
        """Purged time-series cross-validation.

        Args:
            df: DataFrame with OHLCV columns.
            n_splits: Number of folds.
            pct_embargo: Embargo fraction.
            returns: Optional pre-computed returns.

        Returns:
            VolForecastResult with mean fold metrics.
        """
        from sklearn.metrics import mean_squared_error, r2_score

        if returns is None:
            returns = df["Close"].pct_change()

        X_raw = self.generate_features(df, returns)
        y_raw = self.compute_target(returns, self.horizon, annualize=True)
        mask = X_raw.notna().all(axis=1) & y_raw.notna()
        X = X_raw.loc[mask]
        y = y_raw.loc[mask]

        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float64)
        y_arr = y.values.astype(np.float64)

        from .purged_cv import PurgedKFold

        cv = PurgedKFold(
            n_splits=n_splits,
            pct_embargo=pct_embargo,
            label_span=self.horizon,
        )

        from catboost import CatBoostRegressor

        fold_metrics = []
        all_preds = np.zeros_like(y_arr)

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            model = CatBoostRegressor(
                iterations=self.n_estimators,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_seed,
                verbose=False,
                loss_function="Huber:delta=1.0",
                task_type="CPU",
            )
            model.fit(X_arr[train_idx], y_arr[train_idx])
            preds = model.predict(X_arr[test_idx])
            all_preds[test_idx] = preds

            rmse = float(np.sqrt(mean_squared_error(y_arr[test_idx], preds)))
            r2 = float(r2_score(y_arr[test_idx], preds))
            dir_acc = (
                float(np.mean(np.sign(np.diff(y_arr[test_idx])) == np.sign(np.diff(preds))))
                if len(test_idx) > 1
                else float("nan")
            )

            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "rmse": round(rmse, 6),
                    "r2": round(r2, 4),
                    "direction_accuracy": round(dir_acc, 4),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )

        mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
        mean_r2 = float(np.mean([m["r2"] for m in fold_metrics]))
        mean_dir = float(np.nanmean([m["direction_accuracy"] for m in fold_metrics]))
        mae_val = float(np.mean(np.abs(y_arr - all_preds)))

        # Train final model on all data
        self._model = CatBoostRegressor(
            iterations=self.n_estimators,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=False,
            loss_function="Huber:delta=1.0",
            task_type="CPU",
        )
        self._model.fit(X_arr, y_arr)
        self._target_mean = float(y_arr.mean())
        self._target_std = float(y_arr.std())

        feat_imp = self._get_feature_importance()

        return VolForecastResult(
            horizon=self.horizon,
            rmse=mean_rmse,
            mae=mae_val,
            r2=mean_r2,
            direction_accuracy=mean_dir,
            feature_importance=feat_imp,
            n_train=len(X),
            fold_metrics=fold_metrics,
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        if self._model is None or self._feature_names is None:
            return {}
        importance = self._model.get_feature_importance()
        return dict(
            sorted(
                zip(self._feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "target_mean": self._target_mean,
                "target_std": self._target_std,
                "config": {
                    "horizon": self.horizon,
                    "n_estimators": self.n_estimators,
                    "depth": self.depth,
                    "learning_rate": self.learning_rate,
                    "l2_leaf_reg": self.l2_leaf_reg,
                    "random_seed": self.random_seed,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "VolatilityForecaster":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]
        instance = cls(**config)
        instance._model = data["model"]
        instance._feature_names = data.get("feature_names")
        instance._target_mean = data.get("target_mean", 0.0)
        instance._target_std = data.get("target_std", 1.0)
        return instance

    def _check_fitted(self) -> None:
        if self._model is None:
            raise ValueError("Model not trained. Call fit() or cross_validate() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._model is not None else "untrained"
        return f"VolatilityForecaster(h={self.horizon}, status={status})"
