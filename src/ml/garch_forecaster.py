"""
Q3: GARCH/EGARCH Volatility Forecasting.

Forward-looking volatility forecasts using GARCH-family models from the
`arch` library. Complements the CatBoost VolatilityForecaster by providing
a well-understood parametric baseline rooted in volatility clustering —
the single most robust stylized fact in financial returns.

Models:
  GARCH(1,1): Standard generalized autoregressive conditional heteroskedasticity.
    Volatility today depends on past shocks and past volatility. Captures
    volatility clustering (large moves cluster together).
  EGARCH(1,1): Exponential GARCH. Adds leverage effect — negative returns
    increase future volatility more than positive returns of equal magnitude.
  GJR-GARCH(1,1): Glosten-Jagannathan-Runkle GARCH. Alternative leverage effect
    model with indicator function for negative shocks.

Forecasting:
  - Next-bar (1-day) conditional volatility forecast
  - N-day cumulative volatility forecast
  - Rolling-window retraining for adaptive forecasts

Usage:
    >>> forecaster = GARCHForecaster(model="egarch")
    >>> forecaster.fit(returns)
    >>> vol_1d = forecaster.forecast(horizon=1)
    >>> vol_20d = forecaster.forecast(horizon=20)
    >>> result = forecaster.evaluate(returns, horizon=5)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

ANNUALIZATION = np.sqrt(252)

SUPPORTED_MODELS = ("garch", "egarch", "gjr-garch")


@dataclass
class GARCHForecastResult:
    model: str
    horizon: int
    rmse: float
    mae: float
    direction_accuracy: float
    in_sample_n: int
    out_of_sample_n: int
    mean_forecast: float
    std_forecast: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "horizon": self.horizon,
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "direction_accuracy": round(self.direction_accuracy, 4),
            "in_sample_n": self.in_sample_n,
            "out_of_sample_n": self.out_of_sample_n,
            "mean_forecast": round(self.mean_forecast, 6),
            "std_forecast": round(self.std_forecast, 6),
        }

    def __repr__(self) -> str:
        return (
            f"GARCHForecastResult(m={self.model}, h={self.horizon}, "
            f"rmse={self.rmse:.4f}, dir_acc={self.direction_accuracy:.3f})"
        )


class GARCHForecaster:
    """Forward-looking volatility forecaster using GARCH-family models.

    Fits GARCH/EGARCH/GJR-GARCH on a rolling window and produces N-step
    volatility forecasts. Provides a clean alternative to the reactive
    ATR-based volatility regime currently used in the system.

    Args:
        model: GARCH variant (garch, egarch, gjr-garch).
        window: Training window size (bars).
        retrain_every: Re-fit model every N bars (0 = no retrain).
        annualize: Whether to annualize forecasts.
        dist: Error distribution (normal, t, ged, skewt).
    """

    def __init__(
        self,
        model: str = "egarch",
        window: int = 252,
        retrain_every: int = 63,
        annualize: bool = True,
        dist: str = "normal",
    ):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {SUPPORTED_MODELS}, got {model}")
        self.model = model
        self.window = window
        self.retrain_every = retrain_every
        self.annualize = annualize
        self.dist = dist
        self._returns: Optional[pd.Series] = None
        self._last_model = None
        self._last_fit_end: int = 0
        self._params: Dict = {}
        self._converged: bool = False

    def fit(self, returns: pd.Series) -> GARCHForecaster:
        """Fit GARCH model on the full return series.

        Args:
            returns: Daily log or simple returns.

        Returns:
            Self for chaining.
        """
        self._returns = returns.dropna().copy()
        if len(self._returns) < self.window:
            raise ValueError(f"Need at least {self.window} returns, got {len(self._returns)}")

        self._fit_model(self._returns)
        return self

    def _fit_model(self, returns: pd.Series) -> None:
        """Internal: fit GARCH variant on a return series."""
        scaled = returns * 100.0
        try:
            if self.model == "egarch":
                am = arch_model(scaled, vol="EGARCH", p=1, q=1, dist=self.dist)
            elif self.model == "gjr-garch":
                am = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist=self.dist)
            else:
                am = arch_model(scaled, vol="GARCH", p=1, q=1, dist=self.dist)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", options={"maxiter": 2000})
            self._last_model = res
            self._params = {k: float(v) for k, v in res.params.items()}
            self._converged = bool(res.convergence_flag == 0)
            if not self._converged:
                self._fallback_fit(scaled)
        except Exception as e:
            logger.debug("GARCH fit failed: %s, falling back", e)
            self._fallback_fit(scaled)

    def _fallback_fit(self, scaled: np.ndarray) -> None:
        """Fall back to standard GARCH(1,1) if model-specific fit fails."""
        try:
            am = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = am.fit(disp="off", options={"maxiter": 2000})
            self._last_model = res
            self._params = {k: float(v) for k, v in res.params.items()}
            self._converged = bool(res.convergence_flag == 0)
        except Exception:
            self._last_model = None
            self._converged = False

    def forecast(self, horizon: int = 1) -> pd.Series:
        """Generate horizon-step volatility forecasts.

        Uses rolling window retraining if configured.

        Args:
            horizon: Forecast horizon in bars.

        Returns:
            Series of forecast annualized volatility.
        """
        if self._returns is None:
            raise ValueError("Call fit() first")

        returns = self._returns.values
        n = len(returns)
        forecasts = np.full(n, np.nan, dtype=np.float64)

        for i in range(self.window, n):
            if self.retrain_every > 0 and (i - self.window) % self.retrain_every == 0:
                self._fit_model(pd.Series(returns[:i], index=self._returns.index[:i]))

            if self._last_model is None or not self._converged:
                continue

            try:
                fc = self._last_model.forecast(horizon=horizon, reindex=False, method="simulation")
                var_fc = fc.variance.values[-1]
                if horizon == 1:
                    vol = np.sqrt(var_fc[0]) / 100.0
                else:
                    cum_var = np.sum(var_fc)
                    vol = np.sqrt(cum_var) / 100.0
                if self.annualize:
                    vol *= np.sqrt(ANNUALIZATION**2 / horizon)
                forecasts[i] = vol
            except Exception:
                pass

        return pd.Series(forecasts, index=self._returns.index, name=f"{self.model}_vol_{horizon}d")

    def forecast_single(self, returns: pd.Series, horizon: int = 1) -> float:
        """Fit and forecast a single volatility value."""
        scaled = returns.values * 100.0
        if self.model == "egarch":
            am = arch_model(scaled, vol="EGARCH", p=1, q=1, dist=self.dist)
        elif self.model == "gjr-garch":
            am = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist=self.dist)
        else:
            am = arch_model(scaled, vol="GARCH", p=1, q=1, dist=self.dist)
        res = am.fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False, method="simulation")
        var_fc = fc.variance.values[-1]
        vol = np.sqrt(np.sum(var_fc)) / 100.0
        if self.annualize:
            vol *= np.sqrt(ANNUALIZATION**2 / horizon)
        return float(vol)

    def evaluate(
        self,
        returns: Optional[pd.Series] = None,
        horizon: int = 5,
    ) -> GARCHForecastResult:
        """Evaluate GARCH forecast accuracy against realized volatility.

        Args:
            returns: Optional return series (uses training data if None).
            horizon: Forecast horizon for evaluation.

        Returns:
            GARCHForecastResult with metrics.
        """
        if returns is None:
            if self._returns is None:
                raise ValueError("Call fit() first or provide returns")
            returns = self._returns
        else:
            self._returns = returns.dropna().copy()

        forecasts = self.forecast(horizon=horizon)
        realized = returns.rolling(horizon).std().shift(-horizon + 1) * ANNUALIZATION

        mask = forecasts.notna() & realized.notna()
        y_pred = forecasts[mask].values
        y_true = realized[mask].values

        if len(y_true) < 10:
            return GARCHForecastResult(
                model=self.model,
                horizon=horizon,
                rmse=np.nan,
                mae=np.nan,
                direction_accuracy=np.nan,
                in_sample_n=self.window,
                out_of_sample_n=len(y_true),
                mean_forecast=np.nan,
                std_forecast=np.nan,
                notes=["Insufficient data for evaluation"],
            )

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        dir_acc = (
            float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
            if len(y_true) > 1
            else float("nan")
        )

        return GARCHForecastResult(
            model=self.model,
            horizon=horizon,
            rmse=rmse,
            mae=mae,
            direction_accuracy=dir_acc,
            in_sample_n=self.window,
            out_of_sample_n=len(y_true),
            mean_forecast=float(np.mean(y_pred)),
            std_forecast=float(np.std(y_pred)),
        )

    def compare_models(
        self,
        returns: pd.Series,
        horizon: int = 5,
        models: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple GARCH variants on the same data.

        Args:
            returns: Return series.
            horizon: Forecast horizon.
            models: List of model names (default: all three).

        Returns:
            DataFrame with one row per model.
        """
        models = models or list(SUPPORTED_MODELS)
        results = []
        for m in models:
            fc = GARCHForecaster(
                model=m,
                window=self.window,
                retrain_every=self.retrain_every,
                annualize=self.annualize,
                dist=self.dist,
            )
            fc._returns = returns.dropna().copy()
            if len(fc._returns) >= self.window:
                fc._fit_model(fc._returns)
            r = fc.evaluate(returns=returns, horizon=horizon)
            results.append(r.to_dict())
        return pd.DataFrame(results)

    @property
    def params(self) -> Dict:
        return self._params

    @property
    def converged(self) -> bool:
        return self._converged

    def __repr__(self) -> str:
        return f"GARCHForecaster(m={self.model}, w={self.window})"
