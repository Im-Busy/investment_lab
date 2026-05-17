"""
D10: Rolling ARIMA + GARCH Hybrid Forecaster.

Models both the conditional mean (ARIMA) and conditional variance (GARCH)
of financial returns. ARIMA captures momentum and mean-reversion dynamics,
while GARCH captures volatility clustering. Together they provide a complete
description of the conditional return distribution.

Components:
  ARIMAForecaster: Auto-ARIMA with rolling window, AIC-optimal order selection.
  ARIMAGARCHForecaster: Combined ARIMA mean + GARCH variance forecasts.
    Uses standardized residuals from ARIMA to feed GARCH, providing
    conditional volatility that accounts for mean dynamics.

Usage:
    >>> forecaster = ARIMAGARCHForecaster(window=504)
    >>> forecaster.fit(returns)
    >>> mu, sigma = forecaster.forecast(horizon=5)
    >>> result = forecaster.evaluate(horizon=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

logger = logging.getLogger(__name__)

ANNUALIZATION = np.sqrt(252)


@dataclass
class ARIMAForecast:
    mean: float
    std_error: float
    ci_lower: float
    ci_upper: float
    order: Tuple[int, int, int]
    aic: float


@dataclass
class ARIMAGARCHResult:
    horizon: int
    rmse_mean: float
    rmse_vol: float
    direction_accuracy: float
    coverage_95: float
    in_sample_n: int
    out_of_sample_n: int
    mean_mu: float
    mean_sigma: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "horizon": self.horizon,
            "rmse_mean": round(self.rmse_mean, 6),
            "rmse_vol": round(self.rmse_vol, 6),
            "direction_accuracy": round(self.direction_accuracy, 4),
            "coverage_95": round(self.coverage_95, 4),
            "in_sample_n": self.in_sample_n,
            "out_of_sample_n": self.out_of_sample_n,
            "mean_mu": round(self.mean_mu, 6),
            "mean_sigma": round(self.mean_sigma, 6),
        }


class ARIMAForecaster:
    """Rolling-window ARIMA forecaster with auto order selection.

    Searches over (p,d,q) space using AIC. Supports rolling and expanding
    window refitting for adaptive forecasts.

    Args:
        max_p: Maximum AR order (default 5).
        max_q: Maximum MA order (default 5).
        max_d: Maximum differencing order (default 1).
        window: Training window size in bars (0 = expanding).
        retrain_every: Re-fit every N bars (0 = every bar).
        maxiter: Maximum optimizer iterations.
    """

    def __init__(
        self,
        max_p: int = 2,
        max_q: int = 1,
        max_d: int = 0,
        window: int = 504,
        retrain_every: int = 63,
        maxiter: int = 50,
    ):
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.window = window
        self.retrain_every = retrain_every
        self.maxiter = maxiter
        self._returns: Optional[pd.Series] = None
        self._best_order: Tuple[int, int, int] = (1, 0, 0)
        self._last_model = None
        self._last_fit_end: int = 0

    def fit(self, returns: pd.Series) -> ARIMAForecaster:
        """Fit ARIMA and auto-select best order via AIC.

        Args:
            returns: Daily log or simple returns.

        Returns:
            Self for chaining.
        """
        self._returns = returns.dropna().copy()
        if self.window > 0 and len(self._returns) < self.window:
            raise ValueError(f"Need at least {self.window} returns, got {len(self._returns)}")
        self._select_order(self._returns.values)
        return self

    def _select_order(self, series: np.ndarray) -> None:
        """Find best (p,d,q) via AIC on a limited search space."""
        best_aic = float("inf")
        best_order = (0, 0, 0)
        orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (2, 0, 1)]
        for order in orders:
            try:
                model = ARIMA(series, order=order)
                result = model.fit(method_kwargs={"maxiter": self.maxiter})
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = order
            except Exception:
                continue
        if best_order == (0, 0, 0):
            best_order = (1, 0, 0)
        self._best_order = best_order
        logger.debug("ARIMA order: %s, AIC=%.1f", best_order, best_aic)

    def forecast_single(
        self,
        series: np.ndarray,
        horizon: int = 1,
        auto_order: bool = False,
    ) -> ARIMAForecast:
        """Fit on all data and produce a point forecast.

        Args:
            series: Return series.
            horizon: Forecast steps ahead.
            auto_order: Re-select order via AIC (False = use pre-selected order).

        Returns:
            ARIMAForecast with mean and confidence intervals.
        """
        order = self._best_order
        if auto_order and len(series) > 20:
            self._select_order(series)
            order = self._best_order
        try:
            model = ARIMA(series, order=order)
            result = model.fit(method_kwargs={"maxiter": self.maxiter})
            fc = result.get_forecast(steps=horizon)
            mean = float(fc.predicted_mean[-1])
            se = float(fc.se_mean[-1])
            ci = fc.conf_int(alpha=0.05)
            return ARIMAForecast(
                mean=mean,
                std_error=se,
                ci_lower=float(ci[-1, 0]),
                ci_upper=float(ci[-1, 1]),
                order=order,
                aic=float(result.aic),
            )
        except Exception as e:
            logger.debug("ARIMA forecast failed: %s", e)
            return ARIMAForecast(
                mean=float(np.mean(series[-20:])),
                std_error=float(np.std(series)),
                ci_lower=float(np.mean(series[-20:]) - 2 * np.std(series)),
                ci_upper=float(np.mean(series[-20:]) + 2 * np.std(series)),
                order=(0, 0, 0),
                aic=float("inf"),
            )

    def forecast_rolling(
        self,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Produce rolling out-of-sample forecasts (cached between retrains).

        Args:
            horizon: Forecast steps ahead.

        Returns:
            DataFrame with columns: mu, resid.
        """
        if self._returns is None:
            raise ValueError("Call fit() first")

        series = self._returns.values
        n = len(series)
        mu = np.full(n, np.nan, dtype=np.float64)
        resid = np.full(n, np.nan, dtype=np.float64)
        last_fc: Optional[ARIMAForecast] = None

        for i in range(self.window, n):
            if self.retrain_every > 0 and (i - self.window) % self.retrain_every == 0:
                self._select_order(series[:i])
                last_fc = None

            if last_fc is None:
                last_fc = self.forecast_single(series[:i], horizon=horizon, auto_order=False)
            mu[i] = last_fc.mean
            if i < n:
                resid[i] = series[i] - last_fc.mean

        return pd.DataFrame(
            {"mu": mu, "resid": resid},
            index=self._returns.index,
        )

    @property
    def best_order(self) -> Tuple[int, int, int]:
        return self._best_order


class ARIMAGARCHForecaster:
    """Combined ARIMA + GARCH rolling forecaster.

    Two-stage model:
      1. ARIMA models the conditional mean (returns).
      2. GARCH models the conditional variance of ARIMA residuals.

    Provides both mean and volatility forecasts with confidence bands.

    Args:
        arima_window: ARIMA training window (bars).
        garch_window: GARCH training window (bars, for residual GARCH).
        retrain_every: Re-fit frequency (bars).
        garch_model: GARCH variant ("garch", "egarch", "gjr-garch").
        max_p, max_q, max_d: ARIMA order search limits.
    """

    def __init__(
        self,
        arima_window: int = 504,
        garch_window: int = 252,
        retrain_every: int = 63,
        garch_model: str = "egarch",
        max_p: int = 2,
        max_q: int = 1,
        max_d: int = 0,
    ):
        self.arima_window = arima_window
        self.garch_window = garch_window
        self.retrain_every = retrain_every
        self.garch_model = garch_model
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self._returns: Optional[pd.Series] = None
        self._arima_forecaster: Optional[ARIMAForecaster] = None
        self._garch_fitted = False
        self._garch_params: Dict = {}

    def fit(self, returns: pd.Series) -> ARIMAGARCHForecaster:
        """Fit ARIMA on mean and GARCH on residuals.

        Args:
            returns: Daily return series.

        Returns:
            Self for chaining.
        """
        self._returns = returns.dropna().copy()
        min_window = max(self.arima_window, self.garch_window)
        if len(self._returns) < min_window:
            raise ValueError(f"Need at least {min_window} returns, got {len(self._returns)}")

        self._arima_forecaster = ARIMAForecaster(
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            window=self.arima_window,
            retrain_every=self.retrain_every,
        )
        self._arima_forecaster.fit(self._returns)

        # Fit GARCH on residuals from full-sample ARIMA
        arima_fc = self._arima_forecaster.forecast_rolling(horizon=1)
        resid = arima_fc["resid"].dropna().values
        if len(resid) > self.garch_window:
            self._fit_garch_residuals(resid[-self.garch_window :])
        return self

    def _fit_garch_residuals(self, resid: np.ndarray) -> None:
        """Fit GARCH on standardized ARIMA residuals."""
        scaled = resid * 100.0
        try:
            if self.garch_model == "egarch":
                am = arch_model(scaled, vol="EGARCH", p=1, q=1, dist="normal")
            elif self.garch_model == "gjr-garch":
                am = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist="normal")
            else:
                am = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            self._garch_params = {k: float(v) for k, v in res.params.items()}
            self._garch_fitted = bool(res.convergence_flag == 0)
            self._last_garch_model = res
        except Exception:
            try:
                am = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
                res = am.fit(disp="off")
                self._garch_params = {k: float(v) for k, v in res.params.items()}
                self._garch_fitted = bool(res.convergence_flag == 0)
                self._last_garch_model = res
            except Exception:
                self._garch_fitted = False
                self._last_garch_model = None

    def forecast(
        self,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Rolling combined ARIMA mean + GARCH volatility forecasts.

        Uses cached model between retrain intervals for performance.

        Args:
            horizon: Forecast steps ahead.

        Returns:
            DataFrame with mu (mean return), sigma (annualized vol), and bounds.
        """
        if self._returns is None or self._arima_forecaster is None:
            raise ValueError("Call fit() first")

        returns = self._returns.values
        n = len(returns)
        mu = np.full(n, np.nan, dtype=np.float64)
        sigma = np.full(n, np.nan, dtype=np.float64)
        resid = np.full(n, np.nan, dtype=np.float64)

        min_window = max(self.arima_window, self.garch_window)
        last_arima_fc: Optional[ARIMAForecast] = None
        last_garch_sigma: Optional[float] = None

        for i in range(min_window, n):
            refit = self.retrain_every > 0 and (i - min_window) % self.retrain_every == 0

            if refit or last_arima_fc is None:
                train = returns[:i]
                last_arima_fc = self._arima_forecaster.forecast_single(
                    train, horizon=horizon, auto_order=False
                )
            mu[i] = last_arima_fc.mean

            if refit or last_garch_sigma is None:
                if self._garch_fitted and self._last_garch_model is not None:
                    try:
                        fc = self._last_garch_model.forecast(
                            horizon=horizon, reindex=False, method="simulation"
                        )
                        var_fc = fc.variance.values[-1]
                        last_garch_sigma = np.sqrt(np.sum(var_fc)) / 100.0
                    except Exception:
                        last_garch_sigma = None
                else:
                    last_garch_sigma = None

            if last_garch_sigma is not None:
                sigma[i] = last_garch_sigma * ANNUALIZATION

            if refit:
                residual_slice = resid[~np.isnan(resid)][-self.garch_window :]
                if len(residual_slice) > 50:
                    self._fit_garch_residuals(residual_slice)
                self._arima_forecaster._select_order(returns[:i])

            if i < n:
                resid[i] = returns[i] - mu[i]

        idx = self._returns.index
        df = pd.DataFrame({"mu": mu, "sigma": sigma}, index=idx)
        df["ub_95"] = df["mu"] + 1.96 * df["sigma"] / ANNUALIZATION
        df["lb_95"] = df["mu"] - 1.96 * df["sigma"] / ANNUALIZATION
        return df

    def forecast_next(
        self,
        returns: Optional[pd.Series] = None,
        horizon: int = 1,
    ) -> ARIMAForecast:
        """Single-step forecast using recent data.

        Args:
            returns: Optional recent returns (uses training data if None).
            horizon: Forecast horizon.

        Returns:
            ARIMAForecast with mean and confidence intervals.
        """
        data = returns if returns is not None else self._returns
        if data is None:
            raise ValueError("No return data available")
        values = data.dropna().values
        if self._arima_forecaster is None:
            self._arima_forecaster = ARIMAForecaster(
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                window=self.arima_window,
            )
        return self._arima_forecaster.forecast_single(values, horizon=horizon)

    def evaluate(
        self,
        horizon: int = 5,
    ) -> ARIMAGARCHResult:
        """Evaluate combined forecast accuracy.

        Args:
            horizon: Evaluation forecast horizon.

        Returns:
            ARIMAGARCHResult with mean RMSE, vol RMSE, dir accuracy, coverage.
        """
        fc = self.forecast(horizon=horizon)
        returns = self._returns

        if returns is None:
            raise ValueError("Call fit() first")

        realized = returns.shift(-horizon).rolling(horizon).sum()
        realized_vol = returns.rolling(horizon).std().shift(-horizon + 1) * ANNUALIZATION

        mask = fc["mu"].notna() & realized.notna() & realized_vol.notna()
        y_pred = fc["mu"][mask].values
        y_true = realized[mask].values
        vol_pred = fc["sigma"][mask].values
        vol_true = realized_vol[mask].values

        n = len(y_true)
        if n < 10:
            return ARIMAGARCHResult(
                horizon=horizon,
                rmse_mean=np.nan,
                rmse_vol=np.nan,
                direction_accuracy=np.nan,
                coverage_95=np.nan,
                in_sample_n=len(returns),
                out_of_sample_n=n,
                mean_mu=np.nan,
                mean_sigma=np.nan,
                notes=["Insufficient out-of-sample data"],
            )

        rmse_mean = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        rmse_vol = float(np.sqrt(np.mean((vol_true - vol_pred) ** 2)))
        dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        in_band = (y_true >= fc["lb_95"][mask].values) & (y_true <= fc["ub_95"][mask].values)
        coverage = float(np.mean(in_band))

        notes = []
        if coverage < 0.80:
            notes.append("Coverage < 80%: model underconfident or mis-specified")
        if coverage > 0.98:
            notes.append("Coverage > 98%: tolerance too wide")

        mu_pred = fc["mu"].mean()
        if mu_pred > 0.01:
            notes.append(f"Persistent positive bias in forecasts ({mu_pred:.4f})")

        return ARIMAGARCHResult(
            horizon=horizon,
            rmse_mean=rmse_mean,
            rmse_vol=rmse_vol,
            direction_accuracy=dir_acc,
            coverage_95=coverage,
            in_sample_n=len(returns),
            out_of_sample_n=n,
            mean_mu=float(np.mean(y_pred)),
            mean_sigma=float(np.mean(vol_pred)),
            notes=notes,
        )

    @property
    def garch_params(self) -> Dict:
        return self._garch_params

    @property
    def arima_order(self) -> Optional[Tuple[int, int, int]]:
        if self._arima_forecaster is not None:
            return self._arima_forecaster.best_order
        return None
