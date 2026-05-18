"""
D9: Kalman Filter Dynamic Regression — Time-Varying Hedge Ratios.

Unlike static OLS or rolling-window regression, a Kalman filter allows the
hedge ratio (beta) between two assets to evolve continuously. This captures
structural shifts in correlations that fixed-window methods miss.

Models:
  KalmanHedgeEstimator: Online state-space estimation of time-varying beta.
    State = [alpha (intercept), beta (hedge ratio)].
    Observation = y_t = alpha_t + beta_t * x_t + ε_t.
    State transition = random walk: α_{t+1} = α_t + η_α, β_{t+1} = β_t + η_β.

  KalmanHedgePair: Two-asset hedge ratio tracker with P&L simulation.
    Tracks spread = y - beta * x over time. Computes hedge effectiveness
    via variance reduction vs unhedged position.

  Comparison: Kalman hedge vs rolling OLS hedge vs static OLS hedge.

Usage:
    >>> est = KalmanHedgeEstimator()
    >>> betas, alphas = est.fit(y_prices, x_prices)
    >>> pair = KalmanHedgePair(y_prices, x_prices)
    >>> pair_corr = pair.hedge_correlation()
    >>> v_reduction = pair.variance_reduction()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KalmanHedgeResult:
    alpha: np.ndarray
    beta: np.ndarray
    spread: np.ndarray
    pred_error: np.ndarray
    variance_reduction: float
    hedge_effectiveness: float
    terminal_beta: float
    beta_volatility: float

    @property
    def mean_beta(self) -> float:
        return float(np.nanmean(self.beta))

    @property
    def beta_range(self) -> Tuple[float, float]:
        return float(np.nanmin(self.beta)), float(np.nanmax(self.beta))

    def to_dict(self) -> dict:
        return {
            "mean_beta": round(self.mean_beta, 4),
            "terminal_beta": round(self.terminal_beta, 4),
            "beta_vol": round(self.beta_volatility, 4),
            "beta_min": round(self.beta_range[0], 4),
            "beta_max": round(self.beta_range[1], 4),
            "variance_reduction": round(self.variance_reduction * 100, 2),
            "hedge_effectiveness": round(self.hedge_effectiveness, 4),
        }


class KalmanHedgeEstimator:
    """Kalman filter for time-varying hedge ratio estimation.

    Models the relationship y_t = alpha_t + beta_t * x_t + ε_t where
    [alpha_t, beta_t] follow a random walk. Suitable for pairs trading
    and portfolio hedging where the relationship between assets drifts.

    Args:
        delta: Discount factor for adaptive noise estimation (0 < δ ≤ 1).
            Lower = more adaptive. Default 0.98.
        init_beta: Initial hedge ratio guess. Default 1.0.
        init_alpha: Initial intercept guess. Default 0.0.
    """

    def __init__(
        self,
        delta: float = 0.98,
        init_beta: float = 1.0,
        init_alpha: float = 0.0,
    ):
        self.delta = delta
        self.init_beta = init_beta
        self.init_alpha = init_alpha
        self._result: Optional[KalmanHedgeResult] = None

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
    ) -> KalmanHedgeResult:
        """Estimate time-varying hedge ratio via Kalman filter.

        Args:
            y: Dependent variable returns or prices (e.g., asset to hedge).
            x: Independent variable returns or prices (e.g., hedging instrument).

        Returns:
            KalmanHedgeResult with alpha/beta time series and diagnostics.
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = min(len(y), len(x))

        if n < 10:
            return _empty_hedge_result(n)

        y_vals = y[:n]
        x_vals = x[:n]

        alpha_arr = np.full(n, np.nan)
        beta_arr = np.full(n, np.nan)
        spread_arr = np.full(n, np.nan)
        pred_err = np.full(n, np.nan)

        state = np.array([self.init_alpha, self.init_beta])
        P = np.eye(2) * 0.1
        R = np.var(y_vals[: min(20, n)]) * 0.1 if n >= 5 else 1.0
        Q = np.eye(2) * 1e-4

        for t in range(n):
            obs = np.array([1.0, x_vals[t]])

            y_pred = float(obs @ state)
            if not np.isnan(y_vals[t]):
                v = y_vals[t] - y_pred

                S = float(obs @ P @ obs + R)
                K = P @ obs / max(S, 1e-12)

                state = state + K * v
                P = (np.eye(2) - np.outer(K, obs)) @ P

                R = self.delta * R + (1 - self.delta) * v**2
            else:
                v = np.nan

            P = P + Q

            alpha_arr[t] = state[0]
            beta_arr[t] = state[1]
            if not np.isnan(y_vals[t]):
                spread_arr[t] = y_vals[t] - state[1] * x_vals[t]
            pred_err[t] = v if not np.isnan(v) else np.nan

        hedge_effectiveness = _compute_hedge_effectiveness(y_vals, spread_arr)
        var_reduction = _variance_reduction(y_vals, spread_arr)

        result = KalmanHedgeResult(
            alpha=alpha_arr,
            beta=beta_arr,
            spread=spread_arr,
            pred_error=pred_err,
            variance_reduction=var_reduction,
            hedge_effectiveness=hedge_effectiveness,
            terminal_beta=float(beta_arr[-1]) if n > 0 and not np.isnan(beta_arr[-1]) else 1.0,
            beta_volatility=float(np.nanstd(beta_arr)) if n > 0 else 0.0,
        )
        self._result = result
        return result

    def update(
        self,
        y_new: float,
        x_new: float,
    ) -> Tuple[float, float]:
        """Online single-step Kalman update.

        Args:
            y_new: New observation for dependent variable.
            x_new: New observation for independent variable.

        Returns:
            (updated_alpha, updated_beta).
        """
        raise NotImplementedError(
            "Online update not yet supported; use fit() for batch estimation."
        )

    @property
    def result(self) -> Optional[KalmanHedgeResult]:
        return self._result


class KalmanHedgePair:
    """Two-asset hedge ratio tracker with full diagnostics.

    Tracks the spread = y - beta * x between two assets over time
    using a Kalman filter. Computes hedge effectiveness, correlation,
    and variance reduction vs unhedged positions.

    Args:
        delta: Discount factor for Kalman estimator.
        init_beta: Initial hedge ratio.
    """

    def __init__(self, delta: float = 0.98, init_beta: float = 1.0):
        self.estimator = KalmanHedgeEstimator(delta=delta, init_beta=init_beta)
        self._y: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None
        self._result: Optional[KalmanHedgeResult] = None

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
    ) -> KalmanHedgeResult:
        """Estimate Kalman hedge ratio for pair.

        Args:
            y: Target asset prices or returns.
            x: Hedge instrument prices or returns.

        Returns:
            KalmanHedgeResult.
        """
        self._y = np.asarray(y, dtype=float)
        self._x = np.asarray(x, dtype=float)
        self._result = self.estimator.fit(self._y, self._x)
        return self._result

    def hedge_correlation(self) -> float:
        """Pearson correlation between target and hedge instrument."""
        if self._y is None or self._x is None:
            return float("nan")
        valid = ~np.isnan(self._y) & ~np.isnan(self._x)
        if valid.sum() < 5:
            return float("nan")
        return float(np.corrcoef(self._y[valid], self._x[valid])[0, 1])

    def variance_reduction(self) -> float:
        """Reduction in variance from Kalman hedging vs unhedged.

        Returns:
            Fractional variance reduction (0 = no reduction, 1 = perfect).
        """
        if self._result is None:
            return 0.0
        return self._result.variance_reduction

    def compare_methods(
        self,
        lookback: int = 60,
    ) -> pd.DataFrame:
        """Compare Kalman vs rolling OLS vs static OLS hedge ratios.

        Args:
            lookback: Rolling OLS window size.

        Returns:
            DataFrame with time-varying betas from all three methods.
        """
        if self._y is None or self._x is None:
            return pd.DataFrame()

        n = min(len(self._y), len(self._x))
        y = self._y[:n]
        x = self._x[:n]

        self.fit(y, x)
        kalman_betas = self._result.beta if self._result is not None else np.full(n, np.nan)

        rolling_betas = np.full(n, np.nan)
        for t in range(lookback, n):
            window_y = y[t - lookback : t]
            window_x = x[t - lookback : t]
            valid = ~np.isnan(window_y) & ~np.isnan(window_x)
            if valid.sum() < 10:
                continue
            yy = window_y[valid]
            xx = window_x[valid]
            X = np.column_stack([np.ones(len(xx)), xx])
            try:
                rolling_betas[t] = np.linalg.lstsq(X, yy, rcond=None)[0][1]
            except np.linalg.LinAlgError:
                pass

        valid_all = ~np.isnan(y) & ~np.isnan(x)
        static_beta = 1.0
        if valid_all.sum() >= 10:
            yy = y[valid_all]
            xx = x[valid_all]
            X = np.column_stack([np.ones(len(xx)), xx])
            try:
                static_beta = np.linalg.lstsq(X, yy, rcond=None)[0][1]
            except np.linalg.LinAlgError:
                pass

        return pd.DataFrame(
            {
                "kalman": kalman_betas,
                f"rolling_{lookback}": rolling_betas,
                "static": np.full(n, static_beta),
            }
        )

    def hedge_pnl(
        self,
        prices_y: np.ndarray,
        prices_x: np.ndarray,
    ) -> pd.DataFrame:
        """Compute hedged P&L from price series.

        Args:
            prices_y: Price series for target asset.
            prices_x: Price series for hedge instrument.

        Returns:
            DataFrame with unhedged, hedge_ratio, hedged_pnl columns.
        """
        rets_y = np.diff(prices_y) / prices_y[:-1]
        rets_x = np.diff(prices_x) / prices_x[:-1]
        n = len(rets_y)

        self.fit(rets_y, rets_x)
        betas = self._result.beta if self._result is not None else np.ones(n)

        unhedged = np.cumsum(rets_y)
        hedged = np.cumsum(rets_y - betas * rets_x)

        return pd.DataFrame(
            {
                "unhedged_cum": np.insert(unhedged, 0, 0),
                "hedge_ratio": np.insert(betas, 0, betas[0]),
                "hedged_cum": np.insert(hedged, 0, 0),
            }
        )

    def fit_dataframe(
        self,
        df_y: pd.Series,
        df_x: pd.Series,
    ) -> pd.DataFrame:
        """Fit on pandas Series and return results as DataFrame.

        Args:
            df_y: Target time series.
            df_x: Hedge instrument time series.

        Returns:
            DataFrame with alpha, beta, spread, pred_error columns.
        """
        y = df_y.values
        x = df_x.values
        self.fit(y, x)
        if self._result is None:
            return pd.DataFrame()

        n = min(len(y), len(x))
        idx = df_y.index[:n]
        return pd.DataFrame(
            {
                "alpha": self._result.alpha[:n],
                "beta": self._result.beta[:n],
                "spread": self._result.spread[:n],
                "pred_error": self._result.pred_error[:n],
            },
            index=idx,
        )

    @property
    def result(self) -> Optional[KalmanHedgeResult]:
        return self._result


class PortfolioHedgeEstimator:
    """Multi-asset portfolio hedge ratio estimation via Kalman filter.

    Estimates time-varying hedge ratios for a portfolio of N assets
    against a single hedge instrument. Uses parallel Kalman filters.

    Args:
        delta: Kalman discount factor.
    """

    def __init__(self, delta: float = 0.98):
        self.delta = delta
        self._estimators: List[KalmanHedgeEstimator] = []
        self._results: List[KalmanHedgeResult] = []

    def fit(
        self,
        portfolio_returns: np.ndarray,
        hedge_returns: np.ndarray,
    ) -> pd.DataFrame:
        """Estimate hedge ratios for each portfolio asset.

        Args:
            portfolio_returns: (T, N) array of asset returns.
            hedge_returns: (T,) array of hedge instrument returns.

        Returns:
            DataFrame with one beta column per portfolio asset.
        """
        portfolio_returns = np.atleast_2d(portfolio_returns)
        if portfolio_returns.ndim == 1:
            portfolio_returns = portfolio_returns.reshape(-1, 1)
        T, N = portfolio_returns.shape
        hedge_returns = np.asarray(hedge_returns, dtype=float)[:T]

        betas = np.full((T, N), np.nan)
        self._estimators = []
        self._results = []

        for j in range(N):
            est = KalmanHedgeEstimator(delta=self.delta)
            result = est.fit(portfolio_returns[:, j], hedge_returns)
            self._estimators.append(est)
            self._results.append(result)
            t_len = min(T, len(result.beta))
            betas[:t_len, j] = result.beta[:t_len]

        cols = [f"beta_{i}" for i in range(N)]
        return pd.DataFrame(betas, columns=cols)

    @property
    def results(self) -> List[KalmanHedgeResult]:
        return self._results


def _variance_reduction(y: np.ndarray, spread: np.ndarray) -> float:
    """Fractional variance reduction from hedging."""
    valid = ~np.isnan(y) & ~np.isnan(spread)
    if valid.sum() < 5:
        return 0.0
    var_y = float(np.var(y[valid]))
    var_s = float(np.var(spread[valid]))
    if var_y < 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - var_s / var_y))


def _compute_hedge_effectiveness(y: np.ndarray, spread: np.ndarray) -> float:
    """R² of hedged portfolio returns vs zero (higher = better hedge)."""
    valid = ~np.isnan(spread)
    if valid.sum() < 5:
        return 0.0
    ss_hedged = np.sum(spread[valid] ** 2)
    ss_unhedged = np.sum(y[valid] ** 2) if np.sum(~np.isnan(y)) >= 5 else 1.0
    if ss_unhedged < 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ss_hedged / ss_unhedged))


def _empty_hedge_result(n: int) -> KalmanHedgeResult:
    return KalmanHedgeResult(
        alpha=np.full(n, np.nan),
        beta=np.full(n, np.nan),
        spread=np.full(n, np.nan),
        pred_error=np.full(n, np.nan),
        variance_reduction=0.0,
        hedge_effectiveness=0.0,
        terminal_beta=1.0,
        beta_volatility=0.0,
    )
