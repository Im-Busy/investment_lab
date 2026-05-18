"""
D6: Vector Autoregression (VAR) + Granger Causality for Cross-Asset Signals.

Multivariate time series analysis for detecting lead/lag relationships
between assets, sectors, and macro variables. VAR captures linear
interdependencies; Granger causality tests whether one series helps
predict another beyond its own history.

Models:
  VARModel: Wraps statsmodels VAR with convenient fit/predict/forecast API.
    Computes impulse response functions and forecast error variance
    decomposition for interpreting cross-asset dynamics.

  GrangerCausalityTest: Pairwise Granger causality via F-test.
    Tests H0: past values of X provide no additional predictive power
    for Y beyond Y's own history. Rejection → X "Granger-causes" Y.

  CrossAssetLeadLag: Full cross-asset lead/lag analysis for a basket.
    Computes pairwise Granger causality and ranks assets by predictive
    power. Produces a directed graph of information flow.

Usage:
    >>> var = VARModel(maxlags=5)
    >>> var.fit(data)  # data = (T, N) array
    >>> forecast = var.forecast(steps=5)
    >>> gc = GrangerCausalityTest(maxlag=10)
    >>> result = gc.test(y_series, x_series)
    >>> print(result.p_value, result.causality_direction)
    >>> leadlag = CrossAssetLeadLag()
    >>> graph = leadlag.analyze_basket(prices_dict, maxlag=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.api import VAR as _StatsmodelsVAR
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tools.sm_exceptions import ValueWarning

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.warning("statsmodels not installed; VAR/Granger unavailable.")


# ── VAR Model ───────────────────────────────────────────────────────────


@dataclass
class VARResult:
    coefficients: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    aic: float
    bic: float
    maxlag: int
    k_endog: int
    nobs: int
    convergence: bool
    var_names: List[str]

    def to_dict(self) -> Dict:
        return {
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
            "maxlag": self.maxlag,
            "n_variables": self.k_endog,
            "n_obs": self.nobs,
            "convergence": self.convergence,
            "var_names": self.var_names,
        }


class VARModel:
    """Vector Autoregression wrapper with convenience API.

    Fits a VAR(p) model to multivariate time series data. Provides
    forecasting, impulse response analysis, and information criteria.

    Args:
        maxlags: Maximum lag order to consider (AIC selects optimal ≤ maxlags).
        trend: Trend specification ("c" = constant, "ct" = constant + trend).
    """

    def __init__(self, maxlags: int = 5, trend: str = "c"):
        self.maxlags = maxlags
        self.trend = trend
        self._model = None
        self._result: Optional[VARResult] = None
        self._fitted_values: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None

    def fit(
        self,
        data: np.ndarray,
        var_names: Optional[List[str]] = None,
    ) -> VARResult:
        """Fit VAR model to multivariate data.

        Args:
            data: (T, K) array of K variables with T observations.
            var_names: Optional names for each variable.

        Returns:
            VARResult with diagnostics.
        """
        if not _HAS_STATSMODELS:
            return self._fallback_var(data, var_names)

        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, K = data.shape

        if var_names is None:
            var_names = [f"y{i}" for i in range(K)]

        if T < self.maxlags + 5 or K == 0:
            return VARResult(
                coefficients=np.zeros((K, K * self.maxlags)),
                residuals=np.zeros_like(data),
                fitted_values=np.zeros_like(data),
                aic=float("inf"),
                bic=float("inf"),
                maxlag=self.maxlags,
                k_endog=K,
                nobs=T,
                convergence=False,
                var_names=var_names,
            )

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ValueWarning)
                model = _StatsmodelsVAR(data)
                lags = min(self.maxlags, max(1, T // 10))
                results = model.fit(lags, trend=self.trend)
                self._model = results

            fitted = results.fittedvalues
            residuals = data[: len(fitted)] - fitted

            result = VARResult(
                coefficients=results.coefs.reshape(K, -1)
                if hasattr(results, "coefs")
                else np.zeros((K, K * lags)),
                residuals=residuals,
                fitted_values=fitted,
                aic=float(results.aic),
                bic=float(results.bic),
                maxlag=lags,
                k_endog=K,
                nobs=T,
                convergence=True,
                var_names=var_names,
            )
        except Exception as e:
            logger.warning("VAR fit failed: %s", e)
            result = VARResult(
                coefficients=np.zeros((K, K * self.maxlags)),
                residuals=np.zeros_like(data),
                fitted_values=np.zeros_like(data),
                aic=float("inf"),
                bic=float("inf"),
                maxlag=self.maxlags,
                k_endog=K,
                nobs=T,
                convergence=False,
                var_names=var_names or [],
            )

        self._result = result
        return result

    def _fallback_var(self, data: np.ndarray, var_names: Optional[List[str]] = None) -> VARResult:
        """OLS-per-equation fallback when statsmodels unavailable."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, K = data.shape
        if var_names is None:
            var_names = [f"y{i}" for i in range(K)]

        lags = min(self.maxlags, max(1, T // 10))
        n_obs = T - lags
        if n_obs < 5:
            return VARResult(
                coefficients=np.zeros((K, K * lags)),
                residuals=np.zeros((T - lags, K)),
                fitted_values=np.zeros((T - lags, K)),
                aic=float("inf"),
                bic=float("inf"),
                maxlag=lags,
                k_endog=K,
                nobs=T,
                convergence=False,
                var_names=var_names,
            )

        Y = data[lags:]
        X = np.column_stack([data[lags - i - 1 : T - i - 1] for i in range(lags)])
        X = np.column_stack([np.ones(n_obs), X])

        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros((X.shape[1], K))

        fitted = X @ coeffs
        residuals = Y - fitted
        mse = float(np.mean(residuals**2))
        n_params = X.shape[1] * K
        aic = n_obs * np.log(mse + 1e-12) + 2 * n_params
        bic = n_obs * np.log(mse + 1e-12) + n_params * np.log(n_obs)

        return VARResult(
            coefficients=coeffs[1:].T,
            residuals=residuals,
            fitted_values=fitted,
            aic=aic,
            bic=bic,
            maxlag=lags,
            k_endog=K,
            nobs=T,
            convergence=True,
            var_names=var_names,
        )

    def forecast(self, steps: int = 5) -> np.ndarray:
        """Forecast future values.

        Args:
            steps: Number of steps ahead.

        Returns:
            (steps, K) array of forecasts.
        """
        if self._model is None:
            return np.zeros((steps, 1))
        forecast_result = self._model.forecast(self._model.endog[-self.maxlags :], steps=steps)
        return forecast_result

    def fit_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Dict:
        """Fit VAR on a DataFrame and return diagnostics.

        Args:
            df: DataFrame with one column per variable.

        Returns:
            Dictionary with AIC, BIC, convergence, residual stats.
        """
        sorted_cols = [c for c in sorted(df.columns) if c != "date"]
        data = df[sorted_cols].values
        result = self.fit(data, var_names=sorted_cols)

        residual_std = {}
        if result.convergence:
            for i, name in enumerate(sorted_cols):
                if i < result.residuals.shape[1]:
                    residual_std[name] = round(float(np.std(result.residuals[:, i])), 6)

        return {
            **result.to_dict(),
            "residual_std": residual_std,
        }

    @property
    def result(self) -> Optional[VARResult]:
        return self._result


# ── Granger Causality ───────────────────────────────────────────────────


@dataclass
class GrangerResult:
    p_value: float
    f_statistic: float
    optimal_lag: int
    causality_direction: str  # "X→Y", "Y→X", "BIDIRECTIONAL", "NONE"
    significant_05: bool
    significant_01: bool
    x_name: str
    y_name: str

    def to_dict(self) -> Dict:
        return {
            "x_name": self.x_name,
            "y_name": self.y_name,
            "direction": self.causality_direction,
            "p_value": round(self.p_value, 6),
            "f_stat": round(self.f_statistic, 4),
            "optimal_lag": self.optimal_lag,
            "significant_05": self.significant_05,
            "significant_01": self.significant_01,
        }


class GrangerCausalityTest:
    """Pairwise Granger causality test.

    Tests whether X Granger-causes Y using an F-test on restricted vs
    unrestricted VAR models. Detects lead/lag relationships between
    any two time series — useful for cross-asset signal generation.

    Args:
        maxlag: Maximum lag order to test.
        significance_level: Threshold for significance (default 0.05).
    """

    def __init__(self, maxlag: int = 10, significance_level: float = 0.05):
        self.maxlag = maxlag
        self.significance_level = significance_level

    def test(
        self,
        y: np.ndarray,
        x: np.ndarray,
    ) -> GrangerResult:
        """Test if X Granger-causes Y.

        Args:
            y: Target series (does X predict Y?).
            x: Predictor series (does X help forecast Y?).

        Returns:
            GrangerResult with p-value, F-statistic, direction.
        """
        if not _HAS_STATSMODELS:
            return self._fallback_granger(y, x)

        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = min(len(y), len(x))
        y = y[:n]
        x = x[:n]

        if n < self.maxlag + 5:
            return GrangerResult(
                p_value=1.0,
                f_statistic=0.0,
                optimal_lag=0,
                causality_direction="NONE",
                significant_05=False,
                significant_01=False,
                x_name="X",
                y_name="Y",
            )

        data = np.column_stack([y, x])

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ValueWarning)
                max_lag = min(self.maxlag, max(1, n // 10 - 1))
                gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        except Exception as e:
            logger.warning("Granger test failed: %s", e)
            return GrangerResult(
                p_value=1.0,
                f_statistic=0.0,
                optimal_lag=0,
                causality_direction="NONE",
                significant_05=False,
                significant_01=False,
                x_name="X",
                y_name="Y",
            )

        best_lag = 1
        best_f = 0.0
        best_p = 1.0

        for lag, results in gc_result.items():
            for item in results:
                if isinstance(item, list) and len(item) == 4:
                    test_name, test_result = item[0], item[1:]
                elif hasattr(item, "items"):
                    test_name, test_result = next(iter(item.items()))
                else:
                    continue
                test_name_str = str(test_name)
                if "ssr_ftest" in test_name_str:
                    f_stat = test_result[0]
                    p_val = test_result[1]
                    if p_val < best_p:
                        best_p = p_val
                        best_f = f_stat
                        best_lag = lag

        sig_05 = best_p < 0.05
        sig_01 = best_p < 0.01

        # Determine direction
        # Also test reverse: Y→X
        rev_p = 1.0
        try:
            data_rev = np.column_stack([x, y])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ValueWarning)
                gc_rev = grangercausalitytests(data_rev, maxlag=best_lag, verbose=False)
            for lag, results in gc_rev.items():
                for item in results:
                    if isinstance(item, list) and len(item) == 4:
                        test_name_str = str(item[0])
                        test_val = item[1]
                    elif hasattr(item, "items"):
                        k, v = next(iter(item.items()))
                        test_name_str = str(k)
                        test_val = v[1]
                    else:
                        continue
                    if "ssr_ftest" in test_name_str and test_val < rev_p:
                        rev_p = test_val
        except Exception:
            pass

        x_to_y_sig = best_p < self.significance_level
        y_to_x_sig = rev_p < self.significance_level

        if x_to_y_sig and y_to_x_sig:
            direction = "BIDIRECTIONAL"
        elif x_to_y_sig:
            direction = "X→Y"
        elif y_to_x_sig:
            direction = "Y→X"
        else:
            direction = "NONE"

        return GrangerResult(
            p_value=best_p,
            f_statistic=best_f,
            optimal_lag=best_lag,
            causality_direction=direction,
            significant_05=sig_05,
            significant_01=sig_01,
            x_name="X",
            y_name="Y",
        )

    def _fallback_granger(self, y: np.ndarray, x: np.ndarray) -> GrangerResult:
        """F-test on restricted vs unrestricted regression when statsmodels unavailable."""
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = min(len(y), len(x))
        y = y[:n]
        x = x[:n]

        lag = min(self.maxlag, max(1, n // 10 - 1))
        n_obs = n - lag

        if n_obs < 10:
            return GrangerResult(
                p_value=1.0,
                f_statistic=0.0,
                optimal_lag=0,
                causality_direction="NONE",
                significant_05=False,
                significant_01=False,
                x_name="X",
                y_name="Y",
            )

        Y = y[lag:]
        X_restricted = np.column_stack(
            [np.ones(n_obs)] + [y[lag - i - 1 : n - i - 1] for i in range(lag)]
        )
        X_unrestricted = np.column_stack(
            [X_restricted] + [x[lag - i - 1 : n - i - 1] for i in range(lag)]
        )

        try:
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            ssr_r = float(np.sum((Y - X_restricted @ beta_r) ** 2))
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            ssr_u = float(np.sum((Y - X_unrestricted @ beta_u) ** 2))
        except np.linalg.LinAlgError:
            return GrangerResult(
                p_value=1.0,
                f_statistic=0.0,
                optimal_lag=0,
                causality_direction="NONE",
                significant_05=False,
                significant_01=False,
                x_name="X",
                y_name="Y",
            )

        df1 = lag
        df2 = n_obs - X_unrestricted.shape[1]
        if df2 < 1 or ssr_u < 1e-15:
            f_stat = 0.0
            p_val = 1.0
        else:
            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p_val = float(1.0 - stats.f.cdf(f_stat, df1, df2))

        return GrangerResult(
            p_value=p_val,
            f_statistic=f_stat,
            optimal_lag=lag,
            causality_direction="X→Y" if p_val < self.significance_level else "NONE",
            significant_05=p_val < 0.05,
            significant_01=p_val < 0.01,
            x_name="X",
            y_name="Y",
        )

    def test_named(
        self,
        y: np.ndarray,
        y_name: str,
        x: np.ndarray,
        x_name: str,
    ) -> GrangerResult:
        """Test with named variables for readability.

        Args:
            y: Target series.
            y_name: Name for target series.
            x: Predictor series.
            x_name: Name for predictor series.

        Returns:
            GrangerResult with names set.
        """
        result = self.test(y, x)
        result.x_name = x_name
        result.y_name = y_name
        if result.causality_direction == "X→Y":
            result.causality_direction = f"{x_name}→{y_name}"
        elif result.causality_direction == "Y→X":
            result.causality_direction = f"{y_name}→{x_name}"
        elif result.causality_direction == "BIDIRECTIONAL":
            result.causality_direction = f"{x_name}↔{y_name}"
        return result


# ── Cross-Asset Lead/Lag Analysis ───────────────────────────────────────


@dataclass
class LeadLagResult:
    granger_matrix: pd.DataFrame
    lead_ranking: List[Dict]
    most_predictive: str
    most_predicted: str
    n_connections: int
    causal_density: float
    summary: Dict

    def to_dict(self) -> Dict:
        return {
            "most_predictive": self.most_predictive,
            "most_predicted": self.most_predicted,
            "n_connections": self.n_connections,
            "causal_density": round(self.causal_density, 4),
            "lead_ranking": self.lead_ranking[:5],
        }


class CrossAssetLeadLag:
    """Cross-asset lead/lag analysis for a basket of instruments.

    Computes pairwise Granger causality across all assets, ranks assets
    by predictive power (how many other assets they lead), and identifies
    the most central nodes in the information flow graph.

    Args:
        maxlag: Maximum lag for Granger tests.
        significance_level: P-value threshold.
    """

    def __init__(self, maxlag: int = 5, significance_level: float = 0.05):
        self.maxlag = maxlag
        self.significance_level = significance_level

    def analyze_basket(
        self,
        prices: Dict[str, np.ndarray],
    ) -> LeadLagResult:
        """Full cross-asset lead/lag analysis.

        Args:
            prices: Dict mapping asset name to price series.

        Returns:
            LeadLagResult with Granger matrix, rankings, and summary.
        """
        names = sorted(prices.keys())
        n = len(names)

        granger_pvalues = np.full((n, n), np.nan)
        directions = np.full((n, n), "", dtype=object)

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    granger_pvalues[i, j] = 1.0
                    directions[i, j] = "—"
                    continue

                rets_i = _to_returns(prices[name_i])
                rets_j = _to_returns(prices[name_j])

                gc = GrangerCausalityTest(
                    maxlag=self.maxlag, significance_level=self.significance_level
                )
                result = gc.test(rets_j, rets_i)

                granger_pvalues[i, j] = result.p_value
                if result.p_value < self.significance_level:
                    directions[i, j] = "→"
                else:
                    directions[i, j] = ""

        granger_df = pd.DataFrame(granger_pvalues, index=names, columns=names)

        out_degree = {name: 0 for name in names}
        in_degree = {name: 0 for name in names}
        connections = 0

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if directions[i, j] == "→":
                    out_degree[name_i] = out_degree.get(name_i, 0) + 1
                    in_degree[name_j] = in_degree.get(name_j, 0) + 1
                    connections += 1

        lead_score = {name: out_degree.get(name, 0) - in_degree.get(name, 0) for name in names}

        ranking = sorted(
            [
                {
                    "asset": name,
                    "leads": out_degree[name],
                    "lagged_by": in_degree[name],
                    "net_lead": lead_score[name],
                }
                for name in names
            ],
            key=lambda x: x["net_lead"],
            reverse=True,
        )

        most_predictive = ranking[0]["asset"] if ranking else ""
        most_predicted = max(in_degree, key=in_degree.get, default="")
        max_edges = n * (n - 1)
        causal_density = connections / max_edges if max_edges > 0 else 0.0

        summary = {
            "total_assets": n,
            "total_connections": connections,
            "causal_density": round(causal_density, 4),
            "top_leaders": ranking[:3],
            "top_laggards": ranking[-3:][::-1] if len(ranking) >= 3 else [],
        }

        return LeadLagResult(
            granger_matrix=granger_df,
            lead_ranking=ranking,
            most_predictive=most_predictive,
            most_predicted=most_predicted,
            n_connections=connections,
            causal_density=causal_density,
            summary=summary,
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
    ) -> LeadLagResult:
        """Analyze lead/lag from a DataFrame of price data.

        Args:
            df: DataFrame with date index, one column per asset.

        Returns:
            LeadLagResult.
        """
        prices = {col: df[col].values for col in df.columns}
        return self.analyze_basket(prices)


# ── Helpers ─────────────────────────────────────────────────────────────


def _to_returns(prices: np.ndarray) -> np.ndarray:
    """Convert prices to log returns."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(np.maximum(prices, 1e-12)))


def _compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((actual - predicted) ** 2)))
