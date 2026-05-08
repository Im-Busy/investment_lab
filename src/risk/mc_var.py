"""Monte Carlo and Historical VaR / CVaR risk modeling.

Replaces the binomial-approximation VaR in position_probability.py with
proper distributional risk measures:

  - Historical VaR (non-parametric, resamples actual returns)
  - Monte Carlo VaR (parametric with Student's t for fat tails)
  - CVaR / Expected Shortfall (average loss beyond VaR threshold)
  - Bootstrap confidence intervals for all estimates

Key References:
  - Lopez de Prado, "Advances in Financial Machine Learning", Ch. 14
  - Jorion, "Value-at-Risk: The New Benchmark for Managing Financial Risk"
  - Rockafellar & Uryasev, "Optimization of Conditional Value-at-Risk"

Feeds var_95 and cvar_95 into CircuitBreaker.check_full().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy import stats


@dataclass
class VaRResult:
    """Container for VaR/CVaR estimation results.

    Attributes:
        var_95: 95% Value-at-Risk (% drop, positive = loss).
        var_99: 99% Value-at-Risk (% drop).
        cvar_95: 95% Conditional VaR / Expected Shortfall.
        cvar_99: 99% Conditional VaR / Expected Shortfall.
        max_drawdown_est: Estimated maximum drawdown at 95% confidence.
        method: Estimation method used (historical, monte_carlo, parametric_t).
        params: Fitted distribution parameters.
        confidence_intervals: Bootstrap CIs for VaR/CVaR if computed.
    """

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_est: float
    method: str
    params: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float | str]:
        d: Dict[str, float | str] = {
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
            "cvar_95": round(self.cvar_95, 4),
            "cvar_99": round(self.cvar_99, 4),
            "max_drawdown_est": round(self.max_drawdown_est, 4),
            "method": self.method,
        }
        d.update({k: round(v, 4) for k, v in self.params.items()})
        return d

    def __repr__(self) -> str:
        return (
            f"VaRResult(method={self.method}, "
            f"VaR95={self.var_95:.4f}, CVaR95={self.cvar_95:.4f}, "
            f"MaxDD={self.max_drawdown_est:.4f})"
        )


class MCVaR:
    """Monte Carlo VaR / CVaR estimator for portfolio returns.

    Provides three estimation methods with fat-tail awareness:

    1. Historical: Resamples actual returns (non-parametric, captures real
       tail structure).
    2. Monte Carlo: Parametric with Student's t distribution (handles
       leptokurtic/fat-tailed financial returns).
    3. Parametric Normal: Baseline for comparison.

    Usage:
        >>> mcvar = MCVaR(returns)
        >>> result = mcvar.compute(method="monte_carlo")
        >>> print(f"VaR 95: {result.var_95:.2%}")
        >>> print(f"CVaR 95: {result.cvar_95:.2%}")
        >>> cb.check_full(current_price, peak_price, var_95=result.var_95)
    """

    def __init__(
        self,
        returns: np.ndarray,
        confidence_levels: Tuple[float, ...] = (0.95, 0.99),
        mc_samples: int = 100_000,
        seed: int = 42,
    ) -> None:
        """Initialize MC VaR estimator.

        Args:
            returns: 1-D array of historical period returns (e.g., daily %).
            confidence_levels: VaR confidence levels to compute.
            mc_samples: Number of Monte Carlo simulation paths.
            seed: Random seed for reproducibility.
        """
        returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.ndim != 1:
            raise ValueError(f"Expected 1-D returns array, got shape {returns_arr.shape}")

        self._returns = returns_arr[~np.isnan(returns_arr)]
        if len(self._returns) < 30:
            raise ValueError(f"Need >= 30 returns, got {len(self._returns)}")

        self._confidence_levels = sorted(set(confidence_levels))
        self._mc_samples = int(mc_samples)
        self._rng = np.random.default_rng(seed)

        self._hist_mean: float = 0.0
        self._hist_std: float = 0.0
        self._hist_skew: float = 0.0
        self._hist_kurtosis: float = 0.0
        self._hist_var95: float = 0.0
        self._hist_var99: float = 0.0
        self._hist_cvar95: float = 0.0
        self._hist_cvar99: float = 0.0
        self._fitted_df: float = 5.0
        self._fitted_loc: float = 0.0
        self._fitted_scale: float = 0.01
        self._computed: bool = False

    def compute(
        self,
        method: str = "monte_carlo",
        bootstrap: bool = False,
        n_bootstrap: int = 1000,
    ) -> VaRResult:
        """Compute VaR and CVaR using the specified method.

        Args:
            method: One of "historical", "monte_carlo", "parametric_normal".
            bootstrap: If True, compute bootstrap CIs for estimates.
            n_bootstrap: Number of bootstrap resamples for CI computation.

        Returns:
            VaRResult with var_95, var_99, cvar_95, cvar_99, etc.
        """
        self._compute_sample_stats()
        self._fit_t_distribution()

        if method == "historical":
            result = self._historical_var()
        elif method == "monte_carlo":
            result = self._monte_carlo_var()
        elif method == "parametric_normal":
            result = self._parametric_normal_var()
        else:
            raise ValueError(
                f"Unknown method: {method}. Available: historical, monte_carlo, parametric_normal"
            )

        if bootstrap:
            result.confidence_intervals = self._bootstrap_ci(method, n_bootstrap)

        self._computed = True
        return result

    def _compute_sample_stats(self) -> None:
        """Compute sample moments of the returns distribution."""
        self._hist_mean = float(np.mean(self._returns))
        self._hist_std = float(np.std(self._returns, ddof=1))
        self._hist_skew = float(stats.skew(self._returns))
        self._hist_kurtosis = float(stats.kurtosis(self._returns, fisher=True))

    def _fit_t_distribution(self) -> None:
        """Fit a Student's t distribution to the returns.

        Financial returns exhibit fat tails (kurtosis > 3), making
        Student's t more appropriate than Normal for VaR estimation.
        """
        if len(self._returns) > 1:
            df, loc, scale = stats.t.fit(self._returns)
            self._fitted_df = float(df)
            self._fitted_loc = float(loc)
            self._fitted_scale = float(scale)

    def _historical_var(self) -> VaRResult:
        """Compute VaR/CVaR by resampling the empirical distribution.

        Non-parametric: captures the actual tail shape of historical returns.
        """
        sorted_returns = np.sort(self._returns)

        var_values: Dict[str, float] = {}
        cvar_values: Dict[str, float] = {}

        for level in self._confidence_levels:
            idx = int(len(sorted_returns) * (1.0 - level))
            idx = max(0, min(idx, len(sorted_returns) - 1))
            var = -sorted_returns[idx]
            var_values[f"var_{int(level * 100)}"] = var

            tail = sorted_returns[: idx + 1]
            cvar = -float(np.mean(tail)) if len(tail) > 0 else var
            cvar_values[f"cvar_{int(level * 100)}"] = cvar

        return VaRResult(
            var_95=var_values.get("var_95", 0.0),
            var_99=var_values.get("var_99", 0.0),
            cvar_95=cvar_values.get("cvar_95", 0.0),
            cvar_99=cvar_values.get("cvar_99", 0.0),
            max_drawdown_est=self._estimate_max_drawdown(var_values.get("var_95", 0.0)),
            method="historical",
            params={
                "mean": self._hist_mean,
                "std": self._hist_std,
                "skewness": self._hist_skew,
                "excess_kurtosis": self._hist_kurtosis,
            },
        )

    def _monte_carlo_var(self) -> VaRResult:
        """Compute VaR/CVaR via Monte Carlo simulation.

        Uses fitted Student's t distribution to draw fat-tailed paths.
        """
        simulated = stats.t.rvs(
            df=self._fitted_df,
            loc=self._fitted_loc,
            scale=self._fitted_scale,
            size=self._mc_samples,
            random_state=self._rng,
        )
        sorted_sim = np.sort(simulated)

        var_values: Dict[str, float] = {}
        cvar_values: Dict[str, float] = {}

        for level in self._confidence_levels:
            idx = int(self._mc_samples * (1.0 - level))
            idx = max(0, min(idx, self._mc_samples - 1))
            var = -sorted_sim[idx]
            var_values[f"var_{int(level * 100)}"] = var

            tail = sorted_sim[: idx + 1]
            cvar = -float(np.mean(tail)) if len(tail) > 0 else var
            cvar_values[f"cvar_{int(level * 100)}"] = cvar

        return VaRResult(
            var_95=var_values.get("var_95", 0.0),
            var_99=var_values.get("var_99", 0.0),
            cvar_95=cvar_values.get("cvar_95", 0.0),
            cvar_99=cvar_values.get("cvar_99", 0.0),
            max_drawdown_est=self._estimate_max_drawdown(var_values.get("var_95", 0.0)),
            method="monte_carlo",
            params={
                "df": self._fitted_df,
                "loc": self._fitted_loc,
                "scale": self._fitted_scale,
                "mc_samples": self._mc_samples,
            },
        )

    def _parametric_normal_var(self) -> VaRResult:
        """Compute VaR/CVaR using parametric Normal assumption.

        Provided as baseline for comparison; underestimates tail risk.
        """
        var_values: Dict[str, float] = {}
        cvar_values: Dict[str, float] = {}

        for level in self._confidence_levels:
            z_score = stats.norm.ppf(1.0 - level)
            var = -(self._hist_mean + z_score * self._hist_std)
            var_values[f"var_{int(level * 100)}"] = var

            cvar = -self._hist_mean + self._hist_std * stats.norm.pdf(z_score) / (1.0 - level)
            cvar_values[f"cvar_{int(level * 100)}"] = cvar

        return VaRResult(
            var_95=var_values.get("var_95", 0.0),
            var_99=var_values.get("var_99", 0.0),
            cvar_95=cvar_values.get("cvar_95", 0.0),
            cvar_99=cvar_values.get("cvar_99", 0.0),
            max_drawdown_est=self._estimate_max_drawdown(var_values.get("var_95", 0.0)),
            method="parametric_normal",
            params={
                "mean": self._hist_mean,
                "std": self._hist_std,
            },
        )

    def _bootstrap_ci(
        self,
        method: str,
        n_bootstrap: int,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for VaR/CVaR estimates."""
        n = len(self._returns)
        var95_samples = np.zeros(n_bootstrap)
        cvar95_samples = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = self._rng.choice(self._returns, size=n, replace=True)
            temp_mcvar = MCVaR(
                sample,
                confidence_levels=(0.95,),
                mc_samples=min(10_000, self._mc_samples),
                seed=i,
            )
            r = temp_mcvar.compute(method=method, bootstrap=False)
            var95_samples[i] = r.var_95
            cvar95_samples[i] = r.cvar_95

        ci_level = 0.95
        alpha = (1.0 - ci_level) / 2.0

        return {
            "var_95": (
                float(np.percentile(var95_samples, alpha * 100)),
                float(np.percentile(var95_samples, (1.0 - alpha) * 100)),
            ),
            "cvar_95": (
                float(np.percentile(cvar95_samples, alpha * 100)),
                float(np.percentile(cvar95_samples, (1.0 - alpha) * 100)),
            ),
        }

    def _estimate_max_drawdown(self, var_95: float) -> float:
        """Estimate maximum drawdown from VaR95 using empirical scaling.

        For financial returns, max drawdown typically exceeds single-period
        VaR by a factor of 2-5x depending on autocorrelation and volatility
        clustering.

        Uses fitted t-distribution df to scale: higher df (thinner tails)
        → lower scaling factor.
        """
        if self._fitted_df > 10:
            scaling = 3.0
        elif self._fitted_df > 4:
            scaling = 4.0
        else:
            scaling = 5.0

        return float(var_95 * scaling)

    def compute_rolling(
        self,
        window: int = 252,
        step: int = 21,
    ) -> np.ndarray:
        """Compute rolling VaR95 over the return series.

        Args:
            window: Rolling window size in periods.
            step: Step size between windows.

        Returns:
            Array of rolling VaR95 values.
        """
        n = len(self._returns)
        rolling_var = []

        for start in range(0, n - window, step):
            end = start + window
            window_returns = self._returns[start:end]
            temp_mcvar = MCVaR(window_returns, confidence_levels=(0.95,))
            r = temp_mcvar.compute(method="monte_carlo", bootstrap=False)
            rolling_var.append(r.var_95)

        return np.array(rolling_var)

    def summary(self, method: str = "monte_carlo") -> Dict[str, float]:
        """Return a human-readable summary of risk estimates.

        Args:
            method: Computation method.

        Returns:
            Dict with key risk measures.
        """
        result = self.compute(method=method, bootstrap=True)
        summary_dict = result.to_dict()
        return {k: v for k, v in summary_dict.items() if isinstance(v, float)}

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @property
    def fitted_t_params(self) -> Tuple[float, float, float]:
        return (self._fitted_df, self._fitted_loc, self._fitted_scale)

    def compare_methods(self) -> Dict[str, VaRResult]:
        """Compute VaR/CVaR using all three methods for comparison.

        Returns:
            Dict mapping method name → VaRResult.
        """
        results: Dict[str, VaRResult] = {}
        for method in ("historical", "monte_carlo", "parametric_normal"):
            results[method] = self.compute(method=method, bootstrap=False)
        return results
