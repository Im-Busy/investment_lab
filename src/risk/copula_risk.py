"""
Q6: Copula Tail-Risk Models for Basket Strategies.

Joint extreme risk modeling using Gaussian and t-copulas. Fixes the
independence assumption in current VaR/CVaR calculations, which
underestimates simultaneous drawdowns in multi-asset portfolios.

Models:
  Gaussian Copula:
    Rank-transformed correlation structure. Captures linear dependence
    but underestimates tail dependence (joint extremes).

  t-Copula:
    Symmetric tail dependence via degrees-of-freedom parameter.
    Better at capturing joint crash behavior.

Usage:
    >>> copula = GaussianCopulaRisk()
    >>> copula.fit(returns_matrix)
    >>> var_99 = copula.var(confidence=0.99, n_sims=10_000)
    >>> cvar_99 = copula.cvar(confidence=0.99, n_sims=10_000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CopulaRiskResult:
    """Copula-based risk metrics."""

    name: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    n_assets: int
    n_obs: int
    correlation_matrix: np.ndarray
    tail_dependence: float

    def summary(self) -> str:
        return (
            f"{self.name}: VaR95={self.var_95:.4f} VaR99={self.var_99:.4f} "
            f"CVaR95={self.cvar_95:.4f} CVaR99={self.cvar_99:.4f} "
            f"tail_dep={self.tail_dependence:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "var_95": round(self.var_95, 6),
            "var_99": round(self.var_99, 6),
            "cvar_95": round(self.cvar_95, 6),
            "cvar_99": round(self.cvar_99, 6),
            "n_assets": self.n_assets,
            "n_obs": self.n_obs,
            "tail_dependence": round(self.tail_dependence, 4),
        }


class GaussianCopulaRisk:
    """Gaussian copula for joint portfolio risk.

    Models dependence via rank correlations and simulates joint
    return draws from the fitted copula.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._corr: Optional[np.ndarray] = None
        self._marginals: list[object] = []
        self._n_assets: int = 0
        self._tail_dep: float = 0.0

    def fit(self, returns: np.ndarray | pd.DataFrame) -> GaussianCopulaRisk:
        """Fit Gaussian copula to return data.

        Args:
            returns: (n_obs, n_assets) return matrix.
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        returns = returns[~np.any(np.isnan(returns), axis=1)]

        self._n_assets = returns.shape[1]
        self._corr = np.corrcoef(returns, rowvar=False)
        self._marginals = []

        for i in range(self._n_assets):
            params = stats.t.fit(returns[:, i])
            self._marginals.append(params)

        self._tail_dep = self._estimate_tail_dependence(returns)
        return self

    def _estimate_tail_dependence(self, returns: np.ndarray) -> float:
        """Estimate average lower tail dependence from empirical data."""
        tail_deps = []
        for i in range(self._n_assets):
            for j in range(i + 1, self._n_assets):
                q_i = np.percentile(returns[:, i], 5)
                q_j = np.percentile(returns[:, j], 5)
                mask = (returns[:, i] < q_i) & (returns[:, j] < q_j)
                tail_deps.append(mask.mean() / 0.05)  # lambda_L estimate
        return float(np.mean(tail_deps)) if tail_deps else 0.0

    def simulate(self, n_sims: int = 10_000) -> np.ndarray:
        """Simulate joint returns from fitted copula.

        Args:
            n_sims: Number of simulation draws.

        Returns:
            (n_sims, n_assets) simulated return matrix.
        """
        if self._corr is None:
            raise ValueError("Call fit() first")

        norm_draws = self._rng.multivariate_normal(
            mean=np.zeros(self._n_assets), cov=self._corr, size=n_sims
        )
        uniforms = stats.norm.cdf(norm_draws)

        sim_returns = np.zeros((n_sims, self._n_assets))
        for i in range(self._n_assets):
            sim_returns[:, i] = stats.t.ppf(
                np.clip(uniforms[:, i], 1e-10, 1 - 1e-10),
                df=self._marginals[i][0],
                loc=self._marginals[i][1],
                scale=self._marginals[i][2],
            )

        return sim_returns

    def var(self, confidence: float = 0.99, n_sims: int = 10_000) -> float:
        """Value-at-Risk at given confidence level.

        Assumes equal-weight portfolio.
        """
        sims = self.simulate(n_sims)
        port_returns = sims.mean(axis=1)
        return float(np.percentile(port_returns, (1 - confidence) * 100))

    def cvar(self, confidence: float = 0.99, n_sims: int = 10_000) -> float:
        """Conditional Value-at-Risk (expected shortfall)."""
        sims = self.simulate(n_sims)
        port_returns = sims.mean(axis=1)
        var_thresh = np.percentile(port_returns, (1 - confidence) * 100)
        return float(port_returns[port_returns <= var_thresh].mean())

    def evaluate(self, n_sims: int = 10_000) -> CopulaRiskResult:
        """Compute full risk metrics."""
        return CopulaRiskResult(
            name="GaussianCopula",
            var_95=self.var(0.95, n_sims),
            var_99=self.var(0.99, n_sims),
            cvar_95=self.cvar(0.95, n_sims),
            cvar_99=self.cvar(0.99, n_sims),
            n_assets=self._n_assets,
            n_obs=0,
            correlation_matrix=self._corr if self._corr is not None else np.array([]),
            tail_dependence=self._tail_dep,
        )


class TCopulaRisk(GaussianCopulaRisk):
    """t-copula with symmetric tail dependence.

    Better captures joint extreme behavior than Gaussian copula.
    Uses a multivariate t-distribution for the dependence structure.
    """

    def __init__(self, df: int = 4, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.df = df

    def simulate(self, n_sims: int = 10_000) -> np.ndarray:
        """Simulate from t-copula with heavy tails."""
        if self._corr is None:
            raise ValueError("Call fit() first")

        chi2 = self._rng.chisquare(self.df, n_sims) / self.df
        norm_draws = self._rng.multivariate_normal(
            mean=np.zeros(self._n_assets), cov=self._corr, size=n_sims
        )
        t_draws = norm_draws / np.sqrt(chi2[:, np.newaxis])
        uniforms = stats.t.cdf(t_draws, df=self.df)

        sim_returns = np.zeros((n_sims, self._n_assets))
        for i in range(self._n_assets):
            sim_returns[:, i] = stats.t.ppf(
                np.clip(uniforms[:, i], 1e-10, 1 - 1e-10),
                df=self._marginals[i][0],
                loc=self._marginals[i][1],
                scale=self._marginals[i][2],
            )

        return sim_returns

    def evaluate(self, n_sims: int = 10_000) -> CopulaRiskResult:
        result = super().evaluate(n_sims)
        result.name = f"tCopula(df={self.df})"
        return result
