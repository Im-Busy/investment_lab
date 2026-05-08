"""Black-Litterman Portfolio Optimizer.

Combines market equilibrium returns (CAPM priors) with subjective investor
views using Bayesian shrinkage. Produces posterior expected returns that
are more stable and intuitive than raw mean-variance optimization.

Algorithm:
  1. Compute equilibrium returns: Π = λ · Σ · w_mkt
     (What returns would justify current market weights?)
  2. Express investor views: Q = P · μ + ε,  ε ~ N(0, Ω)
  3. Bayesian blend: posterior = f(prior Π, views (P, Q, Ω), uncertainty τ)
  4. Mean-variance optimization on posterior returns → portfolio weights

Key reference: Black & Litterman (1992), "Global Portfolio Optimization"
Formulas from: He & Litterman (1999), "The Intuition Behind Black-Litterman"

Integrates with existing MultiStrategyEngine weight schemes and PositionSizer.

Usage:
    >>> bl = BlackLittermanOptimizer()
    >>> result = bl.optimize(returns_df, market_caps, views)
    >>> print(result.weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class BLView:
    """A single Black-Litterman view.

    Views can be absolute (e.g., "Asset A returns 8%") or relative
    (e.g., "Asset A will outperform Asset B by 3%").

    Attributes:
        assets: List of asset names involved in this view.
        weights: Weights for each asset in the view (sum to 0 for relative views).
        value: Expected value (e.g., 0.05 = 5% return, 0.03 = 3% outperformance).
        confidence: Confidence in this view (0-1, higher = more certain).
    """

    assets: List[str]
    weights: List[float]
    value: float
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if len(self.assets) != len(self.weights):
            raise ValueError(
                f"assets and weights must have same length: "
                f"{len(self.assets)} vs {len(self.weights)}"
            )
        total = sum(self.weights)
        if not np.isfinite(total):
            raise ValueError(f"Invalid weights: {self.weights}")


@dataclass
class BLConfig:
    """Black-Litterman optimizer configuration.

    Attributes:
        tau: Uncertainty in the equilibrium prior (0.01-0.05 typical).
             Higher tau → trusts views more, trusts equilibrium less.
        risk_aversion: Risk aversion parameter λ (2.0-4.0 typical).
        max_weight: Maximum weight per asset as fraction (0-1).
        min_weight: Minimum weight per asset (0 = no short).
        shrinkage: Shrinkage toward equilibrium weights (0-1).
        target_volatility: Target annualized portfolio volatility.
    """

    tau: float = 0.025
    risk_aversion: float = 2.5
    max_weight: float = 0.40
    min_weight: float = 0.0
    shrinkage: float = 0.0
    target_volatility: float = 0.15


@dataclass
class BLResult:
    """Results from Black-Litterman optimization.

    Attributes:
        weights: Optimal portfolio weights per asset.
        prior_returns: Equilibrium (prior) expected returns.
        posterior_returns: Posterior expected returns after views.
        prior_cov: Prior covariance matrix.
        posterior_cov: Posterior covariance matrix.
        portfolio_return: Expected portfolio return (annualized).
        portfolio_volatility: Expected portfolio volatility (annualized).
        sharpe_ratio: Expected Sharpe ratio.
        view_impact: How much each view changed the weights vs prior.
    """

    weights: Dict[str, float]
    prior_returns: Dict[str, float]
    posterior_returns: Dict[str, float]
    prior_cov: pd.DataFrame
    posterior_cov: pd.DataFrame
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    view_impact: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "prior_returns": {k: round(v, 4) for k, v in self.prior_returns.items()},
            "posterior_returns": {k: round(v, 4) for k, v in self.posterior_returns.items()},
            "portfolio_return": round(self.portfolio_return, 4),
            "portfolio_volatility": round(self.portfolio_volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "view_impact": {k: round(v, 4) for k, v in self.view_impact.items()},
        }

    def weights_dataframe(self) -> pd.DataFrame:
        rows = []
        for asset in self.weights:
            rows.append(
                {
                    "asset": asset,
                    "weight": self.weights[asset],
                    "prior_return": self.prior_returns.get(asset, 0.0),
                    "posterior_return": self.posterior_returns.get(asset, 0.0),
                    "return_delta": (
                        self.posterior_returns.get(asset, 0.0) - self.prior_returns.get(asset, 0.0)
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("weight", ascending=False)

    def __repr__(self) -> str:
        top3 = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}: {v:.1%}" for k, v in top3)
        return (
            f"BLResult(sharpe={self.sharpe_ratio:.3f}, "
            f"ret={self.portfolio_return:.1%}, "
            f"vol={self.portfolio_volatility:.1%}, "
            f"top=[{top_str}])"
        )


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimizer.

    Produces optimal portfolio weights by blending market equilibrium
    returns with investor views through Bayesian shrinkage.

    The optimizer supports:
      - Absolute views: "Asset X will return Y%"
      - Relative views: "Asset A will outperform B by C%"
      - Unconstrained and constrained (long-only, max weight) optimization
      - Volatility targeting

    Example:
        >>> returns = pd.DataFrame(...)  # T x N returns
        >>> market_caps = {"AAPL": 3e12, "MSFT": 2.8e12, ...}
        >>> views = [
        ...     BLView(["AAPL", "MSFT"], [1.0, -1.0], 0.03, confidence=0.6),
        ...     BLView(["NVDA"], [1.0], 0.10, confidence=0.4),
        ... ]
        >>> bl = BlackLittermanOptimizer()
        >>> result = bl.optimize(returns, market_caps, views)
        >>> print(result.weights_dataframe())
    """

    def __init__(self, config: Optional[BLConfig] = None) -> None:
        self.config = config or BLConfig()

    def optimize(
        self,
        returns: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        views: Optional[Sequence[BLView]] = None,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> BLResult:
        """Run Black-Litterman optimization.

        Args:
            returns: T x N DataFrame of historical returns (one column per asset).
            market_caps: Dict mapping asset name → market cap. If None, uses
                         equal weights as prior.
            views: List of BLView objects expressing investor views.
            constraints: Optional per-asset weight bounds {asset: (min, max)}.

        Returns:
            BLResult with optimal weights and diagnostics.
        """
        assets = list(returns.columns)
        n = len(assets)
        returns_arr = returns.values.astype(np.float64)

        prior_cov = self._compute_covariance(returns_arr, assets)

        if market_caps is None:
            mkt_weights = np.ones(n) / n
        else:
            mkt_weights = self._market_cap_weights(assets, market_caps)

        prior_returns = self._equilibrium_returns(prior_cov, mkt_weights)

        if views:
            posterior_returns, posterior_cov = self._apply_views(
                prior_returns, prior_cov, views, assets, mkt_weights
            )
        else:
            posterior_returns = prior_returns.copy()
            posterior_cov = prior_cov.copy()

        weights = self._optimize_weights(posterior_returns, posterior_cov, constraints)

        port_ret = float(weights @ posterior_returns)
        port_vol = float(np.sqrt(weights @ posterior_cov @ weights))
        sharpe = port_ret / port_vol if port_vol > 1e-10 else 0.0

        impact = {}
        if views:
            prior_weights = self._optimize_weights(prior_returns, prior_cov, constraints)
            for i, asset in enumerate(assets):
                impact[asset] = float(weights[i] - prior_weights[i])

        result = BLResult(
            weights=dict(zip(assets, weights)),
            prior_returns=dict(zip(assets, prior_returns)),
            posterior_returns=dict(zip(assets, posterior_returns)),
            prior_cov=prior_cov,
            posterior_cov=posterior_cov,
            portfolio_return=port_ret,
            portfolio_volatility=port_vol,
            sharpe_ratio=sharpe,
            view_impact=impact,
        )

        return result

    def _compute_covariance(self, returns: np.ndarray, assets: List[str]) -> np.ndarray:
        """Compute sample covariance matrix with Ledoit-Wolf shrinkage.

        Shrinkage toward a constant-correlation target improves stability
        when n_assets is large relative to n_periods.
        """
        n = returns.shape[1]
        _ = n  # used for dimension context
        cov = np.cov(returns, rowvar=False)

        if self.config.shrinkage > 0:
            try:
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf()
                cov_shrunk = lw.fit(returns).covariance_
                cov = (1.0 - self.config.shrinkage) * cov + self.config.shrinkage * cov_shrunk
            except Exception:
                pass

        cov = _ensure_positive_definite(cov)
        return pd.DataFrame(cov, index=assets, columns=assets).values

    def _market_cap_weights(self, assets: List[str], market_caps: Dict[str, float]) -> np.ndarray:
        """Compute market-cap-weighted prior weights."""
        caps = np.array([market_caps.get(a, 0.0) for a in assets])
        total = caps.sum()
        if total <= 0:
            return np.ones(len(assets)) / len(assets)
        return caps / total

    def _equilibrium_returns(self, cov: np.ndarray, mkt_weights: np.ndarray) -> np.ndarray:
        """Compute implied equilibrium returns: Π = λ · Σ · w_mkt.

        These are the returns that would make w_mkt optimal under MVO.
        """
        risk_aversion = self.config.risk_aversion
        return risk_aversion * (cov @ mkt_weights)

    def _apply_views(
        self,
        prior_returns: np.ndarray,
        prior_cov: np.ndarray,
        views: Sequence[BLView],
        assets: List[str],
        mkt_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply investor views via Bayesian updating.

        Posterior = [(τΣ)^-1 + P'Ω^-1 P]^-1 × [(τΣ)^-1 Π + P'Ω^-1 Q]
        """
        n = len(assets)
        tau = self.config.tau

        tau_cov_inv = np.linalg.inv(tau * prior_cov)

        P_rows = []
        Q_vals = []
        omega_diag = []

        asset_idx = {a: i for i, a in enumerate(assets)}

        for view in views:
            p_row = np.zeros(n)
            for asset, weight in zip(view.assets, view.weights):
                if asset in asset_idx:
                    p_row[asset_idx[asset]] = weight
            P_rows.append(p_row)
            Q_vals.append(view.value)

            view_uncertainty = (1.0 - view.confidence) * 0.1 + 1e-6
            omega_diag.append(view_uncertainty)

        if not P_rows:
            return prior_returns.copy(), prior_cov.copy()

        P = np.array(P_rows)
        Q = np.array(Q_vals)

        omega = np.diag(omega_diag)
        omega_inv = np.linalg.inv(omega)

        posterior_cov_inv = tau_cov_inv + P.T @ omega_inv @ P

        try:
            posterior_cov = np.linalg.inv(posterior_cov_inv)
        except np.linalg.LinAlgError:
            posterior_cov = _ensure_positive_definite(prior_cov.copy())

        posterior_returns = posterior_cov @ (tau_cov_inv @ prior_returns + P.T @ omega_inv @ Q)

        posterior_cov = posterior_cov + prior_cov

        return posterior_returns, _ensure_positive_definite(posterior_cov)

    def _optimize_weights(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Solve unconstrained or constrained MVO for optimal weights.

        Unconstrained: w* = (λΣ)^-1 μ  (then normalize to sum to 1)
        Constrained: use scipy.optimize with linear constraints.
        """
        n = len(returns)
        lamb = self.config.risk_aversion

        if constraints is None:
            try:
                w_raw = np.linalg.solve(lamb * cov, returns)
            except np.linalg.LinAlgError:
                cov_reg = cov + np.eye(n) * 1e-6
                w_raw = np.linalg.solve(lamb * cov_reg, returns)

            w_sum = np.sum(w_raw)
            if abs(w_sum) > 1e-10:
                weights = w_raw / w_sum
            else:
                weights = np.ones(n) / n

            weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
            w_sum = np.sum(weights)
            if w_sum > 1e-10:
                weights /= w_sum
            return weights

        return self._constrained_optimize(returns, cov, constraints)

    def _constrained_optimize(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
        constraints: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        """Solve MVO with per-asset weight bounds via scipy."""
        try:
            from scipy.optimize import minimize
        except ImportError:
            n = len(returns)
            return np.ones(n) / n

        n = len(returns)
        x0 = np.ones(n) / n
        bounds = [(-1.0, 1.0)] * n

        def objective(w: np.ndarray) -> float:
            w_sum = np.sum(w)
            if w_sum < 1e-10:
                return 1e10
            w_norm = w / w_sum
            port_ret = float(w_norm @ returns)
            port_var = float(w_norm @ cov @ w_norm)
            return -port_ret + 0.5 * self.config.risk_aversion * port_var

        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            return np.clip(result.x, self.config.min_weight, self.config.max_weight)

        return np.ones(n) / n

    def compute_efficient_frontier(
        self,
        returns: np.ndarray,
        cov: np.ndarray,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the efficient frontier weights, returns, volatilities.

        Args:
            returns: Expected return vector.
            cov: Covariance matrix.
            n_points: Number of frontier points.

        Returns:
            (weights_matrix, returns_array, volatilities_array)
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            return np.array([]), np.array([]), np.array([])

        n = len(returns)
        target_returns = np.linspace(returns.min(), returns.max(), n_points)

        weights_list = []
        rets_list = []
        vols_list = []

        for target_ret in target_returns:
            x0 = np.ones(n) / n

            def objective(w: np.ndarray) -> float:
                return float(w @ cov @ w)

            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: float(w @ returns) - target_ret},
            ]
            bounds = [(self.config.min_weight, self.config.max_weight)] * n

            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.success:
                w = res.x
                weights_list.append(w)
                rets_list.append(float(w @ returns))
                vols_list.append(float(np.sqrt(w @ cov @ w)))

        if not weights_list:
            return np.array([]), np.array([]), np.array([])

        return (
            np.array(weights_list),
            np.array(rets_list),
            np.array(vols_list),
        )

    def rolling_optimize(
        self,
        returns: pd.DataFrame,
        window: int = 252,
        step: int = 21,
        market_caps: Optional[Dict[str, float]] = None,
        views: Optional[Sequence[BLView]] = None,
    ) -> pd.DataFrame:
        """Run rolling Black-Litterman optimization.

        Args:
            returns: T x N returns DataFrame.
            window: Estimation window in periods.
            step: Step size between rebalances.
            market_caps: Market cap dict (time-invariant or None for equal).
            views: Views list (time-invariant).

        Returns:
            DataFrame of weights over time.
        """
        n = len(returns)
        _assets = list(returns.columns)
        all_weights = []

        for start in range(0, n - window, step):
            end = start + window
            window_returns = returns.iloc[start:end]

            result = self.optimize(window_returns, market_caps, views)
            w_dict = {"date": returns.index[end - 1]}
            w_dict.update(result.weights)
            all_weights.append(w_dict)

        return pd.DataFrame(all_weights).set_index("date")

    def vol_target_weights(
        self,
        result: BLResult,
        target_vol: Optional[float] = None,
    ) -> Dict[str, float]:
        """Scale BL weights to achieve a target portfolio volatility.

        Args:
            result: BLResult from optimize().
            target_vol: Target annualized volatility. Uses config default if None.

        Returns:
            Volatility-targeted weights dict.
        """
        target = target_vol or self.config.target_volatility
        if result.portfolio_volatility <= 1e-10:
            return dict(result.weights)

        scale = target / result.portfolio_volatility
        scale = float(np.clip(scale, 0.1, 3.0))

        weights = np.array([result.weights[a] for a in result.weights])
        scaled = weights * scale

        cash_weight = max(0.0, 1.0 - np.sum(scaled))

        result_dict = {a: s for a, s in zip(result.weights, scaled)}

        if cash_weight < 0:
            total_abs = np.sum(np.abs(np.array(list(result_dict.values()))))
            if total_abs > 1e-10:
                result_dict = {k: v / total_abs for k, v in result_dict.items()}

        return result_dict


def _ensure_positive_definite(cov: np.ndarray) -> np.ndarray:
    """Ensure a covariance matrix is positive definite via eigenvalue clipping."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    recon = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return 0.5 * (recon + recon.T)
