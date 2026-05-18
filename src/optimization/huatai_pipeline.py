"""
R13: Full IR→HP→Risk→QP Optimization Pipeline.

Implements the 华泰证券 4-phase systematic optimization chain:
  1. IR-weighted signal synthesis — aggregate pattern signals using rolling IR weights
  2. HP-filtered return forecast — smooth expected returns via Hodrick-Prescott filter
  3. Factor covariance risk model — compute covariance with Ledoit-Wolf shrinkage
  4. Quadratic programming — max return given risk cap via scipy.optimize

Source: 华泰多因子系列1 §2.5-4

Architecture:
    - HuataiPipeline orchestrates the 4-phase chain
    - HuataiResult encapsulates optimized weights + diagnostics
    - run_huatai_pipeline() is the top-level convenience entry point
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.ml.expected_returns import hp_filter, HP_LAMBDA_DAILY

logger = logging.getLogger(__name__)

# ── Default constraints ──
DEFAULT_MAX_WEIGHT = 0.25
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_RISK_CAP = 0.15  # annualized volatility cap
DEFAULT_RISK_FREE = 0.02  # risk-free rate for Sharpe computation


@dataclass
class HuataiResult:
    """Output of the Huatai 4-phase optimization pipeline.

    Attributes:
        weights: Final weights dict {asset_name: weight}.
        expected_returns: HP-smoothed annualized expected returns per asset.
        covariance: Annualized covariance matrix.
        portfolio_return: Expected annualized portfolio return.
        portfolio_risk: Expected annualized portfolio volatility.
        portfolio_sharpe: Expected Sharpe ratio.
        ir_signal: Raw IR-weighted composite signals.
        hp_trend: HP-filtered trend per asset.
        status: "optimal", "suboptimal", or "failed".
        message: Human-readable optimization summary.
    """

    weights: dict[str, float]
    expected_returns: dict[str, float]
    covariance: Optional[np.ndarray] = None
    portfolio_return: float = 0.0
    portfolio_risk: float = 0.0
    portfolio_sharpe: float = 0.0
    ir_signal: Optional[dict[str, float]] = None
    hp_trend: Optional[dict[str, np.ndarray]] = None
    status: str = "optimal"
    message: str = ""
    n_iterations: int = 0
    constraint_violations: list[str] = field(default_factory=list)


class HuataiPipeline:
    """4-phase optimization pipeline (华泰多因子体系 §2.5-4).

    Phase 1 — IR-Weighted Synthesis:
        Aggregate raw pattern/signal scores per asset using Information Ratio
        weights. IR = mean(ret) / std(ret) over a rolling window.

    Phase 2 — HP-Filtered Return Forecast:
        Apply Hodrick-Prescott filter to cumulative returns to extract smooth
        trend. Use the slope of the trend as the expected return.

    Phase 3 — Factor Covariance Risk Model:
        Compute sample covariance with Ledoit-Wolf shrinkage for robustness.
        Optional factor model covariance decomposition.

    Phase 4 — Quadratic Programming:
        max_w  wᵀ μ − (γ/2) wᵀ Σ w
        s.t.   Σ wᵢ = 1,  0 ≤ wᵢ ≤ w_max

        where μ = HP-smoothed expected returns, Σ = shrunk covariance,
        and γ controls the risk-aversion tradeoff.
    """

    def __init__(
        self,
        risk_cap: float = DEFAULT_RISK_CAP,
        max_weight: float = DEFAULT_MAX_WEIGHT,
        min_weight: float = DEFAULT_MIN_WEIGHT,
        risk_free: float = DEFAULT_RISK_FREE,
        hp_lambda: float = HP_LAMBDA_DAILY,
        risk_aversion: Optional[float] = None,
    ):
        """Initialize the pipeline.

        Args:
            risk_cap: Annualized volatility cap for portfolio constraint.
            max_weight: Maximum weight per asset (0.0–1.0).
            min_weight: Minimum weight per asset (0.0–1.0).
            risk_free: Risk-free rate for Sharpe computation.
            hp_lambda: Smoothness parameter for HP filter.
            risk_aversion: Risk aversion coefficient γ. If None, auto-tuned
                to meet the risk_cap constraint.
        """
        self.risk_cap = risk_cap
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free = risk_free
        self.hp_lambda = hp_lambda
        self.risk_aversion = risk_aversion

    # ── Phase 1: IR-Weighted Signal Synthesis ───────────────────────

    def _ir_weighted_signals(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        window: int = 252,
    ) -> dict[str, float]:
        """Compute IR-weighted composite signal per asset.

        Args:
            signals: (T, N) raw signal scores per asset (e.g., pattern confidence).
            returns: (T, N) forward returns aligned with signals.
            window: Rolling window for IR computation.

        Returns:
            {asset_name: ir_weighted_composite_score}
        """
        if signals.empty or returns.empty:
            return {}

        # Align indices
        common = signals.index.intersection(returns.index)
        if len(common) < window:
            logger.warning(
                "Insufficient data for IR computation: %d bars < %d window",
                len(common),
                window,
            )
            return dict.fromkeys(signals.columns, 0.0)

        sig_aligned = signals.loc[common]
        ret_aligned = returns.loc[common]

        ir_weights: dict[str, float] = {}
        for col in sig_aligned.columns:
            if col not in ret_aligned.columns:
                continue
            sig = sig_aligned[col].values
            ret = (
                ret_aligned[col].values[-window:]
                if len(ret_aligned) > window
                else ret_aligned[col].values
            )

            # IR = mean(signal * return) / std(signal * return)
            # But Huatai uses mean(return) / std(return) for IR
            # We use signal-weighted returns: interpret as signal * forward_return
            weighted_ret = (
                sig[-len(ret) :] * ret[: len(sig[-len(ret) :])]
                if len(sig) >= len(ret)
                else sig * ret[: len(sig)]
            )

            if len(weighted_ret) < 20 or np.std(weighted_ret, ddof=1) < 1e-12:
                ir_weights[col] = 0.0
                continue

            ir = np.mean(weighted_ret) / (np.std(weighted_ret, ddof=1) or 1e-12)
            ir_weights[col] = max(ir, 0.0)

        # Normalize to sum to 1
        total_ir = sum(ir_weights.values())
        if total_ir > 0:
            ir_weights = {k: v / total_ir for k, v in ir_weights.items()}
        else:
            n = len(ir_weights)
            ir_weights = {k: 1.0 / n for k in ir_weights}

        return ir_weights

    # ── Phase 2: HP-Filtered Return Forecast ────────────────────────

    def _hp_expected_returns(
        self,
        prices: pd.DataFrame,
        annualize: bool = True,
    ) -> dict[str, float]:
        """Extract expected returns via HP-filtered trend.

        For each asset:
        1. Compute log cumulative returns.
        2. Apply HP filter to extract smooth trend.
        3. Linear regression of trend on time → slope = expected return.

        Args:
            prices: (T, N) price dataframe.
            annualize: If True, multiply daily slope by 252.

        Returns:
            {asset_name: annualized_expected_return}
        """
        expected: dict[str, float] = {}
        hp_trends: dict[str, np.ndarray] = {}

        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < 20:
                expected[col] = 0.0
                continue

            # Log cumulative returns
            log_ret = np.log(series / series.iloc[0]).values

            # HP filter
            trend = hp_filter(log_ret, lam=self.hp_lambda)
            hp_trends[col] = trend

            # Slope via linear regression (last half-weighting for recency)
            n = len(trend)
            x = np.arange(n)
            slope = np.polyfit(x[-min(n, 252) :], trend[-min(n, 252) :], 1)[0]

            if annualize:
                slope *= 252.0

            expected[col] = slope

        self._hp_trends = hp_trends
        return expected

    # ── Phase 3: Factor Covariance Risk Model ───────────────────────

    def _estimate_covariance(
        self,
        returns: pd.DataFrame,
        use_shrinkage: bool = True,
    ) -> np.ndarray:
        """Estimate covariance matrix with Ledoit-Wolf shrinkage.

        Args:
            returns: (T, N) daily return dataframe.
            use_shrinkage: If True, apply Ledoit-Wolf shrinkage to sample cov.

        Returns:
            (N, N) annualized covariance matrix.
        """
        if returns.empty or returns.shape[1] < 2:
            return np.zeros((1, 1))

        # Drop columns with insufficient data
        valid_cols = [c for c in returns.columns if returns[c].notna().sum() > 30]
        if len(valid_cols) < 2:
            logger.warning("Insufficient return data for covariance estimation")
            return np.eye(len(valid_cols)) * 0.04  # default 20% ann vol

        ret_mat = returns[valid_cols].dropna().values

        if use_shrinkage and ret_mat.shape[0] > len(valid_cols):
            cov = self._ledoit_wolf_shrinkage(ret_mat)
        else:
            cov = np.cov(ret_mat, rowvar=False, ddof=1)

        cov *= 252.0
        self._cov_assets = valid_cols
        return cov

    @staticmethod
    def _ledoit_wolf_shrinkage(x: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf (2004) shrinkage estimator for covariance matrix.

        Shrinks the sample covariance toward a structured target
        (constant-correlation model) to reduce estimation error.

        Args:
            x: (T, N) return matrix.

        Returns:
            (N, N) shrunk covariance (daily, not annualized).
        """
        t, n = x.shape
        xm = x - x.mean(axis=0)
        sample_cov = (xm.T @ xm) / (t - 1)

        # Target: constant-correlation model
        stds = np.sqrt(np.diag(sample_cov))
        corr = sample_cov / np.outer(stds, stds)
        np.fill_diagonal(corr, 0.0)
        mean_corr = np.sum(corr) / (n * (n - 1))
        target = mean_corr * np.outer(stds, stds)
        np.fill_diagonal(target, np.diag(sample_cov))

        # Shrinkage intensity
        pi_mat = np.zeros((n, n))
        for i in range(t):
            xi = xm[i : i + 1]
            diff = xi.T @ xi - sample_cov
            pi_mat += diff**2
        pi_mat /= t

        # Asymptotic shrinkage factor
        gamma = (sample_cov - target) ** 2
        numerator = np.sum(pi_mat)
        denominator = np.sum(gamma)
        delta = numerator / max(denominator, 1e-12)
        delta = min(max(delta, 0.0), 1.0)

        shrunk = delta * target + (1.0 - delta) * sample_cov
        return shrunk

    # ── Phase 4: Quadratic Programming ─────────────────────────────

    def _quadratic_program(
        self,
        expected_returns: dict[str, float],
        covariance: np.ndarray,
        asset_names: list[str],
    ) -> dict[str, float]:
        """Solve max wᵀμ − (γ/2) wᵀΣw with weight + risk constraints.

        Uses SLSQP via scipy.optimize.minimize.

        Args:
            expected_returns: {asset_name: annualized expected return}.
            covariance: (N, N) annualized covariance matrix.
            asset_names: Ordered list of asset names matching covariance cols.

        Returns:
            {asset_name: optimized_weight}
        """
        n = len(asset_names)
        if n == 0:
            return {}
        if n == 1:
            return {asset_names[0]: 1.0}

        mu = np.array([expected_returns.get(name, 0.0) for name in asset_names])
        time_weight = np.array([expected_returns.get(name, 0.0) for name in asset_names])
        Sigma = covariance

        # Auto-tune risk aversion to hit risk_cap
        if self.risk_aversion is None:
            gamma = self._auto_tune_risk_aversion(mu, Sigma, asset_names)
        else:
            gamma = self.risk_aversion

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Risk cap (soft constraint via penalty in objective, or hard via constraint)
        if self.risk_cap > 0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: self.risk_cap - np.sqrt(w @ Sigma @ w),
                }
            )

        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        # Objective: minimize negative utility = -(w·μ − γ/2 w·Σ·w)
        def objective(w: np.ndarray) -> float:
            port_return = w @ mu
            port_risk = np.sqrt(w @ Sigma @ w)
            # Mean-variance utility with risk penalty
            return -(port_return - (gamma / 2.0) * port_risk**2)

        # Initial guess: equal weight
        x0 = np.ones(n) / n

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )

            weights = result.x if result.success else x0
            weights = np.clip(weights, self.min_weight, self.max_weight)
            weights = weights / weights.sum()

            self._last_opt_status = {
                "success": result.success,
                "message": result.message,
                "niter": result.nit,
                "fun": result.fun,
            }
        except Exception as exc:
            logger.warning("QP optimization failed: %s. Falling back to equal weight.", exc)
            weights = np.ones(n) / n
            self._last_opt_status = {
                "success": False,
                "message": str(exc),
                "niter": 0,
                "fun": float("nan"),
            }

        return dict(zip(asset_names, weights))

    def _auto_tune_risk_aversion(
        self, mu: np.ndarray, Sigma: np.ndarray, asset_names: list[str]
    ) -> float:
        """Auto-tune risk aversion γ to target the risk cap.

        Binary search over γ ∈ [0.5, 20] to find the value that produces
        portfolio volatility closest to self.risk_cap without exceeding it.

        Args:
            mu: (N,) expected returns.
            Sigma: (N, N) covariance.
            asset_names: Asset names.

        Returns:
            gamma: Risk aversion coefficient.
        """
        n = len(mu)
        x0 = np.ones(n) / n
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        def _solve_for_gamma(gamma: float) -> float:
            def obj(w):
                return -(w @ mu - (gamma / 2.0) * w @ Sigma @ w)

            res = minimize(
                obj,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=[eq_constraint],
                options={"maxiter": 300, "ftol": 1e-8},
            )
            w = res.x if res.success else x0
            w = np.clip(w, self.min_weight, self.max_weight)
            w = w / w.sum()
            return float(np.sqrt(w @ Sigma @ w))

        target = self.risk_cap
        low, high = 0.1, 50.0
        best_gamma = 2.0

        for _ in range(15):
            mid = (low + high) / 2.0
            vol = _solve_for_gamma(mid)
            if np.isnan(vol):
                high = mid
                continue
            if vol > target * 1.02:
                low = mid  # need more risk aversion
            elif vol < target * 0.98:
                high = mid  # need less risk aversion
            else:
                best_gamma = mid
                break
            best_gamma = mid

        return best_gamma

    # ── Full Pipeline ───────────────────────────────────────────────

    def optimize(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        ir_window: int = 252,
        annualize: bool = True,
    ) -> HuataiResult:
        """Run the full 4-phase optimization pipeline.

        Args:
            signals: (T, N) raw signal scores per asset.
            returns: (T, N) forward or contemporaneous returns per asset.
            prices: (T, N) price series per asset for HP filtering.
            ir_window: Rolling window for IR computation.
            annualize: If True, annualize expected returns and volatility.

        Returns:
            HuataiResult with optimized weights and diagnostics.
        """
        constraint_violations: list[str] = []

        # Validate inputs
        if signals.empty or prices.empty:
            return HuataiResult(
                weights={},
                expected_returns={},
                status="failed",
                message="Empty input data.",
            )

        # Align assets across inputs
        common_assets = list(set(signals.columns) & set(returns.columns) & set(prices.columns))
        if len(common_assets) < 2:
            return HuataiResult(
                weights=dict.fromkeys(common_assets, 1.0),
                expected_returns=dict.fromkeys(common_assets, 0.0),
                status="failed",
                message=f"Need >= 2 common assets, got {len(common_assets)}.",
            )

        # ── Phase 1: IR-Weighted Signal Synthesis ──
        ir_signals = self._ir_weighted_signals(signals, returns, window=ir_window)
        logger.info(
            "Phase 1 (IR weights): %d assets with non-zero IR.",
            sum(1 for v in ir_signals.values() if v > 0),
        )

        # ── Phase 2: HP-Filtered Return Forecast ──
        exp_returns = self._hp_expected_returns(prices, annualize=annualize)
        hp_trends = getattr(self, "_hp_trends", None)
        logger.info("Phase 2 (HP forecast): %d assets processed.", len(exp_returns))

        # ── Phase 3: Covariance Estimation ──
        if returns.shape[1] >= 2:
            cov = self._estimate_covariance(returns, use_shrinkage=True)
            cov_assets = getattr(self, "_cov_assets", common_assets)
        else:
            cov = np.eye(len(common_assets)) * 0.04
            cov_assets = common_assets

        logger.info("Phase 3 (Covariance): estimated %d × %d matrix.", len(cov), len(cov))

        # ── Phase 4: Quadratic Programming ──
        # Ensure asset name alignment
        aligned_assets = [a for a in cov_assets if a in common_assets and a in exp_returns]
        if len(aligned_assets) < 2:
            aligned_assets = common_assets
            cov = cov[: len(aligned_assets), : len(aligned_assets)]

        # Filter expected returns and covariance to aligned assets
        aligned_mu = {a: exp_returns.get(a, 0.0) for a in aligned_assets}
        n_aligned = len(aligned_assets)
        aligned_cov = cov[:n_aligned, :n_aligned]

        weights = self._quadratic_program(aligned_mu, aligned_cov, aligned_assets)
        opt_status = getattr(self, "_last_opt_status", {"success": True, "niter": 0})

        logger.info("Phase 4 (QP): solved in %d iterations.", opt_status.get("niter", 0))

        # Compute portfolio metrics
        w_vec = np.array([weights.get(a, 0.0) for a in aligned_assets])
        mu_vec = np.array([aligned_mu.get(a, 0.0) for a in aligned_assets])

        port_return = float(w_vec @ mu_vec)
        port_risk = float(np.sqrt(w_vec @ aligned_cov @ w_vec))
        port_sharpe = (port_return - self.risk_free) / port_risk if port_risk > 1e-12 else 0.0

        # Check constraints
        if port_risk > self.risk_cap * 1.01:
            constraint_violations.append(f"Risk {port_risk:.4f} exceeds cap {self.risk_cap:.4f}")
        if max(weights.values(), default=0) > self.max_weight * 1.01:
            constraint_violations.append(
                f"Max weight {max(weights.values(), default=0):.4f} exceeds cap {self.max_weight}"
            )

        status = (
            "optimal"
            if opt_status.get("success", False) and not constraint_violations
            else "suboptimal"
        )
        if not opt_status.get("success", False):
            status = "failed"
            constraint_violations.append(opt_status.get("message", "unknown error"))

        return HuataiResult(
            weights=weights,
            expected_returns=aligned_mu,
            covariance=aligned_cov,
            portfolio_return=port_return,
            portfolio_risk=port_risk,
            portfolio_sharpe=port_sharpe,
            ir_signal=ir_signals,
            hp_trend=hp_trends,
            status=status,
            message=opt_status.get("message", ""),
            n_iterations=opt_status.get("niter", 0),
            constraint_violations=constraint_violations,
        )


def run_huatai_pipeline(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    risk_cap: float = DEFAULT_RISK_CAP,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    ir_window: int = 252,
    hp_lambda: float = HP_LAMBDA_DAILY,
) -> HuataiResult:
    """Convenience function to run the full pipeline.

    Args:
        signals: (T, N) raw signal scores per asset.
        returns: (T, N) forward returns per asset.
        prices: (T, N) price series per asset.
        risk_cap: Annualized volatility cap.
        max_weight: Maximum weight per asset.
        min_weight: Minimum weight per asset.
        ir_window: Rolling window for IR computation.
        hp_lambda: Smoothness parameter for HP filter.

    Returns:
        HuataiResult with optimized weights and diagnostics.
    """
    pipeline = HuataiPipeline(
        risk_cap=risk_cap,
        max_weight=max_weight,
        min_weight=min_weight,
        hp_lambda=hp_lambda,
    )
    return pipeline.optimize(signals, returns, prices, ir_window=ir_window)
