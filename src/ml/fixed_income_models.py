"""
Block B: Fixed Income Models — B1 Nelson-Siegel, B4 Credit Spreads, B5 Rate Models.

Yield curve decomposition and interest rate dynamics for macro regime
detection and bond market signal generation.

Models:
  NelsonSiegel: Decomposes yield curve into level (β0), slope (β1),
    and curvature (β2) factors. Fitted via nonlinear least squares (scipy).
    The three factors explain 95%+ of Treasury yield variation.
  CreditSpreadGate: IG/OAS vs HY/OAS spread as risk appetite signal.
    Tightening spreads = risk-on, widening = risk-off.
  VasicekModel: Mean-reverting Ornstein-Uhlenbeck short rate process.
    dr = κ(θ − r)dt + σdW. Yields closed-form bond prices.
  CIRModel: Cox-Ingersoll-Ross square-root diffusion.
    dr = κ(θ − r)dt + σ√r dW. Prevents negative rates.

Usage:
    >>> ns = NelsonSiegel()
    >>> ns.fit(maturities=[1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30],
    ...        yields=[4.5, 4.6, 4.7, 4.8, 5.0, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5])
    >>> print(ns.level, ns.slope_factor, ns.curvature)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ── B1: Nelson-Siegel Yield Curve Decomposition ────────────────────────


@dataclass
class NelsonSiegelResult:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    fitted_yields: np.ndarray
    rmse: float
    r_squared: float
    convergence: bool

    @property
    def level(self) -> float:
        return self.beta0

    @property
    def slope(self) -> float:
        """Negative of beta1 = 10Y − 3M spread proxy."""
        return -self.beta1

    @property
    def curvature(self) -> float:
        return self.beta2

    def to_dict(self) -> Dict:
        return {
            "beta0": round(self.beta0, 4),
            "beta1": round(self.beta1, 4),
            "beta2": round(self.beta2, 4),
            "tau": round(self.tau, 4),
            "rmse_bp": round(self.rmse * 100, 2),
            "r_squared": round(self.r_squared, 4),
            "level": round(self.level, 4),
            "slope": round(self.slope, 4),
            "curvature": round(self.curvature, 4),
        }


class NelsonSiegel:
    """Nelson-Siegel yield curve model.

    The Nelson-Siegel formula expresses yield y(m) for maturity m as:
        y(m) = β0 + β1·(1−e^{−m/τ})/(m/τ) + β2·[(1−e^{−m/τ})/(m/τ) − e^{−m/τ}]

    Interpretation:
        β0: Long-term level (asymptote at infinite maturity).
        −β1: Slope (spread between long and short rates).
        β2: Curvature (hump/trough at medium maturities).
        τ: Decay factor controlling hump location.

    Args:
        init_beta0, init_beta1, init_beta2, init_tau: Initial parameter guesses.
        maxiter: Maximum optimizer iterations.
    """

    def __init__(
        self,
        init_beta0: float = 3.0,
        init_beta1: float = -1.0,
        init_beta2: float = 1.0,
        init_tau: float = 2.0,
        maxiter: int = 1000,
    ):
        self.init_params = np.array([init_beta0, init_beta1, init_beta2, init_tau])
        self.maxiter = maxiter
        self._result: Optional[NelsonSiegelResult] = None

    @staticmethod
    def _ns_yield(maturities: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute yields for given maturities and parameters."""
        beta0, beta1, beta2, tau = params
        tau = max(tau, 0.01)
        m_tau = maturities / tau
        exp_mt = np.exp(-m_tau)
        factor1 = np.where(m_tau < 1e-10, 1.0, (1.0 - exp_mt) / m_tau)
        factor2 = factor1 - exp_mt
        return beta0 + beta1 * factor1 + beta2 * factor2

    def fit(
        self,
        maturities: np.ndarray,
        yields: np.ndarray,
    ) -> NelsonSiegelResult:
        """Fit NS model to observed yields.

        Args:
            maturities: Array of maturities in years.
            yields: Observed yields in percent (e.g., 4.5 = 4.5%).

        Returns:
            NelsonSiegelResult with fitted parameters and diagnostics.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)
        valid = ~np.isnan(yields)
        maturities = maturities[valid]
        yields = yields[valid]

        if len(maturities) < 4:
            return NelsonSiegelResult(
                beta0=np.nan,
                beta1=np.nan,
                beta2=np.nan,
                tau=np.nan,
                fitted_yields=np.full_like(yields, np.nan),
                rmse=np.nan,
                r_squared=np.nan,
                convergence=False,
            )

        def objective(params: np.ndarray) -> float:
            return np.sum((yields - self._ns_yield(maturities, params)) ** 2)

        bounds = [
            (None, None),  # beta0
            (None, None),  # beta1
            (None, None),  # beta2
            (0.1, 30.0),  # tau > 0
        ]

        try:
            res = minimize(
                objective,
                self.init_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.maxiter},
            )
            converged = res.success
            beta0, beta1, beta2, tau = res.x
        except Exception:
            converged = False
            beta0, beta1, beta2, tau = self.init_params

        fitted = self._ns_yield(maturities, np.array([beta0, beta1, beta2, tau]))
        ss_res = np.sum((yields - fitted) ** 2)
        ss_tot = np.sum((yields - np.mean(yields)) ** 2)
        rmse = np.sqrt(ss_res / len(yields))
        r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        result = NelsonSiegelResult(
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
            fitted_yields=fitted,
            rmse=rmse,
            r_squared=r_sq,
            convergence=converged,
        )
        self._result = result
        return result

    def forecast_yield(self, maturity: float) -> float:
        """Forecast yield at any maturity using fitted parameters.

        Args:
            maturity: Maturity in years.

        Returns:
            Forecast yield in percent.
        """
        if self._result is None:
            raise ValueError("Call fit() first")
        params = np.array(
            [
                self._result.beta0,
                self._result.beta1,
                self._result.beta2,
                self._result.tau,
            ]
        )
        return float(self._ns_yield(np.array([maturity]), params)[0])

    def rolling_fit(
        self,
        yield_df: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Fit NS model at each date in a yield curve DataFrame.

        Args:
            yield_df: DataFrame with date column and maturity columns.
            date_col: Name of date column.

        Returns:
            DataFrame with date, beta0, beta1, beta2, tau, rmse per row.
        """
        mat_cols = [c for c in yield_df.columns if c != date_col]
        maturities = np.array(
            [float(c.replace("Y", "").replace("M", "/12").replace("m", "")) for c in mat_cols]
        )

        rows = []
        for _, row in yield_df.iterrows():
            yields = row[mat_cols].values.astype(float)
            r = self.fit(maturities, yields)
            rows.append(
                {
                    "date": row[date_col],
                    "beta0": r.beta0,
                    "beta1": r.beta1,
                    "beta2": r.beta2,
                    "tau": r.tau,
                    "level": r.level,
                    "slope": r.slope,
                    "curvature": r.curvature,
                    "rmse": r.rmse,
                }
            )

        return pd.DataFrame(rows)

    @property
    def level(self) -> float:
        if self._result is None:
            return np.nan
        return self._result.level

    @property
    def slope_factor(self) -> float:
        if self._result is None:
            return np.nan
        return self._result.slope

    @property
    def curvature(self) -> float:
        if self._result is None:
            return np.nan
        return self._result.curvature


# ── B4: Credit Spread Macro Signal ─────────────────────────────────────


@dataclass
class CreditSpreadResult:
    ig_yield: float
    hy_yield: float
    spread: float
    spread_percentile: float
    regime: str  # "TIGHT", "NORMAL", "WIDE", "CRISIS"
    risk_multiplier: float

    def to_dict(self) -> Dict:
        return {
            "ig_yield": round(self.ig_yield, 2),
            "hy_yield": round(self.hy_yield, 2),
            "spread_bp": round(self.spread * 100, 0),
            "spread_pct": round(self.spread_percentile, 2),
            "regime": self.regime,
            "risk_mult": round(self.risk_multiplier, 2),
        }


class CreditSpreadGate:
    """Credit spread macro risk appetite gate.

    Measures the spread between investment-grade and high-yield bond yields
    as a real-time risk appetite indicator. Credit spreads widen during stress
    and tighten during risk-on environments — a leading equity signal.

    Regime classification (percentile-based):
        TIGHT (< 25th pct): risk-on → 1.0x multiplier
        NORMAL (25–75th pct): neutral → 1.0x
        WIDE (75–95th pct): risk-off → 0.75x
        CRISIS (> 95th pct): extreme → 0.50x

    Args:
        lookback: Rolling window for percentile calibration (days).
        crisis_pct: Percentile threshold for CRISIS regime.
        wide_pct: Percentile threshold for WIDE regime.
    """

    def __init__(
        self,
        lookback: int = 504,
        crisis_pct: float = 95.0,
        wide_pct: float = 75.0,
    ):
        self.lookback = lookback
        self.crisis_pct = crisis_pct
        self.wide_pct = wide_pct
        self._spread_history: List[float] = []

    def update(self, ig_yield: float, hy_yield: float) -> CreditSpreadResult:
        """Update with latest IG and HY yields and return regime signal.

        Args:
            ig_yield: Investment-grade yield (e.g., BofA US Corporate).
            hy_yield: High-yield yield (e.g., BofA US High Yield).

        Returns:
            CreditSpreadResult with regime and risk multiplier.
        """
        spread = max(hy_yield - ig_yield, 0.0)
        self._spread_history.append(spread)

        window = (
            self._spread_history[-self.lookback :]
            if len(self._spread_history) >= self.lookback
            else self._spread_history
        )
        pct = self._compute_percentile(spread, window) if len(window) >= 20 else 50.0

        if pct > self.crisis_pct:
            regime, mult = "CRISIS", 0.50
        elif pct > self.wide_pct:
            regime, mult = "WIDE", 0.75
        elif pct < 25.0:
            regime, mult = "TIGHT", 1.0
        else:
            regime, mult = "NORMAL", 1.0

        return CreditSpreadResult(
            ig_yield=ig_yield,
            hy_yield=hy_yield,
            spread=spread,
            spread_percentile=pct,
            regime=regime,
            risk_multiplier=mult,
        )

    @staticmethod
    def _compute_percentile(value: float, history: List[float]) -> float:
        arr = np.array(history)
        return float(np.mean(arr <= value) * 100.0)

    def get_spread_percentile(self, spread: float) -> float:
        """Get current spread's percentile rank in history."""
        if len(self._spread_history) < 20:
            return 50.0
        window = self._spread_history[-self.lookback :]
        return self._compute_percentile(spread, window)


# ── B5: Vasicek & CIR Interest Rate Models ─────────────────────────────


@dataclass
class RateModelResult:
    model: str
    kappa: float
    theta: float
    sigma: float
    r0: float
    half_life: float
    convergence: bool
    rmse: float
    log_likelihood: float

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "kappa": round(self.kappa, 4),
            "theta": round(self.theta, 4),
            "sigma": round(self.sigma, 4),
            "r0": round(self.r0, 4),
            "half_life_days": round(self.half_life, 1),
            "convergence": self.convergence,
            "rmse_bp": round(self.rmse * 100, 1),
            "log_likelihood": round(self.log_likelihood, 2),
        }


class VasicekModel:
    """Vasicek (Ornstein-Uhlenbeck) short rate model.

    SDE: dr_t = κ(θ − r_t)dt + σ dW_t

    r_t mean-reverts to long-run level θ at speed κ.
    σ is the instantaneous volatility.
    Half-life of shocks = ln(2) / κ.

    Args:
        annualize: Scale estimates to annual (True) or keep daily.
    """

    def __init__(self, annualize: bool = True):
        self.annualize = annualize
        self._result: Optional[RateModelResult] = None

    def fit(self, rates: np.ndarray, dt: float = 1 / 252) -> RateModelResult:
        """Calibrate Vasicek model via OLS on discretized SDE.

        Args:
            rates: Array of short rate observations.
            dt: Time step in years (default 1/252 for daily).

        Returns:
            RateModelResult with calibrated parameters.
        """
        rates = np.asarray(rates, dtype=float)
        r_t = rates[:-1]
        r_t1 = rates[1:]
        n = len(r_t)

        if n < 10:
            return RateModelResult(
                model="Vasicek",
                kappa=np.nan,
                theta=np.nan,
                sigma=np.nan,
                r0=float(rates[-1]) if len(rates) > 0 else np.nan,
                half_life=np.nan,
                convergence=False,
                rmse=np.nan,
                log_likelihood=np.nan,
            )

        # OLS: r_{t+1} - r_t = a + b*r_t + ε
        y = r_t1 - r_t
        X = np.column_stack([np.ones(n), r_t])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            a, b = beta[0], beta[1]
        except np.linalg.LinAlgError:
            return RateModelResult(
                model="Vasicek",
                kappa=np.nan,
                theta=np.nan,
                sigma=np.nan,
                r0=float(rates[-1]),
                half_life=np.nan,
                convergence=False,
                rmse=np.nan,
                log_likelihood=np.nan,
            )

        # Back out parameters
        kappa = -b / dt if abs(b) > 1e-12 else 0.1
        theta = -a / b if abs(b) > 1e-12 else float(np.mean(rates))
        eta = y - (a + b * r_t)
        sigma_eps = np.std(eta, ddof=2)
        sigma = sigma_eps / np.sqrt(dt)

        # Predictions
        pred = r_t + (a + b * r_t)
        rmse = float(np.sqrt(np.mean((r_t1 - pred) ** 2)))
        ll = float(np.sum(norm.logpdf(r_t1, loc=pred, scale=sigma_eps)))

        half_life = np.log(2) / kappa if kappa > 0 else float("inf")
        half_life_days = half_life * 252 if self.annualize else half_life

        result = RateModelResult(
            model="Vasicek",
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            r0=float(rates[-1]),
            half_life=half_life_days,
            convergence=True,
            rmse=rmse,
            log_likelihood=ll,
        )
        self._result = result
        return result

    def forecast(self, horizon: int = 21) -> float:
        """Forecast rate at horizon days.

        E[r_T] = θ + (r_0 − θ)e^{−κT}
        """
        if self._result is None:
            raise ValueError("Call fit() first")
        T = horizon / 252.0 if self.annualize else horizon
        kappa = self._result.kappa
        theta = self._result.theta
        r0 = self._result.r0
        return float(theta + (r0 - theta) * np.exp(-kappa * T))

    def zero_coupon_price(self, maturity: float) -> float:
        """Price a zero-coupon bond with given maturity (years).

        P(t,T) = A(t,T) * exp(−B(t,T) * r_t)
        """
        if self._result is None:
            raise ValueError("Call fit() first")
        k, th, sig = self._result.kappa, self._result.theta, self._result.sigma
        B = (1.0 - np.exp(-k * maturity)) / k if k > 1e-12 else maturity
        A = (
            np.exp((th - sig**2 / (2 * k**2)) * (B - maturity) - (sig**2 * B**2) / (4 * k))
            if k > 1e-12
            else np.exp(0.0)
        )
        return float(A * np.exp(-B * self._result.r0))

    @property
    def result(self) -> Optional[RateModelResult]:
        return self._result


class CIRModel:
    """Cox-Ingersoll-Ross square-root diffusion model.

    SDE: dr_t = κ(θ − r_t)dt + σ√r_t dW_t

    Ensures rates stay non-negative when 2κθ ≥ σ² (Feller condition).
    Closed-form bond prices unlike Vasicek with a volatility adjustment.

    Args:
        annualize: Scale estimates to annual (True) or keep daily.
    """

    def __init__(self, annualize: bool = True):
        self.annualize = annualize
        self._result: Optional[RateModelResult] = None

    def fit(self, rates: np.ndarray, dt: float = 1 / 252) -> RateModelResult:
        """Calibrate CIR model via OLS on transformed SDE.

        Transformation: Δr_t / √r_t = κθ/√r_t − κ√r_t + ε_t
        """
        rates = np.asarray(rates, dtype=float)
        r_t = rates[:-1]
        r_t1 = rates[1:]
        n = len(r_t)
        mask = r_t > 1e-10
        r_t_pos = r_t[mask]
        r_t1_pos = r_t1[mask]
        n_pos = len(r_t_pos)

        if n_pos < 10:
            return RateModelResult(
                model="CIR",
                kappa=np.nan,
                theta=np.nan,
                sigma=np.nan,
                r0=float(rates[-1]) if len(rates) > 0 else np.nan,
                half_life=np.nan,
                convergence=False,
                rmse=np.nan,
                log_likelihood=np.nan,
            )

        # OLS: Δr / sqrt(r) = κθ * (1/sqrt(r)) − κ * sqrt(r) + ε
        y = (r_t1_pos - r_t_pos) / np.sqrt(r_t_pos)
        X = np.column_stack([1.0 / np.sqrt(r_t_pos), np.sqrt(r_t_pos)])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            kappa_theta = beta[0]
            kappa = -beta[1]
        except np.linalg.LinAlgError:
            return RateModelResult(
                model="CIR",
                kappa=np.nan,
                theta=np.nan,
                sigma=np.nan,
                r0=float(rates[-1]),
                half_life=np.nan,
                convergence=False,
                rmse=np.nan,
                log_likelihood=np.nan,
            )

        kappa = max(kappa, 0.01) / dt
        theta = max(kappa_theta / max(-beta[1], 0.01), 0.0)

        eta = y - X @ beta
        sigma_eps = np.std(eta, ddof=2)
        sigma = sigma_eps / np.sqrt(dt)

        # For prediction: use Vasicek-like formula for now
        pred = r_t_pos + dt * (kappa * (theta - r_t_pos))
        rmse = float(np.sqrt(np.mean((r_t1_pos - pred) ** 2)))
        ll = float(np.sum(norm.logpdf(r_t1_pos, loc=pred, scale=sigma_eps)))

        half_life = np.log(2) / kappa if kappa > 0 else float("inf")
        half_life_days = half_life * 252 if self.annualize else half_life

        result = RateModelResult(
            model="CIR",
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            r0=float(rates[-1]),
            half_life=half_life_days,
            convergence=True,
            rmse=rmse,
            log_likelihood=ll,
        )
        self._result = result
        return result

    def forecast(self, horizon: int = 21) -> float:
        """Forecast rate at horizon days.

        E[r_T] = θ + (r_0 − θ)e^{−κT}
        """
        if self._result is None:
            raise ValueError("Call fit() first")
        T = horizon / 252.0 if self.annualize else horizon
        k = self._result.kappa
        th = self._result.theta
        r0 = self._result.r0
        return float(th + (r0 - th) * np.exp(-k * T))

    @property
    def result(self) -> Optional[RateModelResult]:
        return self._result

    @property
    def feller_condition(self) -> bool:
        """Check if Feller condition holds (positive rates guaranteed)."""
        if self._result is None:
            return False
        return 2 * self._result.kappa * self._result.theta >= self._result.sigma**2
