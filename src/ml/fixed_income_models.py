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


# ── B2: Bond Duration & Convexity ──────────────────────────────────────


@dataclass
class BondResult:
    price: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    current_yield: float
    coupon: float
    maturity: float
    face_value: float
    frequency: int

    def to_dict(self) -> Dict:
        return {
            "price": round(self.price, 4),
            "ytm": round(self.ytm * 100, 4),
            "macaulay_duration": round(self.macaulay_duration, 4),
            "modified_duration": round(self.modified_duration, 4),
            "convexity": round(self.convexity, 4),
            "dv01": round(self.dv01, 4),
            "current_yield": round(self.current_yield * 100, 4),
            "coupon": round(self.coupon * 100, 4),
            "maturity": round(self.maturity, 4),
            "face_value": self.face_value,
            "frequency": self.frequency,
        }


class BondPricer:
    """Fixed-income bond pricing with duration and convexity analytics.

    Computes price, yield, duration (Macaulay/modified), convexity, DV01,
    and current yield for a vanilla coupon bond. Supports Newton-Raphson
    root-finding for price→YTM inversion.

    Formulas:
        Price = Σ [c/(1+y/f)^(i)] + FV/(1+y/f)^(nf)
        MacaulayDuration = (1/P) * Σ [t_i * CF_i / (1+y/f)^(i)]
        ModifiedDuration = MacaulayDuration / (1 + y/f)
        Convexity = (1/P) * Σ [t_i*(t_i+1/f) * CF_i / (1+y/f)^(i+2)] / f^2
        DV01 = ModifiedDuration * Price / 10_000

    Args:
        face_value: Par value (default 100).
        frequency: Coupons per year (2 = semiannual, default US convention).
    """

    def __init__(self, face_value: float = 100.0, frequency: int = 2):
        self.face_value = face_value
        self.frequency = frequency

    def price(
        self,
        ytm: float,
        coupon: float,
        maturity: float,
    ) -> float:
        """Compute bond price from yield-to-maturity.

        Args:
            ytm: Yield-to-maturity as decimal (e.g., 0.05 = 5%).
            coupon: Annual coupon rate as decimal.
            maturity: Time to maturity in years.

        Returns:
            Clean price (present value of future cash flows).
        """
        periods = int(round(maturity * self.frequency))
        if periods <= 0:
            return self.face_value

        ytm_per = ytm / self.frequency
        cpn_payment = coupon * self.face_value / self.frequency
        t = np.arange(1, periods + 1) / self.frequency
        discount = 1.0 / (1.0 + ytm_per) ** np.arange(1, periods + 1)
        pv_coupons = cpn_payment * np.sum(discount)
        pv_face = self.face_value * discount[-1]
        return float(pv_coupons + pv_face)

    def compute(
        self,
        ytm: float,
        coupon: float,
        maturity: float,
    ) -> BondResult:
        """Full bond analytics: price, duration, convexity, DV01.

        Args:
            ytm: Yield-to-maturity as decimal.
            coupon: Annual coupon rate as decimal.
            maturity: Time to maturity in years.

        Returns:
            BondResult with all analytics.
        """
        periods = int(round(maturity * self.frequency))
        if periods <= 0:
            return BondResult(
                price=self.face_value,
                ytm=ytm,
                macaulay_duration=0.0,
                modified_duration=0.0,
                convexity=0.0,
                dv01=0.0,
                current_yield=coupon,
                coupon=coupon,
                maturity=maturity,
                face_value=self.face_value,
                frequency=self.frequency,
            )

        ytm_per = ytm / self.frequency
        cpn_payment = coupon * self.face_value / self.frequency
        t = np.arange(1, periods + 1) / self.frequency
        discount = 1.0 / (1.0 + ytm_per) ** np.arange(1, periods + 1)

        pv_coupons = cpn_payment * np.sum(discount)
        pv_face = self.face_value * discount[-1]
        price = float(pv_coupons + pv_face)

        weighted_t = t * cpn_payment * discount
        weighted_t[-1] += t[-1] * pv_face
        macaulay_d = float(np.sum(weighted_t) / price)

        modified_d = macaulay_d / (1.0 + ytm_per)

        f2 = self.frequency * self.frequency
        convex_t = t * (t + 1.0 / self.frequency) * cpn_payment * discount
        convex_t[-1] += t[-1] * (t[-1] + 1.0 / self.frequency) * pv_face
        convexity = float(np.sum(convex_t) / (price * (1.0 + ytm_per) ** 2 * f2))

        dv01 = modified_d * price / 10_000.0

        return BondResult(
            price=price,
            ytm=ytm,
            macaulay_duration=macaulay_d,
            modified_duration=modified_d,
            convexity=convexity,
            dv01=dv01,
            current_yield=(coupon * self.face_value / price),
            coupon=coupon,
            maturity=maturity,
            face_value=self.face_value,
            frequency=self.frequency,
        )

    def ytm_from_price(
        self,
        target_price: float,
        coupon: float,
        maturity: float,
        initial_guess: float = 0.05,
        tolerance: float = 1e-8,
        max_iter: int = 100,
    ) -> float:
        """Solve for YTM given a bond price via Newton-Raphson.

        Args:
            target_price: Observed clean price.
            coupon: Annual coupon rate as decimal.
            maturity: Time to maturity in years.
            initial_guess: Starting YTM guess (default 5%).
            tolerance: Convergence tolerance.
            max_iter: Maximum iterations.

        Returns:
            Implied YTM as decimal.
        """
        y = initial_guess
        for _ in range(max_iter):
            p = self.price(y, coupon, maturity)
            dp = (self.price(y + 1e-6, coupon, maturity) - p) / 1e-6
            if abs(dp) < 1e-15:
                break
            y_new = y - (p - target_price) / dp
            if abs(y_new - y) < tolerance:
                return float(y_new)
            y = max(y_new, -0.99)
        return float(y)

    def price_etf_proxy(
        self,
        ytm: float,
        avg_maturity: float,
        avg_coupon: float = 0.04,
    ) -> BondResult:
        """Price and analyze a bond ETF like TLT/IEF as a proxy bond.

        Treats the ETF as a single coupon bond with weighted-average
        maturity and coupon. Use for duration-based risk estimation.

        Args:
            ytm: Current yield-to-maturity.
            avg_maturity: Weighted-average maturity in years.
            avg_coupon: Weighted-average coupon rate.

        Returns:
            BondResult with duration and convexity analytics.
        """
        return self.compute(ytm=ytm, coupon=avg_coupon, maturity=avg_maturity)

    def price_impact(
        self,
        ytm: float,
        coupon: float,
        maturity: float,
        yield_change_bp: float,
    ) -> Dict:
        """Estimate price change from yield shift using duration+convexity.

        ΔP/P ≈ −ModifiedDuration * Δy + 0.5 * Convexity * (Δy)²

        Args:
            ytm: Current YTM.
            coupon: Annual coupon rate.
            maturity: Time to maturity.
            yield_change_bp: Yield change in basis points (positive = yield up).

        Returns:
            Dictionary with estimated price, delta, and percentage change.
        """
        result = self.compute(ytm, coupon, maturity)
        dy = yield_change_bp / 10_000.0
        pct_change = -result.modified_duration * dy + 0.5 * result.convexity * dy**2
        new_price = result.price * (1.0 + pct_change)
        return {
            "current_price": round(result.price, 4),
            "yield_change_bp": yield_change_bp,
            "mod_duration": round(result.modified_duration, 4),
            "convexity": round(result.convexity, 4),
            "pct_change": round(pct_change * 100, 6),
            "estimated_price": round(new_price, 4),
            "dv01_estimate": round(result.dv01 * abs(yield_change_bp), 4),
        }

    def bond_yield_series(
        self,
        prices: np.ndarray,
        coupon: float,
        maturity: float,
    ) -> np.ndarray:
        """Convert price series to YTM series via root-finding.

        Args:
            prices: Array of observed clean prices.
            coupon: Annual coupon rate.
            maturity: Time to maturity in years.

        Returns:
            Array of implied YTMs.
        """
        ytms = np.zeros(len(prices))
        for i, p in enumerate(prices):
            if i == 0:
                ytms[i] = coupon
            else:
                ytms[i] = (
                    ytms[i - 1] if p <= 0 else self.ytm_from_price(p, coupon, maturity, ytms[i - 1])
                )
            if ytms[i] < 0:
                ytms[i] = coupon
        return ytms
