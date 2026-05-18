"""
Block B11: Credit Default Swap (CDS) Pricing.

Market-implied default probability and credit risk modeling from CDS spreads.
CDS spreads reflect the market's collective assessment of default risk —
a leading macro indicator for equity stress.

Models:
    CDSPricer: Bootstraps hazard rates from CDS spreads, computes survival
        probabilities, risky PV, and par spread validation.
    CDSCurve: Multi-tenor CDS term structure with implied forward default
        probabilities and hazard rate term structure.

Usage:
    >>> cds = CDSPricer(recovery_rate=0.40)
    >>> cds.fit(spreads={1: 0.005, 3: 0.008, 5: 0.012}, risk_free={1: 0.03, 3: 0.035, 5: 0.04})
    >>> cds.survival_prob(5)
    >>> cds.forward_default_prob(1, 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class CDSSpread:
    maturity: float
    spread: float
    hazard_rate: float
    survival_prob: float
    risky_pv01: float
    default_leg_pv: float
    premium_leg_pv: float
    par_spread: float

    def to_dict(self) -> Dict:
        return {
            "maturity": self.maturity,
            "spread_bp": round(self.spread * 10000, 1),
            "hazard_rate": round(self.hazard_rate, 6),
            "survival_prob": round(self.survival_prob, 4),
            "risky_pv01": round(self.risky_pv01, 4),
            "default_leg_pv": round(self.default_leg_pv, 6),
            "premium_leg_pv": round(self.premium_leg_pv, 6),
            "par_spread_bp": round(self.par_spread * 10000, 1),
        }


@dataclass
class CDSCurveResult:
    tenors: np.ndarray
    spreads: np.ndarray
    hazard_rates: np.ndarray
    survival_probs: np.ndarray
    default_probs: np.ndarray
    recovery_rate: float
    segments: List[CDSSpread] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "tenors": self.tenors.tolist(),
            "spreads_bp": (self.spreads * 10000).round(1).tolist(),
            "hazard_rates_pct": (self.hazard_rates * 100).round(4).tolist(),
            "survival_probs": self.survival_probs.round(4).tolist(),
            "default_probs_pct": (self.default_probs * 100).round(2).tolist(),
            "recovery_rate": round(self.recovery_rate, 2),
        }

    def survival_prob(self, t: float) -> float:
        cutoff = np.searchsorted(self.tenors, t)
        if cutoff == 0:
            return 1.0
        if cutoff >= len(self.tenors):
            t_n, h_n = self.tenors[-1], self.hazard_rates[-1]
        else:
            t_n = self.tenors[cutoff - 1]
            t_next = self.tenors[cutoff]
            h_n = self.hazard_rates[cutoff - 1]
            h_next = self.hazard_rates[cutoff]
            w = (t - t_n) / (t_next - t_n) if t_next > t_n else 0.0
            h = h_n + w * (h_next - h_n)
            return float(np.exp(-h * t))
        return float(np.exp(-h_n * t))

    def forward_default_prob(self, t1: float, t2: float) -> float:
        s1 = self.survival_prob(t1)
        s2 = self.survival_prob(t2)
        if s1 <= 0:
            return 1.0
        return float(max(0.0, 1.0 - s2 / s1))

    @property
    def five_year_default_prob(self) -> float:
        return float(1.0 - self.survival_prob(5.0))


class CDSPricer:
    """Credit Default Swap pricing via hazard rate bootstrapping.

    Bootstraps piecewise-constant hazard rates from market CDS spreads
    assuming quarterly premium payments and continuous accrual.

    dP/dt = −λ(t)·P(t)  ⟹  P(t) = exp(−∫₀ᵗ λ(s) ds)

    CDS spread S satisfies:
        S * Σ Δt_i * P(t_i) * D(t_i) + accrual = (1−R) * Σ [P(t_{i−1}) − P(t_i)] * D(t_i)

    Args:
        recovery_rate: Recovery rate upon default (0.0–1.0, default 0.40).
        frequency: Premium payments per year (default 4 = quarterly).
        day_count: Day-count fraction per period (default 0.25 for quarterly).
    """

    def __init__(
        self,
        recovery_rate: float = 0.40,
        frequency: int = 4,
        day_count: float = 0.25,
    ):
        self.recovery_rate = recovery_rate
        self.frequency = frequency
        self.day_count = day_count
        self._curve: Optional[CDSCurveResult] = None

    def fit(
        self,
        spreads: Dict[float, float],
        risk_free: Optional[Dict[float, float]] = None,
    ) -> CDSCurveResult:
        """Bootstrap hazard rates from CDS spread term structure.

        Args:
            spreads: Dict mapping maturity (years) → CDS spread (decimal, e.g., 0.005 = 50bp).
            risk_free: Dict mapping maturity → risk-free rate. If None, assumes flat 3%.

        Returns:
            CDSCurveResult with bootstrapped hazard rate curve.
        """
        tenors = np.array(sorted(spreads.keys()))
        spreads_arr = np.array([spreads[t] for t in tenors])
        recovery = self.recovery_rate
        lgd = 1.0 - recovery

        if risk_free is None:
            risk_free_vals = np.full_like(tenors, 0.03)
            risk_free_tenors = tenors.copy()
        else:
            r_tenors = sorted(risk_free.keys())
            risk_free_vals = np.array([risk_free[t] for t in r_tenors])
            risk_free_tenors = np.array(list(r_tenors))

        r_interp = interp1d(
            risk_free_tenors,
            risk_free_vals,
            kind="linear",
            fill_value=(risk_free_vals[0], risk_free_vals[-1]),
            bounds_error=False,
        )

        hazard_rates = np.zeros(len(tenors))
        survival_probs = np.ones(len(tenors))
        segments: List[CDSSpread] = []

        for i, (T, S) in enumerate(zip(tenors, spreads_arr)):
            r = float(r_interp(T))

            # Payment schedule: quarterly from 0 to T
            n_payments = int(round(T * self.frequency))
            payment_times = np.linspace(self.day_count, T, n_payments)

            if i == 0:
                prev_T = 0.0
                prev_h = 0.0
                prev_survival = 1.0
            else:
                prev_T = tenors[i - 1]
                prev_h = hazard_rates[i - 1]
                prev_survival = survival_probs[i - 1]

            # Bootstrap: define cumulative survival as function of hazard h
            # P(t) = prev_survival * exp(−h * (t − prev_T)) for t > prev_T
            def survival(t: float) -> float:
                if t <= prev_T:
                    if i == 0:
                        return 1.0
                    return prev_survival * np.exp(-hazard_rates[i - 1] * max(t - tenors[i - 2], 0))
                return prev_survival * np.exp(-h * (t - prev_T))

            def premium_leg(h: float) -> float:
                pv = 0.0
                cf = S * self.day_count
                for t_pay in payment_times:
                    pv += cf * survival(t_pay) * np.exp(-r * t_pay)
                return pv

            def default_leg(h: float) -> float:
                pv = 0.0
                prev_p = prev_survival
                t_prev = prev_T
                step = 1.0 / 365.0
                t = prev_T + step
                while t <= T:
                    p = survival(t)
                    dp = max(prev_p - p, 0.0)
                    pv += dp * np.exp(-r * t)
                    prev_p = p
                    t_prev = t
                    t += step
                return pv

            h = prev_h if prev_h > 0 else 0.001

            def objective(h_val: float) -> float:
                return premium_leg(h_val) - lgd * default_leg(h_val)

            try:
                f_lo = objective(0.0001)
                f_hi = objective(0.50)
                if f_lo * f_hi > 0:
                    h = 0.0001 if abs(f_lo) < abs(f_hi) else 0.50
                else:
                    h = brentq(objective, 0.0001, 0.50, xtol=1e-8, maxiter=100)
            except (ValueError, RuntimeError):
                h = prev_h if prev_h > 0 else S / lgd

            hazard_rates[i] = h
            survival_probs[i] = prev_survival * np.exp(-h * (T - prev_T))

            segments.append(
                CDSSpread(
                    maturity=T,
                    spread=S,
                    hazard_rate=h,
                    survival_prob=float(survival_probs[i]),
                    risky_pv01=float(premium_leg(h) / S if S > 0 else 0.0),
                    default_leg_pv=float(default_leg(h)),
                    premium_leg_pv=float(premium_leg(h)),
                    par_spread=float(
                        lgd * default_leg(h) / premium_leg(1.0) if premium_leg(1.0) > 0 else 0.0
                    ),
                )
            )

        self._curve = CDSCurveResult(
            tenors=tenors,
            spreads=spreads_arr,
            hazard_rates=hazard_rates,
            survival_probs=survival_probs,
            default_probs=1.0 - survival_probs,
            recovery_rate=recovery,
            segments=segments,
        )
        return self._curve

    @staticmethod
    def price_cds(
        spread: float,
        maturity: float,
        risk_free_rate: float,
        recovery_rate: float = 0.40,
        frequency: int = 4,
        day_count: float = 0.25,
    ) -> Dict:
        """Quick CDS pricing with flat hazard rate assumption.

        Under constant hazard rate λ = S / (1−R), prices both legs.

        Args:
            spread: CDS spread (decimal).
            maturity: Contract maturity in years.
            risk_free_rate: Risk-free rate (decimal).
            recovery_rate: Recovery rate.
            frequency: Premium payments per year.
            day_count: Day count fraction.

        Returns:
            Dictionary with price, hazard rate, survival prob, default prob.
        """
        lgd = 1.0 - recovery_rate
        h = spread / lgd if lgd > 0 else 0.001
        survival = np.exp(-h * maturity)
        default_prob = 1.0 - survival

        n = int(round(maturity * frequency))
        payment_times = np.linspace(day_count, maturity, n)
        prem_pv = spread * day_count * np.sum(np.exp(-(h + risk_free_rate) * payment_times))

        def_pv = 0.0
        step = 1.0 / 365.0
        t = step
        prev_s = 1.0
        while t <= maturity:
            s = np.exp(-h * t)
            def_pv += max(prev_s - s, 0) * np.exp(-risk_free_rate * t)
            prev_s = s
            t += step
        def_pv *= lgd

        return {
            "spread_bp": round(spread * 10000, 1),
            "hazard_rate": round(h, 6),
            "survival_prob": round(survival, 4),
            "default_prob": round(default_prob, 4),
            "premium_leg_pv": round(prem_pv, 6),
            "default_leg_pv": round(def_pv, 6),
            "fair_value": round(def_pv - prem_pv, 6),
        }

    def default_probability(
        self,
        spread: float,
        maturity: float,
        recovery_rate: Optional[float] = None,
    ) -> float:
        """Market-implied probability of default (risk-neutral).

        P(default by T) = 1 − exp(−λ·T) where λ ≈ S / (1−R)

        Args:
            spread: CDS spread (decimal).
            maturity: Time horizon in years.
            recovery_rate: Recovery rate (uses instance default if None).

        Returns:
            Implied default probability (0–1).
        """
        rr = recovery_rate if recovery_rate is not None else self.recovery_rate
        lgd = 1.0 - rr
        h = spread / lgd if lgd > 0 else 0.0
        return float(1.0 - np.exp(-h * maturity))

    def implied_spread(
        self,
        default_prob: float,
        maturity: float,
        recovery_rate: Optional[float] = None,
    ) -> float:
        """Inverse: compute implied CDS spread from default probability.

        Args:
            default_prob: Target default probability.
            maturity: Time horizon.
            recovery_rate: Recovery rate.

        Returns:
            Implied CDS spread (decimal).
        """
        if default_prob >= 1.0:
            return 1.0
        rr = recovery_rate if recovery_rate is not None else self.recovery_rate
        lgd = 1.0 - rr
        h = -np.log(1.0 - default_prob) / maturity
        return float(h * lgd)

    @property
    def curve(self) -> Optional[CDSCurveResult]:
        return self._curve

    def credit_risk_features(
        self,
        spreads: Dict[float, float],
        risk_free: Optional[Dict[float, float]] = None,
    ) -> Dict:
        """Extract credit risk features from CDS term structure.

        Args:
            spreads: CDS spread curve.
            risk_free: Risk-free rate curve.

        Returns:
            Dictionary of credit risk features for ML pipelines.
        """
        curve = self.fit(spreads, risk_free)
        features = {
            "cds_1y_bp": float(curve.spreads[0] * 10000) if len(curve.spreads) > 0 else np.nan,
            "cds_5y_bp": float(curve.spreads[-1] * 10000) if len(curve.spreads) > 0 else np.nan,
            "cds_slope_1y5y": float(curve.spreads[-1] - curve.spreads[0])
            if len(curve.spreads) >= 2
            else np.nan,
            "hazard_rate_1y": float(curve.hazard_rates[0])
            if len(curve.hazard_rates) > 0
            else np.nan,
            "hazard_rate_5y": float(curve.hazard_rates[-1])
            if len(curve.hazard_rates) > 0
            else np.nan,
            "five_year_default_prob": curve.five_year_default_prob,
            "default_prob_1y": float(1.0 - curve.survival_prob(1.0)),
            "default_prob_3y": float(1.0 - curve.survival_prob(3.0)),
            "forward_prob_1y_3y": curve.forward_default_prob(1.0, 3.0),
            "hazard_slope": float(curve.hazard_rates[-1] - curve.hazard_rates[0])
            if len(curve.hazard_rates) >= 2
            else np.nan,
            "recovery_rate": curve.recovery_rate,
        }
        return features

    def credit_risk_regime(
        self,
        spread_5y: float,
        spread_change_30d: float = 0.0,
        spread_change_90d: float = 0.0,
    ) -> Dict:
        """Classify credit risk regime from CDS spread levels and changes.

        Args:
            spread_5y: Current 5-year CDS spread (decimal).
            spread_change_30d: 30-day spread change (decimal).
            spread_change_90d: 90-day spread change (decimal).

        Returns:
            Dictionary with regime classification and risk metrics.
        """
        sp_bp = spread_5y * 10000
        chg_bp_30d = spread_change_30d * 10000
        chg_bp_90d = spread_change_90d * 10000

        # Level-based regime
        if sp_bp <= 50:
            level_regime = "TIGHT"
            level_mult = 1.0
        elif sp_bp <= 150:
            level_regime = "NORMAL"
            level_mult = 0.95
        elif sp_bp <= 500:
            level_regime = "WIDE"
            level_mult = 0.75
        else:
            level_regime = "CRISIS"
            level_mult = 0.40

        # Momentum-based signal
        if chg_bp_90d > 100:
            momentum = "WIDENING_RAPIDLY"
            momentum_mult = 0.60
        elif chg_bp_30d > 30:
            momentum = "WIDENING"
            momentum_mult = 0.80
        elif chg_bp_30d < -30:
            momentum = "TIGHTENING"
            momentum_mult = 1.15
        elif chg_bp_90d < -50:
            momentum = "TIGHTENING_STEADILY"
            momentum_mult = 1.25
        else:
            momentum = "STABLE"
            momentum_mult = 1.0

        combined_mult = level_mult * momentum_mult

        return {
            "level_regime": level_regime,
            "momentum_regime": momentum,
            "spread_bp": round(sp_bp, 1),
            "spread_change_30d_bp": round(chg_bp_30d, 1),
            "spread_change_90d_bp": round(chg_bp_90d, 1),
            "risk_multiplier": round(combined_mult, 2),
            "default_prob_5y": self.default_probability(spread_5y, 5.0),
        }
