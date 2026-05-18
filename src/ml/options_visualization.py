"""
Block A12: Options Payoff & Volatility Visualization.

3D volatility surfaces, theta/time decay plots, payoff diagrams, and
volatility skew/smile visualization for options analytics.

Visualizations:
    payoff_diagram(): PnL at expiration for any option combination.
    vol_surface_3d(): 3D implied volatility surface (strike × maturity × IV).
    theta_curve(): Theta decay over time for a single option position.
    skew_chart(): Volatility skew at a given maturity.
    term_structure_plot(): Volatility term structure at ATM.

Usage:
    >>> from src.ml.options_visualization import OptionsVisualizer
    >>> viz = OptionsVisualizer()
    >>> fig = viz.payoff_diagram(strategy="straddle", S_range=(80, 120), K=100, premium=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionLeg:
    option_type: str  # "call" or "put"
    strike: float
    premium: float
    position: str = "long"  # "long" or "short"
    quantity: int = 1

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        sign = 1 if self.position == "long" else -1
        if self.option_type == "call":
            intrinsic = np.maximum(S_T - self.strike, 0.0)
        else:
            intrinsic = np.maximum(self.strike - S_T, 0.0)
        return sign * self.quantity * (intrinsic - self.premium)


class OptionsVisualizer:
    """Options payoff diagram and volatility surface visualization.

    Provides static chart generation for common options strategies and
    volatility analytics. Designed for use in dashboards, reports, and
    trade analysis notebooks.

    All methods return (figure_data, metadata) dictionaries suitable
    for plotly/matplotlib rendering or JSON serialization.

    Args:
        title_prefix: Optional prefix for chart titles.
        figsize: Default figure size (width, height).
    """

    _STRATEGIES = {
        "long_call": [OptionLeg("call", 0, 0, "long")],
        "long_put": [OptionLeg("put", 0, 0, "long")],
        "short_call": [OptionLeg("call", 0, 0, "short")],
        "short_put": [OptionLeg("put", 0, 0, "short")],
        "straddle": [
            OptionLeg("call", 0, 0, "long"),
            OptionLeg("put", 0, 0, "long"),
        ],
        "strangle": [
            OptionLeg("call", 0, 0, "long"),
            OptionLeg("put", 0, 0, "long"),
        ],
        "covered_call": [
            OptionLeg("call", 0, 0, "short"),
        ],
        "protective_put": [
            OptionLeg("put", 0, 0, "long"),
        ],
        "iron_condor": [
            OptionLeg("put", 0, 0, "short"),
            OptionLeg("put", 0, 0, "long"),
            OptionLeg("call", 0, 0, "short"),
            OptionLeg("call", 0, 0, "long"),
        ],
        "butterfly": [
            OptionLeg("call", 0, 0, "long"),
            OptionLeg("call", 0, 0, "short"),
            OptionLeg("call", 0, 0, "short"),
            OptionLeg("call", 0, 0, "long"),
        ],
    }

    def __init__(
        self,
        title_prefix: str = "",
        figsize: Tuple[float, float] = (10, 6),
    ):
        self.title_prefix = title_prefix
        self.figsize = figsize

    @staticmethod
    def _build_legs(
        strategy: str,
        K: float,
        premium: Union[float, List[float]],
        strikes: Optional[List[float]] = None,
    ) -> List[OptionLeg]:
        """Build option legs from strategy name and parameters."""
        prem_list = [premium] if not isinstance(premium, list) else premium

        if strategy == "long_call":
            return [OptionLeg("call", K, prem_list[0], "long")]
        elif strategy == "long_put":
            return [OptionLeg("put", K, prem_list[0], "long")]
        elif strategy == "short_call":
            return [OptionLeg("call", K, prem_list[0], "short")]
        elif strategy == "short_put":
            return [OptionLeg("put", K, prem_list[0], "short")]
        elif strategy == "straddle":
            prem_c = prem_list[0] if len(prem_list) >= 1 else premium
            prem_p = prem_list[1] if len(prem_list) >= 2 else premium
            return [
                OptionLeg("call", K, prem_c, "long"),
                OptionLeg("put", K, prem_p, "long"),
            ]
        elif strategy == "strangle":
            K_call = strikes[0] if strikes else K * 1.05
            K_put = strikes[1] if strikes and len(strikes) >= 2 else K * 0.95
            prem_c = prem_list[0] if len(prem_list) >= 1 else premium
            prem_p = prem_list[1] if len(prem_list) >= 2 else premium
            return [
                OptionLeg("call", K_call, prem_c, "long"),
                OptionLeg("put", K_put, prem_p, "long"),
            ]
        elif strategy == "covered_call":
            # Stock + short call (stock PnL = S_T − S_0)
            return [OptionLeg("call", K, prem_list[0] if prem_list else premium, "short")]
        elif strategy == "protective_put":
            return [OptionLeg("put", K, prem_list[0] if prem_list else premium, "long")]
        elif strategy == "iron_condor":
            K1 = strikes[0] if strikes and len(strikes) >= 1 else K * 0.90
            K2 = strikes[1] if strikes and len(strikes) >= 2 else K * 0.95
            K3 = strikes[2] if strikes and len(strikes) >= 3 else K * 1.05
            K4 = strikes[3] if strikes and len(strikes) >= 4 else K * 1.10
            ps = prem_list + [premium] * max(0, 4 - len(prem_list))
            return [
                OptionLeg("put", K1, ps[0], "short"),
                OptionLeg("put", K2, ps[1], "long"),
                OptionLeg("call", K3, ps[2], "short"),
                OptionLeg("call", K4, ps[3], "long"),
            ]
        elif strategy == "butterfly":
            K1 = K * 0.90
            K2 = K
            K3 = K * 1.10
            ps = prem_list + [premium] * max(0, 4 - len(prem_list))
            return [
                OptionLeg("call", K1, ps[0], "long"),
                OptionLeg("call", K2, ps[1], "short"),
                OptionLeg("call", K2, ps[2], "short"),
                OptionLeg("call", K3, ps[3], "long"),
            ]
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. Use: {list(OptionsVisualizer._STRATEGIES.keys())}"
            )

    def payoff_diagram(
        self,
        strategy: str = "straddle",
        S_range: Tuple[float, float] = (80, 120),
        K: float = 100,
        premium: Union[float, List[float]] = 3.0,
        strikes: Optional[List[float]] = None,
        S_0: float = 100,
        n_points: int = 200,
    ) -> Dict:
        """Generate payoff diagram data for an options strategy.

        Args:
            strategy: Strategy name (e.g., "straddle", "iron_condor").
            S_range: (min, max) price range for x-axis.
            K: Base strike price.
            premium: Option premium(s) — single float or list per leg.
            strikes: List of strikes for multi-strike strategies.
            S_0: Current stock price (for covered_call stock PnL baseline).
            n_points: Number of price points.

        Returns:
            Dictionary with price array, PnL array, strategy name, and
            key levels (max profit, max loss, breakevens).
        """
        S = np.linspace(S_range[0], S_range[1], n_points)
        legs = self._build_legs(strategy, K, premium, strikes)

        total_pnl = np.zeros(n_points)
        for leg in legs:
            total_pnl += leg.payoff(S)

        # Covered call: add stock PnL
        if strategy == "covered_call":
            total_pnl += S - S_0

        # Protective put: add stock PnL
        if strategy == "protective_put":
            total_pnl += S - S_0

        max_profit = float(np.max(total_pnl))
        max_loss = float(np.min(total_pnl))

        # Find breakevens (where PnL crosses zero)
        breakevens = []
        for i in range(1, len(S)):
            if total_pnl[i - 1] * total_pnl[i] <= 0:
                be = float(
                    S[i - 1]
                    - total_pnl[i - 1] * (S[i] - S[i - 1]) / (total_pnl[i] - total_pnl[i - 1])
                )
                breakevens.append(round(be, 2))

        return {
            "strategy": strategy,
            "S": S.tolist(),
            "pnl": total_pnl.tolist(),
            "max_profit": round(max_profit, 4),
            "max_loss": round(max_loss, 4),
            "breakevens": breakevens,
            "legs": [
                {
                    "type": leg.option_type,
                    "strike": leg.strike,
                    "premium": leg.premium,
                    "position": leg.position,
                    "quantity": leg.quantity,
                }
                for leg in legs
            ],
            "pnl_at_S0": round(float(np.interp(S_0, S, total_pnl)), 4),
        }

    def vol_surface_3d(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_matrix: np.ndarray,
        F: Optional[float] = None,
    ) -> Dict:
        """Generate 3D volatility surface data.

        Args:
            strikes: Array of strike prices.
            maturities: Array of times to expiration (years).
            iv_matrix: Implied vol matrix [n_maturities, n_strikes].
            F: Forward price (auto-detected if None).

        Returns:
            Dictionary with meshgrid-like data for 3D plotting.
        """
        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        iv_matrix = np.asarray(iv_matrix, dtype=float)

        if F is None:
            F = strikes[np.argmin(np.abs(strikes - strikes.mean()))]

        # Moneyness
        m_grid, t_grid = np.meshgrid(strikes / F, maturities)

        # ATM vol term structure
        atm_idx = np.argmin(np.abs(strikes - F))
        atm_vols = iv_matrix[:, atm_idx]

        # Skew per maturity: 90% moneyness IV − 110% moneyness IV
        left_idx = np.argmin(np.abs(strikes - 0.90 * F))
        right_idx = np.argmin(np.abs(strikes - 1.10 * F))
        skews = iv_matrix[:, left_idx] - iv_matrix[:, right_idx]

        # Smile per maturity
        smiles = (iv_matrix[:, left_idx] + iv_matrix[:, right_idx]) / 2 - iv_matrix[:, atm_idx]

        return {
            "moneyness": (strikes / F).round(4).tolist(),
            "maturities": maturities.round(4).tolist(),
            "iv_matrix": iv_matrix.round(6).tolist(),
            "F": round(F, 2),
            "atm_vol_term": atm_vols.round(4).tolist(),
            "skew_per_maturity": skews.round(4).tolist(),
            "smile_per_maturity": smiles.round(4).tolist(),
            "total_variance": (iv_matrix**2 * maturities[:, np.newaxis]).round(6).tolist(),
        }

    def theta_curve(
        self,
        S: float,
        K: float,
        r: float,
        sigma: float,
        T_max: float = 1.0,
        n_points: int = 100,
        q: float = 0.0,
    ) -> Dict:
        """Generate theta decay curve for an ATM option.

        Shows how time decay accelerates as expiration approaches.

        Args:
            S: Spot price.
            K: Strike.
            r: Risk-free rate.
            sigma: Volatility.
            T_max: Maximum time to expiration (years).
            n_points: Number of time points.
            q: Dividend yield.

        Returns:
            Dictionary with time-to-expiry array and theta values.
        """
        from scipy.stats import norm

        T_arr = np.linspace(0.001, T_max, n_points)

        theta_c = np.zeros(n_points)
        theta_p = np.zeros(n_points)
        option_values_c = np.zeros(n_points)
        extrinsic_c = np.zeros(n_points)

        for i, T in enumerate(T_arr):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            nd1 = norm.pdf(d1)
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            Nm_d2 = norm.cdf(-d2)

            # Call price
            call = S * np.exp(-q * T) * Nd1 - K * np.exp(-r * T) * Nd2
            option_values_c[i] = call

            # Extrinsic value
            intrinsic_c = max(S - K * np.exp(-r * T), 0)
            extrinsic_c[i] = max(call - intrinsic_c, 0)

            # Theta (per day)
            t1 = -S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
            t2 = -r * K * np.exp(-r * T) * Nd2
            t3 = q * S * np.exp(-q * T) * Nd1
            theta_c[i] = (t1 + t2 + t3) / 365.0

            t4 = -S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
            t5 = r * K * np.exp(-r * T) * Nm_d2
            t6 = -q * S * np.exp(-q * T) * (1 - Nd1)
            theta_p[i] = (t4 + t5 + t6) / 365.0

        # Find theta acceleration point (second derivative inflection)
        dtheta = np.gradient(theta_c, T_arr[1] - T_arr[0])
        ddtheta = np.gradient(dtheta, T_arr[1] - T_arr[0])
        accel_days = float(T_arr[np.argmax(np.abs(ddtheta))]) * 365.0

        return {
            "days_to_expiry": (T_arr * 365).round(1).tolist(),
            "theta_call_per_day": theta_c.round(6).tolist(),
            "theta_put_per_day": theta_p.round(6).tolist(),
            "option_value_call": option_values_c.round(4).tolist(),
            "extrinsic_value_call": extrinsic_c.round(4).tolist(),
            "theta_acceleration_days": round(accel_days, 1),
            "theta_30d": round(float(theta_c[np.argmin(np.abs(T_arr - 30 / 365))]), 6),
            "theta_7d": round(float(theta_c[np.argmin(np.abs(T_arr - 7 / 365))]), 6),
            "theta_1d": round(float(theta_c[-1]), 6),
            "params": {"S": S, "K": K, "r": r, "sigma": sigma, "T_max": T_max, "q": q},
        }

    def skew_chart(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        F: Optional[float] = None,
        label: str = "",
    ) -> Dict:
        """Generate volatility skew/smile chart data.

        Args:
            strikes: Strike prices.
            ivs: Implied volatilities at each strike.
            F: Forward/spot price for moneyness calculation.
            label: Chart label (e.g., maturity).

        Returns:
            Dictionary with strikes, IVs, moneyness, and skew metrics.
        """
        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)
        if F is None:
            F = strikes[np.argmin(np.abs(strikes - strikes.mean()))]

        moneyness = strikes / F
        atm_idx = np.argmin(np.abs(moneyness - 1.0))
        atm_vol = float(ivs[atm_idx])

        left_90 = np.argmin(np.abs(moneyness - 0.90))
        right_110 = np.argmin(np.abs(moneyness - 1.10))
        skew = float(ivs[left_90] - ivs[right_110])
        smile = float((ivs[left_90] + ivs[right_110]) / 2 - atm_vol)

        # Skew slope: linear regression of IV vs moneyness
        valid = ~np.isnan(ivs)
        if valid.sum() >= 3:
            slope, intercept = np.polyfit(moneyness[valid], ivs[valid], 1)
        else:
            slope, intercept = np.nan, np.nan

        return {
            "strikes": strikes.tolist(),
            "ivs": ivs.round(6).tolist(),
            "moneyness": moneyness.round(4).tolist(),
            "F": round(F, 2),
            "atm_vol": round(atm_vol, 4),
            "skew": round(skew, 4),
            "smile": round(smile, 4),
            "skew_slope": round(float(slope), 6),
            "skew_intercept": round(float(intercept), 4),
            "label": label,
        }

    def term_structure_plot(
        self,
        maturities: np.ndarray,
        atm_ivs: np.ndarray,
    ) -> Dict:
        """Generate volatility term structure chart data.

        Args:
            maturities: Times to expiration (years).
            atm_ivs: ATM implied volatilities.

        Returns:
            Dictionary with term structure data and contango/backwardation flag.
        """
        maturities = np.asarray(maturities, dtype=float)
        atm_ivs = np.asarray(atm_ivs, dtype=float)

        contango = atm_ivs[-1] > atm_ivs[0] if len(atm_ivs) >= 2 else None
        term_slope = (
            float((atm_ivs[-1] - atm_ivs[0]) / (maturities[-1] - maturities[0]))
            if len(atm_ivs) >= 2 and maturities[-1] > maturities[0]
            else np.nan
        )

        return {
            "maturities_years": maturities.round(4).tolist(),
            "atm_ivs": atm_ivs.round(4).tolist(),
            "contango": contango,
            "term_slope_bp_per_year": round(term_slope * 10000, 1)
            if not np.isnan(term_slope)
            else None,
            "min_vol": round(float(np.min(atm_ivs)), 4),
            "max_vol": round(float(np.max(atm_ivs)), 4),
            "spread": round(float(atm_ivs[-1] - atm_ivs[0]) * 10000, 1)
            if len(atm_ivs) >= 2
            else None,
        }

    def greeks_heatmap(
        self,
        S_range: Tuple[float, float],
        T: float,
        r: float,
        sigma: float,
        greek: str = "gamma",
        n_points: int = 50,
        q: float = 0.0,
    ) -> Dict:
        """Generate Greeks heatmap data (strike vs spot price).

        Args:
            S_range: (min, max) spot price range.
            T: Time to expiration.
            r: Risk-free rate.
            sigma: Volatility.
            greek: Greek to compute ("delta", "gamma", "theta", "vega", "rho").
            n_points: Grid resolution.
            q: Dividend yield.

        Returns:
            Dictionary with grid data for the specified Greek.
        """
        from scipy.stats import norm

        S_grid = np.linspace(S_range[0], S_range[1], n_points)
        K_grid = np.linspace(S_range[0] * 0.5, S_range[1] * 1.5, n_points)

        greek_vals = np.zeros((n_points, n_points))
        for i, S in enumerate(S_grid):
            for j, K in enumerate(K_grid):
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                nd1 = norm.pdf(d1)

                if greek == "delta":
                    greek_vals[i, j] = np.exp(-q * T) * norm.cdf(d1)
                elif greek == "gamma":
                    greek_vals[i, j] = np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T))
                elif greek == "theta":
                    t1 = -S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
                    t2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
                    t3 = q * S * np.exp(-q * T) * norm.cdf(d1)
                    greek_vals[i, j] = (t1 + t2 + t3) / 365.0
                elif greek == "vega":
                    greek_vals[i, j] = S * np.exp(-q * T) * nd1 * np.sqrt(T) / 100.0
                elif greek == "rho":
                    greek_vals[i, j] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0

        return {
            "greek": greek,
            "S": S_grid.tolist(),
            "K": K_grid.tolist(),
            "values": greek_vals.round(6).tolist(),
            "T": T,
            "r": r,
            "sigma": sigma,
            "max_abs": round(float(np.max(np.abs(greek_vals))), 6),
        }
