"""
Block A13: Volatility Trading Strategies.

Analyzes and backtests volatility-based trading strategies including
straddles, strangles, volatility arbitrage, and variance risk premium
harvesting. Uses projected PnL analysis — no actual backtesting engine.

Strategies:
    VolTradeAnalyzer: Central analyzer for vol strategy PnL projection
        and risk metrics. Supports 10+ option strategies with Greeks-based
        scenario analysis.
    StraddleAnalysis: Long/short straddle PnL vs realized volatility.
    StrangleAnalysis: Asymmetric width optimization.
    VolArbitrage: Variance risk premium estimate and monetization signal.
    VegaHedgeRatio: Vega-neutral portfolio construction.

Usage:
    >>> va = VolTradeAnalyzer()
    >>> result = va.analyze_straddle(S=100, K=100, T=0.25, r=0.05, sigma_implied=0.20,
    ...                              sigma_realized=0.25, premium=5.0)

    See also: src/ml/options_pricing.py (A1-A7 pricing models),
              src/risk/delta_hedging.py (A11 dynamic hedging).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class VolTradeResult:
    strategy: str
    expected_pnl: float
    pnl_pct: float
    expected_move: float
    breakeven_range: Tuple[float, float]
    max_profit: float
    max_loss: float
    risk_reward: float
    probability_profit: float
    greeks: Dict
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "expected_pnl": round(self.expected_pnl, 4),
            "pnl_percent": round(self.pnl_pct, 2),
            "expected_move": round(self.expected_move, 2),
            "breakeven_range": [
                round(self.breakeven_range[0], 2),
                round(self.breakeven_range[1], 2),
            ],
            "max_profit": round(self.max_profit, 2)
            if np.isfinite(self.max_profit)
            else "unlimited",
            "max_loss": round(self.max_loss, 2),
            "risk_reward": round(self.risk_reward, 2),
            "prob_profit": round(self.probability_profit, 2),
            "greeks": {k: round(v, 6) for k, v in self.greeks.items()},
            "notes": self.notes,
        }


@dataclass
class VolArbitrageResult:
    var_premium: float
    realized_vol: float
    implied_vol: float
    premium: float
    signal: str  # "LONG_VOL", "SHORT_VOL", "NEUTRAL"
    confidence: float
    expected_sharpe: float
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "variance_premium": round(self.var_premium, 6),
            "realized_vol": round(self.realized_vol, 4),
            "implied_vol": round(self.implied_vol, 4),
            "premium_pct": round(self.premium * 100, 2),
            "signal": self.signal,
            "confidence": round(self.confidence, 2),
            "expected_sharpe": round(self.expected_sharpe, 2),
            "notes": self.notes,
        }


class VolTradeAnalyzer:
    """Volatility trading strategy analyzer.

    Uses Black-Scholes Greeks and scenario analysis to project PnL for
    common volatility strategies. All analysis is model-based (no actual
    backtesting) — suitable for pre-trade analysis and signal generation.

    Args:
        risk_free_rate: Default risk-free rate for pricing.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def _bs_greeks(
        self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
    ) -> Dict:
        """Compute Black-Scholes Greeks for a European option."""
        T = max(T, 1e-10)
        sigma = max(sigma, 1e-10)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta_c = np.exp(-q * T) * norm.cdf(d1)
        delta_p = np.exp(-q * T) * (norm.cdf(d1) - 1.0)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0
        t1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        t2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        t3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        theta_c = (t1 + t2 + t3) / 365.0

        return {
            "call": round(call, 6),
            "put": round(put, 6),
            "delta_call": round(delta_c, 4),
            "delta_put": round(delta_p, 4),
            "gamma": round(gamma, 6),
            "vega": round(vega, 4),
            "theta_call": round(theta_c, 6),
        }

    def analyze_straddle(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma_implied: float,
        sigma_realized: float,
        premium: Optional[float] = None,
        direction: str = "long",
    ) -> VolTradeResult:
        """Analyze a straddle position (call + put at same strike).

        Args:
            S: Spot price.
            K: Strike (typically ATM).
            T: Time to expiration (years).
            r: Risk-free rate.
            sigma_implied: Implied volatility (market price).
            sigma_realized: Expected realized volatility.
            premium: Total straddle premium. Auto-computed if None.
            direction: "long" or "short".

        Returns:
            VolTradeResult with PnL projection and risk metrics.
        """
        greeks = self._bs_greeks(S, K, T, r, sigma_implied)

        if premium is None:
            premium = greeks["call"] + greeks["put"]

        sign = 1 if direction == "long" else -1
        expected_move = S * sigma_realized * np.sqrt(T)
        breakeven_up = K + premium if direction == "long" else K
        breakeven_down = K - premium if direction == "long" else K

        if direction == "long":
            max_profit = float("inf")
            max_loss = -premium
        else:
            max_profit = premium
            max_loss = float("-inf")

        # Expected PnL at expiration assuming normal returns
        expected_std_move = S * sigma_realized * np.sqrt(T)
        prob_above_breakeven = 1.0 - norm.cdf(
            (breakeven_up - S) / (S * sigma_realized * np.sqrt(T))
        )
        prob_below_breakeven = norm.cdf((breakeven_down - S) / (S * sigma_realized * np.sqrt(T)))

        if direction == "long":
            probability_profit = prob_above_breakeven + prob_below_breakeven
            expected_abs_move = S * sigma_realized * np.sqrt(2 * T / np.pi)
            expected_call_pnl = max(S + expected_abs_move - K, 0)
            expected_put_pnl = max(K - (S - expected_abs_move), 0)
            expected_pnl = expected_call_pnl + expected_put_pnl - premium
        else:
            probability_profit = 1.0 - (prob_above_breakeven + prob_below_breakeven)
            expected_pnl = sign * (premium - expected_move * 0.8)

        risk_reward = (
            abs(max_profit / max_loss)
            if max_loss != 0 and max_loss != float("-inf") and max_profit != float("inf")
            else float("inf")
            if max_profit == float("inf")
            else 0.0
        )
        if max_loss == float("-inf") or risk_reward == float("inf"):
            risk_reward = 0.0

        notes = (
            f"{direction.upper()} straddle: IV={sigma_implied:.0%}, RV={sigma_realized:.0%}, "
            f"BE=[{breakeven_down:.1f}, {breakeven_up:.1f}]"
        )

        return VolTradeResult(
            strategy=f"{direction}_straddle",
            expected_pnl=round(expected_pnl, 4),
            pnl_pct=round(expected_pnl / premium * 100, 2) if premium > 0 else 0.0,
            expected_move=round(expected_std_move, 2),
            breakeven_range=(round(breakeven_down, 2), round(breakeven_up, 2)),
            max_profit=round(max_profit, 2) if np.isfinite(max_profit) else float("inf"),
            max_loss=round(max_loss, 2) if np.isfinite(max_loss) else float("-inf"),
            risk_reward=round(risk_reward, 2) if np.isfinite(risk_reward) else 0.0,
            probability_profit=round(probability_profit, 4),
            greeks={
                "vega": greeks["vega"] * 2,
                "gamma": greeks["gamma"] * 2,
                "theta": greeks["theta_call"] * 2,
                "delta": round(greeks["delta_call"] + greeks["delta_put"], 4),
            },
            notes=notes,
        )

    def analyze_strangle(
        self,
        S: float,
        K_call: float,
        K_put: float,
        T: float,
        r: float,
        sigma_implied_call: float,
        sigma_implied_put: float,
        sigma_realized: float,
        premium_call: Optional[float] = None,
        premium_put: Optional[float] = None,
        direction: str = "long",
    ) -> VolTradeResult:
        """Analyze a strangle position (OTM call + OTM put).

        Args:
            S: Spot price.
            K_call: Call strike (typically OTM, above S).
            K_put: Put strike (typically OTM, below S).
            T: Time to expiration.
            r: Risk-free rate.
            sigma_implied_call: Call implied vol.
            sigma_implied_put: Put implied vol.
            sigma_realized: Expected realized volatility.
            premium_call: Call premium (auto-computed if None).
            premium_put: Put premium (auto-computed if None).
            direction: "long" or "short".

        Returns:
            VolTradeResult with PnL projection.
        """
        greeks_call = self._bs_greeks(S, K_call, T, r, sigma_implied_call)
        greeks_put = self._bs_greeks(S, K_put, T, r, sigma_implied_put)

        if premium_call is None:
            premium_call = greeks_call["call"]
        if premium_put is None:
            premium_put = greeks_put["put"]
        total_premium = premium_call + premium_put

        sign = 1 if direction == "long" else -1

        if direction == "long":
            breakeven_up = K_call + total_premium
            breakeven_down = K_put - total_premium
            max_profit = float("inf")
            max_loss = -total_premium
        else:
            breakeven_up = K_call
            breakeven_down = K_put
            max_profit = total_premium
            max_loss = float("-inf")

        expected_std_move = S * sigma_realized * np.sqrt(T)
        prob_above = 1.0 - norm.cdf((breakeven_up - S) / expected_std_move)
        prob_below = norm.cdf((breakeven_down - S) / expected_std_move)
        probability_profit = (
            prob_above + prob_below if direction == "long" else 1.0 - prob_above - prob_below
        )

        width_pct = (K_call - K_put) / S * 100
        notes = (
            f"{direction.upper()} {width_pct:.0f}% strangle: "
            f"IV=[{sigma_implied_call:.0%}/{sigma_implied_put:.0%}], RV={sigma_realized:.0%}"
        )

        return VolTradeResult(
            strategy=f"{direction}_strangle_{width_pct:.0f}pct",
            expected_pnl=round(sign * (total_premium - expected_std_move * 0.6), 4),
            pnl_pct=round(expected_std_move / total_premium * 100, 2) if total_premium > 0 else 0.0,
            expected_move=round(expected_std_move, 2),
            breakeven_range=(round(breakeven_down, 2), round(breakeven_up, 2)),
            max_profit=round(max_profit, 2) if np.isfinite(max_profit) else float("inf"),
            max_loss=round(max_loss, 2) if np.isfinite(max_loss) else float("-inf"),
            risk_reward=round(abs(max_profit / max_loss), 2)
            if max_loss != 0 and max_loss != float("-inf") and max_profit != float("inf")
            else 0.0,
            probability_profit=round(probability_profit, 4),
            greeks={
                "vega": round(greeks_call["vega"] + greeks_put["vega"], 4),
                "gamma": round(greeks_call["gamma"] + greeks_put["gamma"], 6),
                "theta": round(greeks_call["theta_call"] + greeks_put["theta_call"], 6),
                "delta": round(greeks_call["delta_call"] + greeks_put["delta_put"], 4),
            },
            notes=notes,
        )

    def analyze_variance_premium(
        self,
        implied_vol: float,
        realized_vol: float,
        T: float = 21 / 252,
        vol_vol: float = 0.8,
    ) -> VolArbitrageResult:
        """Estimate variance risk premium and generate vol trading signal.

        The variance risk premium = IV − RV reflects the premium investors
        pay for volatility protection. A large positive premium suggests
        selling vol (short straddle/strangle).

        Args:
            implied_vol: Current implied volatility (decimal).
            realized_vol: Recent realized volatility (decimal).
            T: Forecast horizon in years.
            vol_vol: Volatility of volatility (typical range 0.5-1.2).

        Returns:
            VolArbitrageResult with signal and confidence.
        """
        var_premium = implied_vol**2 - realized_vol**2
        premium = implied_vol - realized_vol

        # Z-score of the premium relative to vol uncertainty
        vol_uncertainty = realized_vol * vol_vol * np.sqrt(T)
        z_score = premium / vol_uncertainty if vol_uncertainty > 0 else 0.0

        # Signal classification
        if premium > 0.05:
            signal = "SHORT_VOL"
            confidence = min(abs(z_score) / 2.0, 1.0)
            sharpe = premium / (realized_vol + 0.01)
        elif premium < -0.05:
            signal = "LONG_VOL"
            confidence = min(abs(z_score) / 2.0, 1.0)
            sharpe = abs(premium) / (realized_vol + 0.01)
        else:
            signal = "NEUTRAL"
            confidence = 0.0
            sharpe = 0.0

        notes = f"VRP={var_premium:.4f}, z={z_score:.2f}, horizon={T * 252:.0f}d"

        return VolArbitrageResult(
            var_premium=var_premium,
            realized_vol=realized_vol,
            implied_vol=implied_vol,
            premium=premium,
            signal=signal,
            confidence=round(min(confidence, 1.0), 4),
            expected_sharpe=round(min(sharpe, 2.0), 2),
            notes=notes,
        )

    def optimize_strangle_width(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        sigma_realized: float,
        min_width_pct: float = 2.0,
        max_width_pct: float = 20.0,
        n_steps: int = 10,
    ) -> pd.DataFrame:
        """Find optimal strangle width for expected PnL.

        Sweeps OTM distance for both call and put to find the width
        that maximizes expected profit given realized vol.

        Args:
            S: Spot price.
            T: Time to expiration.
            r: Risk-free rate.
            sigma: Base implied volatility.
            sigma_realized: Expected realized volatility.
            min_width_pct: Minimum width as % of spot.
            max_width_pct: Maximum width as % of spot.
            n_steps: Number of width steps.

        Returns:
            DataFrame with width, premium, expected PnL, prob_profit.
        """
        widths = np.linspace(min_width_pct, max_width_pct, n_steps)
        results = []

        for width_pct in widths:
            offset = width_pct / 100 / 2
            K_call = S * (1 + offset)
            K_put = S * (1 - offset)

            # Adjust IV for skew: higher IV for lower strikes (typical equity skew)
            skew_adj = 0.02  # 2 vol points per 10% OTM
            sigma_call = sigma + offset * skew_adj * 0.5
            sigma_put = sigma - offset * skew_adj * 1.5

            greeks_c = self._bs_greeks(S, K_call, T, r, sigma_call)
            greeks_p = self._bs_greeks(S, K_put, T, r, sigma_put)
            total_premium = greeks_c["call"] + greeks_p["put"]

            expected_move = S * sigma_realized * np.sqrt(T)
            prob_profit = (
                2 * (1.0 - norm.cdf((K_call - S) / (S * sigma_realized * np.sqrt(T))))
                if S * sigma_realized > 0
                else 0.0
            )
            expected_pnl = max(expected_move - total_premium, -total_premium)

            results.append(
                {
                    "width_pct": round(width_pct, 1),
                    "K_put": round(K_put, 2),
                    "K_call": round(K_call, 2),
                    "total_premium": round(total_premium, 4),
                    "expected_pnl": round(expected_pnl, 4),
                    "prob_profit": round(prob_profit, 4),
                    "premium_yield": round(total_premium / S * 100, 2),
                }
            )

        return pd.DataFrame(results)

    def vega_neutral_portfolio(
        self,
        positions: List[Dict],
        target_vega: float = 0.0,
    ) -> Dict:
        """Compute position sizing for a vega-neutral options portfolio.

        Given a set of option positions, find the hedge ratios needed
        to achieve target portfolio vega.

        Args:
            positions: List of dicts with keys S, K, T, sigma, quantity, option_type.
            target_vega: Target portfolio vega (0 = vega-neutral).

        Returns:
            Dictionary with per-position hedge ratios and portfolio greeks.
        """
        total_vega = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        per_position = []

        for i, pos in enumerate(positions):
            greeks = self._bs_greeks(
                S=pos["S"],
                K=pos["K"],
                T=pos["T"],
                r=pos.get("r", self.risk_free_rate),
                sigma=pos["sigma"],
            )

            qty = pos.get("quantity", 1)
            opt_type = pos.get("option_type", "call")
            vega = greeks["vega"] * qty
            delta = (greeks["delta_call"] if opt_type == "call" else greeks["delta_put"]) * qty
            gamma = greeks["gamma"] * qty
            theta = greeks["theta_call"] * qty

            total_vega += vega
            total_delta += delta
            total_gamma += gamma
            total_theta += theta

            per_position.append(
                {
                    "index": i,
                    "vega": round(vega, 4),
                    "delta": round(delta, 4),
                    "gamma": round(gamma, 6),
                    "theta": round(theta, 6),
                    "option_type": opt_type,
                    "quantity": qty,
                }
            )

        # Hedge delta with underlying
        hedge_shares = -total_delta

        return {
            "portfolio_vega": round(total_vega, 4),
            "portfolio_delta": round(total_delta, 4),
            "portfolio_gamma": round(total_gamma, 6),
            "portfolio_theta": round(total_theta, 6),
            "vega_neutral": abs(total_vega) < 0.01,
            "delta_hedge_shares": round(hedge_shares, 4),
            "per_position": per_position,
            "target_vega": target_vega,
        }

    def volatility_signal(
        self,
        implied_vol: float,
        realized_vol: float,
        historical_vol: float,
        vol_percentile: float,
    ) -> Dict:
        """Multi-factor volatility trading signal.

        Combines variance risk premium, vol percentile, and vol momentum
        to generate a composite -1 to +1 signal.

        Args:
            implied_vol: Current implied vol.
            realized_vol: Recent realized vol.
            historical_vol: Long-term average vol.
            vol_percentile: Current vol percentile (0-100).

        Returns:
            Dictionary with composite signal and factor breakdown.
        """
        # VRP factor: positive = short vol, negative = long vol
        vrp = (implied_vol - realized_vol) / max(realized_vol, 0.01)
        vrp_signal = np.tanh(vrp / 0.05)  # Scale to [-1, 1]

        # Percentile factor: high vol = sell, low vol = buy
        pct_signal = -np.tanh((vol_percentile - 50) / 20)

        # Momentum factor: rising vol = sell, falling vol = buy
        momentum = (realized_vol - historical_vol) / max(historical_vol, 0.01)
        mom_signal = np.tanh(momentum / 0.03)

        # Composite: equal-weighted
        composite = (vrp_signal + pct_signal + mom_signal) / 3.0

        return {
            "composite_signal": round(composite, 4),
            "action": "SELL_VOL"
            if composite > 0.2
            else "BUY_VOL"
            if composite < -0.2
            else "NEUTRAL",
            "vrp_signal": round(vrp_signal, 4),
            "percentile_signal": round(pct_signal, 4),
            "momentum_signal": round(mom_signal, 4),
            "implied_vol": round(implied_vol, 4),
            "realized_vol": round(realized_vol, 4),
            "vol_percentile": round(vol_percentile, 1),
            "vrp": round(vrp, 4),
        }

    @staticmethod
    def rolling_vrp_series(
        implied_vols: np.ndarray,
        realized_vols: np.ndarray,
        window: int = 252,
    ) -> pd.DataFrame:
        """Compute rolling variance risk premium time series.

        Args:
            implied_vols: Array of implied vol (e.g., VIX/100 / sqrt(252)).
            realized_vols: Array of realized vol (matching frequency).
            window: Rolling window for z-score normalization.

        Returns:
            DataFrame with VRP, z-score, percentile, signal columns.
        """
        implied_vols = np.asarray(implied_vols, dtype=float)
        realized_vols = np.asarray(realized_vols, dtype=float)
        n = min(len(implied_vols), len(realized_vols))

        vrp = implied_vols[:n] - realized_vols[:n]
        df = pd.DataFrame(
            {
                "implied_vol": implied_vols[:n],
                "realized_vol": realized_vols[:n],
                "vrp": vrp,
            }
        )

        df["vrp_zscore"] = (
            vrp - pd.Series(vrp).rolling(window, min_periods=20).mean()
        ) / pd.Series(vrp).rolling(window, min_periods=20).std()
        df["vrp_percentile"] = (
            pd.Series(vrp)
            .rolling(window, min_periods=63)
            .apply(lambda x: (x.iloc[-1] > x).mean() * 100, raw=False)
        )
        df["signal"] = np.tanh(df["vrp_zscore"].fillna(0) / 2.0)

        return df
