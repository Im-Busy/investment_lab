"""
Q7: Market Impact Model — Almgren-Chriss Optimal Execution.

Quantifies permanent + temporary price impact of trades to bridge
backtest-to-live performance. Every backtest currently assumes zero
market impact cost.

Models:
  Permanent Impact:
    Information leakage. Proportional to trade size, independent of speed.
    Impact = eta * sigma * (X / V_daily)

  Temporary Impact:
    Liquidity cost of execution speed. Decreases with slower execution.
    Impact = epsilon * sigma * (X / (V_daily * tau))^beta

  Almgren-Chriss Implementation Shortfall:
    Total cost = permanent + temporary impact, minimized by trading
    at optimal speed v* that balances impact vs timing risk.

Usage:
    >>> impact = MarketImpact(eta=0.1, epsilon=0.5, beta=0.6)
    >>> cost = impact.total_cost(notional=100_000, daily_volume=50e6, vol=0.20, participation=0.05)
    >>> schedule = impact.optimal_schedule(notional=100_000, daily_volume=50e6, vol=0.20)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImpactResult:
    """Market impact cost estimation result."""

    notional: float
    daily_volume: float
    annual_vol: float
    participation_rate: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_impact_bps: float
    total_cost: float
    optimal_participation: Optional[float] = None
    notes: str = ""

    def summary(self) -> str:
        return (
            f"Impact: {self.total_impact_bps:.2f} bps (${self.total_cost:.2f}) "
            f"[perm={self.permanent_impact_bps:.2f} temp={self.temporary_impact_bps:.2f}] "
            f"on ${self.notional:,.0f}"
        )

    def to_dict(self) -> dict:
        return {
            "notional": round(self.notional, 2),
            "daily_volume": round(self.daily_volume, 2),
            "annual_vol": round(self.annual_vol, 4),
            "participation_rate": round(self.participation_rate, 4),
            "permanent_impact_bps": round(self.permanent_impact_bps, 2),
            "temporary_impact_bps": round(self.temporary_impact_bps, 2),
            "total_impact_bps": round(self.total_impact_bps, 2),
            "total_cost": round(self.total_cost, 2),
            "optimal_participation": round(self.optimal_participation, 4)
            if self.optimal_participation
            else None,
        }


class MarketImpact:
    """Almgren-Chriss market impact model.

    Args:
        eta: Permanent impact coefficient (default 0.1 = 10% of daily vol).
        epsilon: Temporary impact coefficient (default 0.5).
        beta: Temporary impact exponent (default 0.6, typically 0.5-1.0).
        risk_aversion: Trader risk aversion for optimal schedule.
        daily_vol_scaling: sqrt(252) for annualized vol -> daily vol.
    """

    def __init__(
        self,
        eta: float = 0.10,
        epsilon: float = 0.50,
        beta: float = 0.60,
        risk_aversion: float = 1e-6,
    ):
        self.eta = eta
        self.epsilon = epsilon
        self.beta = beta
        self.risk_aversion = risk_aversion
        self.daily_vol_scaling = np.sqrt(1 / 252)

    def permanent_impact_bps(
        self, notional: float, daily_volume: float, annual_vol: float
    ) -> float:
        """Permanent price impact in basis points.

        I_perm = eta * sigma_daily * (X / V)
        where X = notional, V = daily volume in dollars, sigma = daily vol.
        """
        sigma_daily = annual_vol * self.daily_vol_scaling
        signed_volume = notional / max(daily_volume, 1.0)
        return float(self.eta * sigma_daily * signed_volume * 10_000)

    def temporary_impact_bps(
        self, notional: float, daily_volume: float, annual_vol: float, participation: float
    ) -> float:
        """Temporary price impact in basis points.

        I_temp = epsilon * sigma_daily * (X / (V * tau))^beta
        where tau = participation rate (fraction of daily volume).
        """
        sigma_daily = annual_vol * self.daily_vol_scaling
        v_rate = notional / (daily_volume * max(participation, 0.001))
        return float(self.epsilon * sigma_daily * (v_rate**self.beta) * 10_000)

    def total_impact_bps(
        self, notional: float, daily_volume: float, annual_vol: float, participation: float = 0.05
    ) -> float:
        """Total implementation shortfall in basis points."""
        perm = self.permanent_impact_bps(notional, daily_volume, annual_vol)
        temp = self.temporary_impact_bps(notional, daily_volume, annual_vol, participation)
        return perm + temp

    def total_cost(
        self, notional: float, daily_volume: float, annual_vol: float, participation: float = 0.05
    ) -> float:
        """Total cost in dollars."""
        bps = self.total_impact_bps(notional, daily_volume, annual_vol, participation)
        return notional * bps / 10_000

    def evaluate(
        self, notional: float, daily_volume: float, annual_vol: float, participation: float = 0.05
    ) -> ImpactResult:
        """Full impact cost evaluation."""
        perm = self.permanent_impact_bps(notional, daily_volume, annual_vol)
        temp = self.temporary_impact_bps(notional, daily_volume, annual_vol, participation)
        total_bps = perm + temp
        total_dollars = notional * total_bps / 10_000

        notes = []
        if total_bps > 50:
            notes.append("HIGH: consider reducing position size or using algo")
        elif total_bps > 10:
            notes.append("MODERATE: acceptable for institutional trading")
        else:
            notes.append("LOW: negligible impact cost")

        return ImpactResult(
            notional=notional,
            daily_volume=daily_volume,
            annual_vol=annual_vol,
            participation_rate=participation,
            permanent_impact_bps=perm,
            temporary_impact_bps=temp,
            total_impact_bps=total_bps,
            total_cost=total_dollars,
            notes="; ".join(notes),
        )

    def optimal_participation(
        self, notional: float, daily_volume: float, annual_vol: float
    ) -> float:
        """Estimate optimal participation rate to minimize total cost.

        Approximate solution from Almgren-Chriss (2001):
        tau* = (epsilon * beta * X / (eta * V))^(1/(1+beta))
        """
        signed_vol = notional / max(daily_volume, 1.0)
        num = self.epsilon * self.beta * signed_vol
        den = self.eta
        return float(min((num / max(den, 1e-10)) ** (1 / (1 + self.beta)), 0.25))

    def backtest_cost_adjustment(
        self,
        trades: list[dict],
        daily_volume: float,
        annual_vol: float,
    ) -> float:
        """Estimate total market impact for a list of trades.

        Args:
            trades: List of dicts with 'notional' key.
            daily_volume: Average daily dollar volume.
            annual_vol: Annualized volatility.

        Returns:
            Total estimated impact cost in dollars.
        """
        total = 0.0
        for t in trades:
            notional = abs(t.get("notional", 0))
            if notional > 0:
                total += self.total_cost(notional, daily_volume, annual_vol)
        return total
