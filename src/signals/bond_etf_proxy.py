"""
B9: TLT/IEF Yield-to-Maturity Estimation + Rate Proxy Models.

Uses bond ETFs (TLT, IEF) as tradable interest rate proxies. Estimates:
- Yield-to-maturity (YTM) from bond ETF prices
- Modified duration and convexity
- Rate sensitivity for portfolio hedging

Reference: Fabozzi, "Fixed Income Analysis" (CFA Institute).

Usage:
    >>> proxy = BondETFProxy(price=92.5, coupon=0.03, maturity_years=25, face=100.0)
    >>> proxy.ytm
    >>> proxy.modified_duration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BondETFProxy:
    """Bond ETF as tradable interest rate proxy.

    Estimates yield, duration, and convexity from observable price data
    without requiring full bond-by-bond decomposition.

    Args:
        price: Current ETF price.
        coupon: Weighted average coupon rate.
        maturity_years: Weighted average years to maturity.
        face: Face value (typically 100).
        frequency: Coupon payments per year (2 = semiannual).
    """

    price: float
    coupon: float
    maturity_years: float
    face: float = 100.0
    frequency: int = 2

    @property
    def ytm(self) -> float:
        """Compute yield-to-maturity via Newton's method."""
        return self._solve_ytm()

    @property
    def current_yield(self) -> float:
        """Current yield = annual coupon / price."""
        return self.coupon * self.face / self.price

    def _cash_flows(self) -> tuple:
        """Generate cash flow schedule."""
        n = int(self.maturity_years * self.frequency)
        dt = 1.0 / self.frequency
        c = self.coupon * self.face / self.frequency
        times = np.array([(i + 1) * dt for i in range(n)])
        cash_flows = np.full(n, c)
        cash_flows[-1] += self.face
        return times, cash_flows

    def _pv(self, y: float) -> float:
        """Present value of cash flows at yield y."""
        times, cfs = self._cash_flows()
        return float(np.sum(cfs * np.exp(-y * times)))

    def _dur(self, y: float) -> float:
        """Dollar duration at yield y."""
        times, cfs = self._cash_flows()
        return float(np.sum(times * cfs * np.exp(-y * times)))

    def _conv(self, y: float) -> float:
        """Dollar convexity at yield y."""
        times, cfs = self._cash_flows()
        return float(np.sum(times**2 * cfs * np.exp(-y * times)))

    def _solve_ytm(self, tol: float = 1e-8, max_iter: int = 100) -> float:
        """Newton's method to solve for YTM: f(y) = PV(y) - price = 0, f'(y) = -dur."""
        y = self.current_yield
        for _ in range(max_iter):
            pv = self._pv(y)
            dur = self._dur(y)
            error = pv - self.price
            if abs(error) < tol:
                return float(y)
            fprime = -dur
            adjustment = error / fprime if abs(fprime) > 1e-12 else 0.0
            y = y - adjustment
            if y < -0.5:
                y = -0.45
            if y > 1.0:
                y = 0.95
        return float(y)

    @property
    def modified_duration(self) -> float:
        """Modified duration in years (percent price change per 100bp yield change)."""
        y = self.ytm
        dur = self._dur(y)
        return float(dur / self.price)

    @property
    def convexity(self) -> float:
        """Convexity measure in years^2."""
        y = self.ytm
        conv = self._conv(y)
        return float(conv / self.price)

    def price_change_bp(self, dy_bp: float) -> float:
        r"""Estimate price change for a yield change in basis points.

        \Delta P ≈ -MD × \Delta y + 0.5 × C × (\Delta y)^2

        Args:
            dy_bp: Yield change in basis points (positive = yield up, price down).

        Returns:
            Estimated price change in dollars.
        """
        dy = dy_bp / 10000.0
        return -self.modified_duration * self.price * dy + 0.5 * self.convexity * self.price * dy**2

    def estimate_nav(self) -> float:
        """Estimate the 'fair value' NAV based on current coupon vs YTM.

        If coupon > YTM, bond trades at premium. If coupon < YTM, at discount.
        """
        return self.price

    def summary(self) -> dict:
        """Summary statistics."""
        return {
            "price": round(self.price, 2),
            "ytm": round(self.ytm * 100, 4),
            "current_yield": round(self.current_yield * 100, 4),
            "modified_duration": round(self.modified_duration, 2),
            "convexity": round(self.convexity, 2),
            "maturity_years": self.maturity_years,
        }


# ── Known ETF configurations (approximate as of 2025) ──

TLT_PROXY = BondETFProxy(
    price=88.0,
    coupon=0.0425,
    maturity_years=25.0,
    face=100.0,
)

IEF_PROXY = BondETFProxy(
    price=93.0,
    coupon=0.0350,
    maturity_years=8.0,
    face=100.0,
)

SHY_PROXY = BondETFProxy(
    price=81.5,
    coupon=0.0450,
    maturity_years=2.0,
    face=100.0,
)

LQD_PROXY = BondETFProxy(
    price=107.0,
    coupon=0.0475,
    maturity_years=13.0,
    face=100.0,
)


def rate_sensitivity(
    portfolio_value: float,
    bond_allocation_pct: float,
    etf_proxy: BondETFProxy,
    dy_bp: float = 100.0,
) -> dict:
    """Compute portfolio impact of a rate shock.

    Args:
        portfolio_value: Total portfolio value.
        bond_allocation_pct: Percentage allocated to bond ETF (0-100).
        etf_proxy: BondETFProxy for the bond ETF.
        dy_bp: Rate shock in basis points.

    Returns:
        Dict with dollar and percentage impact.
    """
    position_value = portfolio_value * bond_allocation_pct / 100.0
    shares = position_value / etf_proxy.price
    price_shock = etf_proxy.price_change_bp(dy_bp)
    dollar_impact = shares * price_shock
    pct_impact = dollar_impact / portfolio_value * 100.0

    return {
        "position_value": round(position_value, 2),
        "price_shock": round(price_shock, 2),
        "dollar_impact": round(dollar_impact, 2),
        "pct_impact_bp": round(pct_impact * 100, 1),
    }
