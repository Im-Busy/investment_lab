"""
A11: Delta Hedging Strategies — Dynamic Portfolio Protection.

Simulates discrete delta hedging for options positions, computes P&L
attribution, gamma scalping, and portfolio-level net delta optimization.

Models:
  DeltaHedgeSimulator: Discrete delta hedging with transaction costs.
  DeltaHedgeResult: Per-path hedge P&L, turnover, and attribution.
  GammaScalper: Gamma/theta P&L decomposition for vol trading.
  PortfolioHedgeOptimizer: Net delta computation and rebalancing for basket positions.

Usage:
    >>> sim = DeltaHedgeSimulator(S0=100, K=100, T=0.25, r=0.05, sigma=0.20)
    >>> result = sim.run(n_paths=1000, n_steps=63)
    >>> result.hedge_pnl_mean
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import market_impact as _mi  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class DeltaHedgeResult:
    hedge_pnl: np.ndarray
    option_payoff: np.ndarray
    total_pnl: np.ndarray
    turnover: np.ndarray
    rebalance_count: np.ndarray
    option_premium: float

    @property
    def hedge_pnl_mean(self) -> float:
        return float(np.mean(self.hedge_pnl))

    @property
    def hedge_pnl_std(self) -> float:
        return float(np.std(self.hedge_pnl))

    @property
    def total_pnl_mean(self) -> float:
        return float(np.mean(self.total_pnl))

    @property
    def avg_turnover(self) -> float:
        return float(np.mean(self.turnover))

    @property
    def avg_rebalances(self) -> float:
        return float(np.mean(self.rebalance_count))

    def summary(self) -> dict:
        return {
            "option_premium": round(self.option_premium, 4),
            "hedge_pnl_mean": round(self.hedge_pnl_mean, 4),
            "hedge_pnl_std": round(self.hedge_pnl_std, 4),
            "total_pnl_mean": round(self.total_pnl_mean, 4),
            "avg_turnover": round(self.avg_turnover, 6),
            "avg_rebalances": round(self.avg_rebalances, 1),
        }


class DeltaHedgeSimulator:
    """Simulate discrete delta hedging for a short options position.

    Sells one option contract, delta-hedges with underlying shares at
    discrete intervals. Tracks hedge P&L, turnover, and final attribution.

    Args:
        S0: Initial spot price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Volatility (constant for BS delta).
        option_type: "call" or "put".
        q: Dividend yield.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float = 0.25,
        r: float = 0.05,
        sigma: float = 0.20,
        option_type: str = "call",
        q: float = 0.0,
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.q = q

    def _bs_delta(self, S: float, tau: float) -> float:
        """Black-Scholes delta."""
        import math

        if tau <= 0:
            if self.option_type == "call":
                return 1.0 if S > self.K else 0.0
            return -1.0 if S < self.K else 0.0

        d1 = (math.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (
            self.sigma * math.sqrt(tau)
        )
        from scipy.stats import norm

        delta = norm.cdf(d1)
        if self.option_type == "put":
            delta -= 1.0
            if self.q > 0:
                delta += self.q * tau  # approximate dividend adjustment
        return delta

    def _bs_price(self, S: float, tau: float) -> float:
        """Black-Scholes price."""
        from scipy.stats import norm
        import math

        if tau <= 0:
            if self.option_type == "call":
                return max(S - self.K, 0.0)
            return max(self.K - S, 0.0)

        d1 = (math.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * tau) / (
            self.sigma * math.sqrt(tau)
        )
        d2 = d1 - self.sigma * math.sqrt(tau)
        if self.option_type == "call":
            return S * norm.cdf(d1) - self.K * math.exp(-self.r * tau) * norm.cdf(d2)
        return self.K * math.exp(-self.r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def run(
        self,
        n_paths: int = 1000,
        n_steps: int = 63,
        transaction_cost: float = 0.001,
    ) -> DeltaHedgeResult:
        """Run delta hedging simulation.

        Args:
            n_paths: Number of Monte Carlo paths.
            n_steps: Number of rebalancing steps.
            transaction_cost: Per-share transaction cost as fraction of price.

        Returns:
            DeltaHedgeResult with per-path statistics.
        """
        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        S = np.full(n_paths, self.S0, dtype=np.float64)
        cash = np.zeros(n_paths, dtype=np.float64)
        shares = np.zeros(n_paths, dtype=np.float64)
        turnover = np.zeros(n_paths, dtype=np.float64)
        rebalance_count = np.zeros(n_paths, dtype=np.float64)

        # Initial option sale
        option_premium = self._bs_price(self.S0, self.T)
        cash[:] = option_premium

        # Initial delta hedge
        delta = self._bs_delta(self.S0, self.T)
        shares[:] = -delta
        cash[:] += delta * self.S0

        for step in range(n_steps):
            tau = self.T - step * dt
            if tau <= 0:
                break

            # Advance underlying
            Z = np.random.randn(n_paths)
            S *= np.exp(drift + vol * Z)

            # Compute new delta
            new_delta = np.array([self._bs_delta(s, tau - dt) for s in S])

            # Rebalance
            dshares = new_delta + shares  # new_delta is target, shares is current (negative)
            trade_value = np.abs(dshares) * S
            turnover += trade_value / self.S0
            rebalance_count += np.abs(dshares) > 1e-8

            cost = np.abs(dshares) * S * transaction_cost
            cash += dshares * S - cost
            shares = -new_delta  # negative = short option, long shares

        # Final settlement
        if self.option_type == "call":
            payoff = np.maximum(S - self.K, 0)
        else:
            payoff = np.maximum(self.K - S, 0)

        final_value = cash + shares * S - payoff
        hedge_pnl = final_value

        return DeltaHedgeResult(
            hedge_pnl=hedge_pnl,
            option_payoff=payoff,
            total_pnl=hedge_pnl,
            turnover=turnover,
            rebalance_count=rebalance_count,
            option_premium=option_premium,
        )


class GammaScalper:
    """Gamma/theta P&L attribution for options positions.

    Decomposes daily option P&L into delta, gamma, theta, and unexplained
    components following the standard dealer hedging framework.

    Args:
        S0: Initial spot.
        K: Strike.
        T: Days to expiration.
        r: Risk-free rate.
        sigma: Implied volatility.
        option_type: "call" or "put".
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float = 30.0,
        r: float = 0.05,
        sigma: float = 0.20,
        option_type: str = "call",
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

    def _compute(self, S_arr: np.ndarray, dt: float = 1 / 252) -> dict:
        """Compute P&L attribution given a price path.

        Args:
            S_arr: Array of underlying prices.
            dt: Time step in years.

        Returns:
            Dict with delta_pnl, gamma_pnl, theta_pnl, actual_pnl, unexplained.
        """
        n = len(S_arr)
        delta_pnl_arr = np.zeros(n - 1)
        gamma_pnl_arr = np.zeros(n - 1)
        theta_pnl_arr = np.zeros(n - 1)
        actual_pnl_arr = np.zeros(n - 1)
        unexplained_arr = np.zeros(n - 1)

        tau = self.T / 365.25  # convert days to years initially

        for i in range(n - 1):
            if tau <= 0:
                break

            S_t = S_arr[i]
            S_next = S_arr[i + 1]
            dS = S_next - S_t

            d1 = (np.log(S_t / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (
                self.sigma * np.sqrt(tau)
            )
            from scipy.stats import norm

            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S_t * self.sigma * np.sqrt(tau))
            theta = -S_t * norm.pdf(d1) * self.sigma / (
                2 * np.sqrt(tau)
            ) - self.r * self.K * np.exp(-self.r * tau) * norm.cdf(d1 - self.sigma * np.sqrt(tau))

            if self.option_type == "put":
                delta -= 1.0
                theta += self.r * self.K * np.exp(-self.r * tau)

            delta_pnl_arr[i] = delta * dS
            gamma_pnl_arr[i] = 0.5 * gamma * dS**2
            theta_pnl_arr[i] = theta * dt

            actual_pnl_arr[i] = delta_pnl_arr[i] + gamma_pnl_arr[i] + theta_pnl_arr[i]

            tau -= dt

        return {
            "delta_pnl": float(np.sum(delta_pnl_arr)),
            "gamma_pnl": float(np.sum(gamma_pnl_arr)),
            "theta_pnl": float(np.sum(theta_pnl_arr)),
            "total_attributed": float(np.sum(actual_pnl_arr)),
        }

    def simulate(self, n_paths: int = 1000) -> dict:
        """Simulate gamma scalping P&L under GBM.

        Args:
            n_paths: Number of simulation paths.

        Returns:
            Dict with mean and std of each P&L component.
        """
        n_steps = max(int(self.T), 1)
        dt = self.T / 365.25 / n_steps

        results = {"delta_pnl": [], "gamma_pnl": [], "theta_pnl": [], "total": []}

        for _ in range(n_paths):
            S = self.S0 * np.exp(
                np.cumsum(
                    (self.r - 0.5 * self.sigma**2) * dt
                    + self.sigma * np.sqrt(dt) * np.random.randn(n_steps + 1)
                )
            )
            S[0] = self.S0
            pnl = self._compute(S, dt)
            results["delta_pnl"].append(pnl["delta_pnl"])
            results["gamma_pnl"].append(pnl["gamma_pnl"])
            results["theta_pnl"].append(pnl["theta_pnl"])
            results["total"].append(pnl["total_attributed"])

        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in results.items()}


class PortfolioHedgeOptimizer:
    """Compute net delta and rebalancing requirements for multi-option portfolios.

    Aggregates delta across multiple option positions and computes
    hedge shares needed to achieve target net delta.

    Args:
        positions: List of (quantity, call/put, strike, maturity_years) tuples.
        S0: Current spot price.
        r: Risk-free rate.
        sigma: Volatility.
    """

    def __init__(
        self,
        S0: float,
        r: float = 0.05,
        sigma: float = 0.20,
    ):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self._positions: List[Tuple[float, str, float, float]] = []

    def add_position(
        self, quantity: float, option_type: str, strike: float, maturity_years: float
    ) -> None:
        """Add an option position to the portfolio.

        Args:
            quantity: Number of contracts (positive = long, negative = short).
            option_type: "call" or "put".
            strike: Strike price.
            maturity_years: Time to expiration in years.
        """
        self._positions.append((quantity, option_type, strike, maturity_years))

    def net_delta(self, S: Optional[float] = None) -> float:
        """Compute net portfolio delta.

        Args:
            S: Current spot (defaults to S0).

        Returns:
            Net delta (shares needed per option contract).
        """
        spot = S if S is not None else self.S0
        total = 0.0
        from scipy.stats import norm

        for qty, opt_type, K, T in self._positions:
            if T <= 0:
                if opt_type == "call":
                    delta = 1.0 if spot > K else 0.0
                else:
                    delta = -1.0 if spot < K else 0.0
            else:
                d1 = (np.log(spot / K) + (self.r + 0.5 * self.sigma**2) * T) / (
                    self.sigma * np.sqrt(T)
                )
                delta = norm.cdf(d1)
                if opt_type == "put":
                    delta -= 1.0
            total += qty * delta
        return total

    def hedge_shares(self, target_delta: float = 0.0) -> float:
        """Shares of underlying needed to hedge to target delta.

        Args:
            target_delta: Desired net portfolio delta.

        Returns:
            Number of shares to buy (positive) or sell (negative).
        """
        net = self.net_delta()
        return target_delta - net

    def summary(self) -> dict:
        """Portfolio hedge summary."""
        net = self.net_delta()
        shares = self.hedge_shares()
        return {
            "spot": round(self.S0, 2),
            "net_delta": round(net, 4),
            "hedge_shares": round(shares, 2),
            "num_positions": len(self._positions),
        }
