"""
Kelly Criterion Position Sizing — Direction C Phase 5.

Based on "Investing Is Compression" (Stiffelman, NAND Capital):
- Classic Kelly: f* = (p * b - q) / b  (probability * payoff - loss_prob) / payoff
- Information-theoretic decomposition: growth = money_term + entropy_term - divergence
- General form for multi-outcome bets
- Winner fraction heuristic with entropy bound

Also implements practical variants:
- Half-Kelly (fraction=0.5) and quarter-Kelly (fraction=0.25) for conservative sizing
- Rolling IS edge estimation (no look-ahead) via expanding window statistics
- Convex Kelly fraction cap at 25% of equity per single position
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


EDGE_MIN_WINDOW = 20
FULL_KELLY_CAP = 0.25
HALF_KELLY_CAP = 0.125


@dataclass
class KellyEdge:
    """Edge estimation for Kelly criterion."""

    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    num_trades: int


@dataclass
class KellyAllocation:
    """Result of Kelly criterion calculation."""

    full_kelly_fraction: float
    adjusted_fraction: float
    edge: KellyEdge
    method: str  # "classic", "information", "winner_fraction"
    metadata: dict = field(default_factory=dict)


class KellyAllocator:
    """Kelly criterion position sizing with edge estimation.

    Implements three Kelly variants from "Investing Is Compression":
    1. Classic Kelly for binary outcomes (win/loss)
    2. Information-theoretic Kelly for multi-outcome bets
    3. Winner fraction heuristic with entropy bound

    Parameters:
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter-Kelly).
            Lower values reduce variance at cost of lower growth.
        max_allocation: Hard cap on fraction of equity per position (default 0.25).
        min_allocation: Minimum allocation fraction (default 0.0, no minimum).
        method: "classic" | "information" | "winner_fraction"
        edge_lookback: Trading days for rolling edge estimation (default 252).
        edge_min_trades: Minimum trades required for edge estimation (default 5).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_allocation: float = 0.25,
        min_allocation: float = 0.0,
        method: str = "classic",
        edge_lookback: int = 252,
        edge_min_trades: int = 5,
    ) -> None:
        self.kelly_fraction = max(0.0, min(1.0, kelly_fraction))
        self.max_allocation = max(0.0, min(1.0, max_allocation))
        self.min_allocation = max(0.0, min(self.max_allocation, min_allocation))
        self.method = method
        self.edge_lookback = edge_lookback
        self.edge_min_trades = edge_min_trades

    def estimate_edge_from_trades(
        self,
        trade_returns: pd.Series,
        current_idx: Optional[int] = None,
    ) -> Optional[KellyEdge]:
        """Estimate edge (win_rate, avg_win, avg_loss) from trade history.

        Uses expanding window up to current_idx for no-look-ahead safety.
        If current_idx is None, uses the full series.

        Returns None if insufficient trades.
        """
        if current_idx is not None:
            window = trade_returns.iloc[:current_idx]
        else:
            window = trade_returns

        if len(window) < self.edge_min_trades:
            return None

        wins = window[window > 0]
        losses = window[window < 0]
        total = len(window)

        if total == 0:
            return None

        win_rate = len(wins) / total
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

        if avg_loss == 0:
            return None

        profit_factor = (
            (len(wins) * avg_win) / (len(losses) * avg_loss) if len(losses) > 0 else float("inf")
        )

        return KellyEdge(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            num_trades=total,
        )

    def estimate_edge_from_probability(
        self,
        model_probability: float,
        avg_win_pct: float = 0.05,
        avg_loss_pct: float = 0.03,
    ) -> KellyEdge:
        """Estimate edge from model probability (e.g., PatternClassifier output).

        Uses model calibration: model_probability approximates win_rate.
        Default win/loss ratios (5%/3%) are sensible defaults for ATR-based stops.
        """
        prob = max(0.01, min(0.99, model_probability))
        avg_win = abs(avg_win_pct)
        avg_loss = abs(avg_loss_pct)

        if avg_loss == 0:
            avg_loss = 0.001

        pf = (prob * avg_win) / ((1 - prob) * avg_loss) if prob < 1.0 else float("inf")

        return KellyEdge(
            win_rate=prob,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=pf,
            num_trades=max(self.edge_min_trades, 1),
        )

    def classic_kelly(self, edge: KellyEdge) -> float:
        """Classic Kelly for binary outcome: f* = (p * b - q) / b.

        Where p=win_rate, b=avg_win/avg_loss, q=1-p.
        Returns fraction in [0, 1].
        """
        if edge.avg_loss == 0:
            return 0.0

        b = edge.avg_win / edge.avg_loss  # payoff ratio
        p = edge.win_rate
        q = 1 - p

        f_star = (p * b - q) / b
        return max(0.0, min(1.0, f_star))

    def information_kelly(self, edge: KellyEdge) -> float:
        """Information-theoretic Kelly: factor into money + entropy - divergence.

        From the paper Section 4 decomposition:
        Growth = E[log(1 + f * r)] where r is return distribution.
        Numerically optimize f to maximize expected log growth.

        For discrete outcomes with k possible returns r_i and probabilities p_i:
        f* = argmax_f Σ p_i * log(1 + f * r_i)
        """
        if edge.avg_loss == 0:
            return 0.0

        # Build return distribution from edge
        win_return = edge.avg_win
        loss_return = -edge.avg_loss
        p_win = edge.win_rate
        p_loss = 1 - p_win

        candidates = np.linspace(0.0, 1.0, 1001)
        best_f = 0.0
        best_growth = -np.inf

        for f in candidates:
            # Expected log growth
            growth_win = np.log(1 + f * win_return) if (1 + f * win_return) > 0 else -np.inf
            growth_loss = np.log(1 + f * loss_return) if (1 + f * loss_return) > 0 else -np.inf
            expected_growth = p_win * growth_win + p_loss * growth_loss

            if expected_growth > best_growth:
                best_growth = expected_growth
                best_f = f

        return max(0.0, min(1.0, best_f))

    def winner_fraction_kelly(self, edge: KellyEdge) -> float:
        """Winner fraction heuristic from Section 6 of the paper.

        Allocates capital in proportion to each asset's probability of
        dominating the candidate set. The growth shortfall is bounded by
        the entropy of the winner fraction distribution.

        For a single asset, this simplifies to: f = win_rate (since that's
        the probability it dominates a candidate set of itself vs cash).
        """
        raw_fraction = edge.win_rate
        # Entropy-based penalty: H = -p*log(p) - (1-p)*log(1-p)
        # Higher entropy (closer to 0.5) -> higher uncertainty -> reduce allocation
        p = max(0.01, min(0.99, edge.win_rate))
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        max_entropy = 1.0  # entropy at p=0.5

        entropy_penalty = entropy / max_entropy
        adjusted = raw_fraction * (1.0 - 0.3 * entropy_penalty)

        return max(0.0, min(1.0, adjusted))

    def compute(
        self,
        edge: KellyEdge,
        current_equity: float = 100_000.0,
    ) -> KellyAllocation:
        """Compute Kelly allocation for a given edge.

        Args:
            edge: KellyEdge with win_rate, avg_win, avg_loss, profit_factor.
            current_equity: Current portfolio equity (for metadata only).

        Returns:
            KellyAllocation with full and adjusted fractions.
        """
        if self.method == "classic":
            full_kelly = self.classic_kelly(edge)
        elif self.method == "information":
            full_kelly = self.information_kelly(edge)
        elif self.method == "winner_fraction":
            full_kelly = self.winner_fraction_kelly(edge)
        else:
            full_kelly = self.classic_kelly(edge)

        adjusted = full_kelly * self.kelly_fraction
        adjusted = max(0.0, min(adjusted, self.max_allocation))
        if adjusted > 0 and adjusted < self.min_allocation:
            adjusted = self.min_allocation

        return KellyAllocation(
            full_kelly_fraction=round(full_kelly, 6),
            adjusted_fraction=round(adjusted, 6),
            edge=edge,
            method=self.method,
            metadata={
                "kelly_fraction_applied": self.kelly_fraction,
                "equity": current_equity,
                "trade_capital": round(current_equity * adjusted, 2),
            },
        )

    def compute_return_distribution(
        self,
        trade_returns: list[float],
    ) -> Optional[KellyAllocation]:
        """Compute Kelly from empirical return distribution.

        For multi-outcome bets (not just binary win/loss), this maximizes
        expected log growth over the empirical distribution using numeric
        optimization.

        Args:
            trade_returns: List of per-trade return fractions (e.g., [0.05, -0.02, 0.03, ...]).

        Returns:
            KellyAllocation or None if insufficient data.
        """
        if len(trade_returns) < self.edge_min_trades:
            return None

        returns = np.asarray(trade_returns)
        n = len(returns)

        # Numeric optimization over f in [0, 1]
        candidates = np.linspace(0.0, 1.0, 1001)
        best_f = 0.0
        best_growth = -np.inf

        for f in candidates:
            growth_vals = np.log(1 + f * returns)
            # filter out invalid (ruin)
            growth_vals = growth_vals[np.isfinite(growth_vals)]
            if len(growth_vals) == 0:
                continue
            expected_growth = growth_vals.mean()

            if expected_growth > best_growth:
                best_growth = expected_growth
                best_f = f

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

        edge = KellyEdge(
            win_rate=len(wins) / n if n > 0 else 0.0,
            avg_win=avg_win,
            avg_loss=avg_loss if avg_loss > 0 else 0.001,
            profit_factor=(len(wins) * avg_win) / (len(losses) * avg_loss)
            if len(losses) > 0 and avg_loss > 0
            else 0.0,
            num_trades=n,
        )

        adjusted = best_f * self.kelly_fraction
        adjusted = max(0.0, min(adjusted, self.max_allocation))

        return KellyAllocation(
            full_kelly_fraction=round(best_f, 6),
            adjusted_fraction=round(adjusted, 6),
            edge=edge,
            method="information_empirical",
            metadata={
                "kelly_fraction_applied": self.kelly_fraction,
                "expected_log_growth": round(best_growth, 6),
                "evaluations": len(candidates),
            },
        )

    def should_trade(self, allocation: KellyAllocation) -> bool:
        """Determine if the trade has positive edge worth taking.

        Returns True if adjusted Kelly fraction > 0 and edge is positive.
        """
        if allocation.adjusted_fraction <= 0:
            return False
        if allocation.edge.win_rate <= 0 or allocation.edge.profit_factor < 1.0:
            return False
        return True

    def compute_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        allocation: KellyAllocation,
    ) -> int:
        """Convert Kelly fraction to actual share count.

        Args:
            equity: Current account equity.
            entry_price: Planned entry price.
            stop_price: Stop loss price.
            allocation: KellyAllocation from compute().

        Returns:
            Number of shares (integer >= 1).
        """
        if allocation.adjusted_fraction <= 0:
            return 0

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0

        risk_capital = equity * allocation.adjusted_fraction
        shares = int(risk_capital / risk_per_share)
        return max(shares, 0)


def compute_kelly_from_history(
    trade_returns,
    kelly_fraction: float = 0.5,
    method: str = "classic",
) -> Optional[dict]:
    """Convenience: compute Kelly fraction from trade return history.

    Args:
        trade_returns: List or Series of per-trade return percentages (e.g., 0.05 = +5%).
        kelly_fraction: Fraction of full Kelly (0.5 = half-Kelly).
        method: "classic" | "information".

    Returns:
        Dict with 'full_kelly', 'adjusted_kelly', 'edge', or None.
    """
    if isinstance(trade_returns, pd.Series):
        rets = list(trade_returns)
    else:
        rets = list(trade_returns)

    allocator = KellyAllocator(kelly_fraction=kelly_fraction, method=method)
    result = allocator.compute_return_distribution(rets)
    if result is None:
        return None
    return {
        "full_kelly": result.full_kelly_fraction,
        "adjusted_kelly": result.adjusted_fraction,
        "edge": result.edge,
        "should_trade": allocator.should_trade(result),
    }


def compute_kelly_from_probability(
    prob: float,
    avg_win_pct: float = 0.05,
    avg_loss_pct: float = 0.03,
    kelly_fraction: float = 0.5,
    method: str = "classic",
) -> dict:
    """Convenience: compute Kelly fraction from model probability.

    Args:
        prob: Model probability of profitable outcome.
        avg_win_pct: Typical win size (e.g., 0.05 = +5% from TP).
        avg_loss_pct: Typical loss size (e.g., 0.03 = -3% from SL).
        kelly_fraction: Fraction of full Kelly.
        method: "classic" | "information".

    Returns:
        Dict with 'full_kelly', 'adjusted_kelly', 'edge'.
    """
    allocator = KellyAllocator(kelly_fraction=kelly_fraction, method=method)
    edge = allocator.estimate_edge_from_probability(
        model_probability=prob,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
    )
    result = allocator.compute(edge)
    return {
        "full_kelly": result.full_kelly_fraction,
        "adjusted_kelly": result.adjusted_fraction,
        "edge": result.edge,
        "should_trade": allocator.should_trade(result),
    }


def estimate_minimum_capital(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    target_annual_return: float = 10_000,
    trades_per_year: int = 12,
    kelly_fraction: float = 0.5,
) -> float:
    """Estimate minimum capital for a target annual return.

    Uses half-Kelly sizing to compute the capital base needed to produce
    a given dollar return per year.

    Args:
        win_rate: Probability of winning a trade (0-1).
        avg_win: Average positive return per trade (e.g., 0.05).
        avg_loss: Average negative return per trade (must be negative).
        target_annual_return: Desired annual dollar return.
        trades_per_year: Expected number of trades per year.
        kelly_fraction: Kelly fraction multiplier.

    Returns:
        Minimum capital required, or inf if edge is negative.
    """
    allocator = KellyAllocator(kelly_fraction=kelly_fraction)
    edge = allocator.estimate_edge_from_probability(
        model_probability=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=abs(avg_loss),
    )
    if edge.win_rate <= 0:
        return float("inf")

    result = allocator.compute(edge)
    trade_capital = result.metadata.get("trade_capital", 0.0)
    expected_return = edge.win_rate * edge.avg_win + (1 - edge.win_rate) * edge.avg_loss
    expected_return_per_trade = result.adjusted_fraction * max(expected_return, 0.0)
    if expected_return_per_trade <= 0:
        return float("inf")

    return target_annual_return / (trades_per_year * expected_return_per_trade)


def format_kelly_report(result) -> str:
    """Format a Kelly allocation as a readable report string.

    Args:
        result: KellyAllocation from KellyAllocator.compute(), or dict from
                compute_kelly_from_history/compute_kelly_from_probability.

    Returns:
        Multi-line formatted string.
    """
    if isinstance(result, dict):
        full_kelly = result.get("full_kelly", 0.0)
        adjusted = result.get("adjusted_kelly", 0.0)
        edge_dict = result.get("edge", {})
        trade_capital = result.get("trade_capital", 0)
        method = result.get("method", "classic")
        win_rate = (
            edge_dict.get("win_rate", 0.0)
            if isinstance(edge_dict, dict)
            else getattr(edge_dict, "win_rate", 0.0)
        )
        avg_win = (
            edge_dict.get("avg_win", 0.0)
            if isinstance(edge_dict, dict)
            else getattr(edge_dict, "avg_win", 0.0)
        )
        avg_loss = (
            edge_dict.get("avg_loss", 0.0)
            if isinstance(edge_dict, dict)
            else getattr(edge_dict, "avg_loss", 0.0)
        )
        profit_factor = (
            edge_dict.get("profit_factor", 0.0)
            if isinstance(edge_dict, dict)
            else getattr(edge_dict, "profit_factor", 0.0)
        )
        expected_growth = 0.0
    else:
        e = getattr(result, "edge", None)
        full_kelly = getattr(result, "full_kelly_fraction", 0.0)
        adjusted = getattr(result, "adjusted_fraction", 0.0)
        trade_capital = (
            result.metadata.get("trade_capital", 0) if hasattr(result, "metadata") else 0
        )
        method = getattr(result, "method", "classic")
        win_rate = e.win_rate if e else 0.0
        avg_win = e.avg_win if e else 0.0
        avg_loss = e.avg_loss if e else 0.0
        profit_factor = e.profit_factor if e else 0.0
        odds = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expected_growth = win_rate * odds - (1 - win_rate)

    lines = [
        "=" * 55,
        " KELLY POSITION SIZING REPORT (Phase 12c P3-2)",
        "=" * 55,
        "",
        f"  Win Rate:          {float(win_rate):.1%}",
        f"  Avg Win:           {float(avg_win):+.1%}",
        f"  Avg Loss:          {float(avg_loss):+.1%}",
        f"  Profit Factor:     {float(profit_factor):.2f}",
        f"  Edge (p*b-q):      {float(expected_growth):+.4f}",
        "",
        "  Recommended Allocation:",
        f"    Full-Kelly:      {float(full_kelly):.1%} of capital",
        f"    Adjusted-Kelly:  {float(adjusted):.1%} of capital",
        f"    Method:          {method}",
        "",
        f"  Trade Capital:     ${float(trade_capital):,.0f}",
        "",
        "  Guidelines:",
        "    - Adjusted fraction > 0.02: tradeable",
        "    - Adjusted fraction > 0.05: strong edge",
        "    - Half-Kelly balances growth vs drawdown",
        "    - Quarter-Kelly for risk-averse or small samples",
        "=" * 55,
    ]
    return "\n".join(lines)
