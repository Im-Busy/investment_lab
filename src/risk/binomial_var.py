"""P24-33: Binomial VAR for event-driven risk measurement.

Forward-looking risk model for event-driven strategies:
  N trades × break probability → binomial loss distribution
  → VaR / CVaR at any confidence level

Unlike historical VaR (which assumes returns are from a continuous distribution),
binomial VAR explicitly models the discrete win/loss nature of individual trades.
More accurate for event-driven strategies where each signal is a discrete binary
decision with a known (or estimated) success probability.

Reference: E6 in Master Comparison Report, "Risk Management for Event-Driven Funds".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BinomialVaRResult:
    """Binomial VAR/CVaR estimation result for an event-driven portfolio."""

    n_trades: int
    """Expected number of positions (deals) in the period."""

    success_prob: float
    """Probability of a successful (profitable) trade."""

    avg_win_pct: float
    """Average gain per winning trade (as decimal, e.g., 0.02 = 2%)."""

    avg_loss_pct: float
    """Average loss per losing trade (as positive decimal, e.g., 0.015 = 1.5%)."""

    confidence: float
    """Confidence level for VaR (e.g., 0.95)."""

    var_pct: float
    """Value-at-Risk as percentage of capital."""

    cvar_pct: float
    """Conditional VaR (expected loss if worse than VaR)."""

    worst_case_pct: float
    """Worst-case loss if ALL trades lose."""

    expected_loss_pct: float
    """Expected loss from the binomial distribution (can be negative = profit)."""

    loss_breakeven_k: int
    """Minimum number of losses needed to break even."""

    prob_exceed_var: float
    """Actual probability of loss exceeding VaR (≤ 1 - confidence)."""

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_trades": self.n_trades,
            "success_prob": round(self.success_prob, 4),
            "avg_win_pct": round(self.avg_win_pct, 4),
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "confidence": self.confidence,
            "var_pct": round(self.var_pct, 4),
            "cvar_pct": round(self.cvar_pct, 4),
            "worst_case_pct": round(self.worst_case_pct, 4),
            "expected_loss_pct": round(self.expected_loss_pct, 4),
            "loss_breakeven_k": self.loss_breakeven_k,
            "prob_exceed_var": round(self.prob_exceed_var, 6),
        }


def compute_binomial_var(
    n_trades: int,
    success_prob: float,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = 0.015,
    confidence: float = 0.95,
) -> BinomialVaRResult:
    """Compute binomial VaR/CVaR for an event-driven strategy.

    Models the number of losing trades as Binomial(n_trades, 1-success_prob).
    Each loss costs avg_loss_pct, each win earns avg_win_pct.

    Args:
        n_trades: Expected number of positions in the evaluation period.
        success_prob: Probability a trade is profitable (win rate).
        avg_win_pct: Average profit per winning trade (as decimal).
        avg_loss_pct: Average loss per losing trade (as positive decimal).
        confidence: VaR confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        BinomialVaRResult with VaR, CVaR, and diagnostic metrics.

    Example:
        >>> result = compute_binomial_var(n_trades=50, success_prob=0.60, avg_win_pct=0.02, avg_loss_pct=0.015)
        >>> result.var_pct  # 95% VaR as percent of capital
    """
    if not 0 < success_prob < 1:
        raise ValueError(f"success_prob must be in (0, 1), got {success_prob}")
    if n_trades < 1:
        raise ValueError(f"n_trades must be >= 1, got {n_trades}")
    if avg_win_pct < 0 or avg_loss_pct < 0:
        raise ValueError("avg_win_pct and avg_loss_pct must be non-negative")

    break_prob = 1.0 - success_prob
    rv = stats.binom(n=n_trades, p=break_prob)

    k_values = np.arange(0, n_trades + 1)
    losses = k_values * avg_loss_pct
    wins = (n_trades - k_values) * avg_win_pct
    pnl = wins - losses  # positive = profit, negative = loss
    probs = rv.pmf(k_values)

    # Sort PnL from worst to best for VaR calculation
    sort_idx = np.argsort(pnl)
    sorted_pnl = pnl[sort_idx]
    sorted_probs = probs[sort_idx]

    cumulative = np.cumsum(sorted_probs)
    var_idx = int(np.searchsorted(cumulative, 1 - confidence, side="left"))
    var_idx = min(var_idx, n_trades)

    var_pct = -sorted_pnl[var_idx]  # flip sign: VaR is positive loss
    cvar_mask = np.arange(len(sorted_pnl)) <= var_idx
    cvar_pct = -float(np.average(sorted_pnl[cvar_mask], weights=sorted_probs[cvar_mask]))

    worst_case_pct = -pnl[-1]  # all loses → worst PnL at k=n_trades
    expected_loss_pct = -float(np.dot(pnl, probs))  # expected PnL (neg = expected profit)

    break_even = (
        n_trades * avg_win_pct / (avg_win_pct + avg_loss_pct)
        if (avg_win_pct + avg_loss_pct) > 0
        else float("inf")
    )
    loss_breakeven_k = int(np.ceil(break_even))

    prob_exceed_var = 1.0 - float(rv.cdf(var_idx - 1)) if var_idx > 0 else 1.0

    return BinomialVaRResult(
        n_trades=int(n_trades),
        success_prob=float(success_prob),
        avg_win_pct=float(avg_win_pct),
        avg_loss_pct=float(avg_loss_pct),
        confidence=float(confidence),
        var_pct=round(var_pct, 6),
        cvar_pct=round(cvar_pct, 6),
        worst_case_pct=round(worst_case_pct, 6),
        expected_loss_pct=round(expected_loss_pct, 6),
        loss_breakeven_k=loss_breakeven_k,
        prob_exceed_var=round(prob_exceed_var, 8),
    )


def size_position_binomial(
    capital: float,
    n_trades: int,
    success_prob: float,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = 0.015,
    max_var_pct: float = 0.05,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Determine position size based on binomial VaR constraint.

    Given a maximum allowable VaR as % of capital, compute the maximum
    position size per trade that stays within the risk limit.

    Args:
        capital: Total capital available.
        n_trades: Expected number of concurrent positions.
        success_prob: Win probability per trade.
        avg_win_pct: Average win size (as decimal).
        avg_loss_pct: Average loss size (as positive decimal).
        max_var_pct: Maximum allowable VaR as fraction of capital.
        confidence: VaR confidence level.

    Returns:
        Dict with max_size_per_trade, total_committed, and risk_utilization.
    """
    result = compute_binomial_var(
        n_trades=n_trades,
        success_prob=success_prob,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        confidence=confidence,
    )

    if result.var_pct <= 0:
        return {
            "max_size_per_trade": capital,
            "total_committed": capital * n_trades,
            "risk_utilization": 0.0,
            "status": "no_risk",
        }

    scale = max_var_pct / result.var_pct
    max_size_per_trade = capital * min(scale, 1.0)

    return {
        "max_size_per_trade": round(max_size_per_trade, 2),
        "total_committed": round(max_size_per_trade * n_trades, 2),
        "risk_utilization": round(1.0 / scale, 4) if scale > 1 else 1.0,
        "status": "ok",
    }
