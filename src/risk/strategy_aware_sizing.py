"""Strategy-aware position sizing using Kelly-derived fractions.

Computes position size per strategy type based on rolling win rate,
edge estimates, and strategy-specific Kelly fractions.

Usage:
    from src.risk.strategy_aware_sizing import StrategyAwareSizer

    sizer = StrategyAwareSizer(strategy_type="trend", capital=100_000)
    size_pct = sizer.compute(win_rate=0.55, avg_win=200.0, avg_loss=-150.0)
    # size_pct = 0.12  → 12% of capital
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

STRATEGY_KELLY: dict[str, float] = {
    "trend": 0.5,
    "mean_reversion": 0.25,
    "pairs": 0.5,
    "ml": 0.25,
}


class StrategyAwareSizer:
    """Position sizing with Kelly-derived fractions per strategy type.

    Parameters:
        strategy_type: One of 'trend', 'mean_reversion', 'pairs', 'ml'.
        capital: Total capital in dollars.
        max_position_pct: Maximum position size as % of capital (default 0.25).
        min_position_pct: Minimum position size as % of capital (default 0.02).
        half_kelly: Use half-Kelly for conservative sizing (default True).
    """

    def __init__(
        self,
        strategy_type: str = "trend",
        capital: float = 100_000.0,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.02,
        half_kelly: bool = True,
    ) -> None:
        self.strategy_type = strategy_type
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.half_kelly = half_kelly

    def _kelly_fraction(self) -> float:
        return STRATEGY_KELLY.get(self.strategy_type, 0.25)

    def compute(
        self,
        win_rate: float,
        avg_win: float = 1.0,
        avg_loss: float = -1.0,
    ) -> float:
        """Compute position size as fraction of capital.

        Args:
            win_rate: Rolling win rate (0.0 to 1.0).
            avg_win: Average winning trade amount.
            avg_loss: Average losing trade amount (negative).

        Returns:
            Position size as fraction of capital (e.g. 0.10 = 10%).
        """
        if avg_loss >= 0:
            return self.min_position_pct

        loss_rate = 1.0 - max(0.0, min(1.0, win_rate))
        avg_gain = max(avg_win, 0.0)
        avg_cost = abs(avg_loss)

        if avg_cost < 1e-10:
            return self.min_position_pct

        b = avg_gain / avg_cost
        p = max(0.0, min(1.0, win_rate))
        q = loss_rate

        kelly = (p * b - q) / max(b, 1e-10) if b > 1e-10 else 0.0
        kelly = max(0.0, min(1.0, kelly))

        fraction = kelly * self._kelly_fraction()
        if self.half_kelly:
            fraction *= 0.5

        return max(self.min_position_pct, min(self.max_position_pct, fraction))

    def compute_dollar(
        self,
        win_rate: float,
        avg_win: float = 1.0,
        avg_loss: float = -1.0,
    ) -> float:
        """Compute position size in dollars."""
        return self.compute(win_rate, avg_win, avg_loss) * self.capital

    def compute_shares(
        self,
        price: float,
        win_rate: float,
        avg_win: float = 1.0,
        avg_loss: float = -1.0,
    ) -> int:
        """Compute number of shares to trade."""
        dollar_size = self.compute_dollar(win_rate, avg_win, avg_loss)
        if price <= 0:
            return 0
        return max(0, int(dollar_size / price))
