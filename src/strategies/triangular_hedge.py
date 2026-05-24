"""P24-25 through P24-28: Triangular hedging framework.

Salvageable hedging system based on Ciacci 2020 correlation methodology.
Components:
  P24-25: Scientific pair selection via negative correlation potential
  P24-26: EMA-gated triangular hedge entry
  P24-27: Binary state-machine hedge pattern (alternator)
  P24-28: Hedge-only variant (remove averaging — 61% DD cause)

The averaging/martingale component is STRICTLY REMOVED
(confirmed destructive by 2+ papers with 61% DD empirical).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HedgeState(Enum):
    """Binary state machine states for triangular hedge."""

    NEUTRAL = auto()
    LONG_LEG1 = auto()
    LONG_LEG2 = auto()
    HEDGED = auto()  # Both legs active


@dataclass
class HedgePair:
    """Correlation-based pair for triangular hedge."""

    asset_a: str
    asset_b: str
    correlation: float
    hedge_ratio: float
    is_eligible: bool


@dataclass
class HedgeSignal:
    """Triangular hedge trading signal."""

    state: HedgeState
    action: str  # "open_long_a", "open_long_b", "close_all", "hold"
    size_multiplier: float
    confidence: float


# ── P24-25: Correlation-based pair selection ──


def compute_correlation_matrix(
    prices: Dict[str, pd.Series],
    window: int = 60,
) -> pd.DataFrame:
    """Compute rolling correlation matrix for candidate hedge pairs."""
    df = pd.DataFrame({k: v.pct_change().dropna() for k, v in prices.items()})
    if len(df) < window:
        return pd.DataFrame()
    return df.rolling(window).corr().iloc[-len(prices) :, -len(prices) :]


def select_hedge_pairs(
    prices: Dict[str, pd.Series],
    min_correlation: float = -0.3,
    max_correlation: float = -0.05,
    window: int = 60,
) -> List[HedgePair]:
    """Select pairs with negative correlation potential suitable for hedging.

    Args:
        prices: Dict[ticker, price_series].
        min_correlation: Minimum negative correlation to consider.
        max_correlation: Maximum negative correlation (too negative = cointegrated, not hedge).
        window: Rolling window for correlation.

    Returns:
        List of HedgePair with correlation and hedge ratio.
    """
    corr_matrix = compute_correlation_matrix(prices, window)
    if corr_matrix.empty:
        return []

    pairs: List[HedgePair] = []
    tickers = list(prices.keys())

    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            if a not in corr_matrix.index or b not in corr_matrix.columns:
                continue
            corr = float(corr_matrix.loc[a, b])

            hedge_ratio = 1.0
            if abs(corr) > 1e-10:
                std_a = float(prices[a].pct_change().std())
                std_b = float(prices[b].pct_change().std())
                hedge_ratio = std_a / (std_b + 1e-10)
                hedge_ratio = float(np.clip(hedge_ratio, 0.3, 3.0))

            is_eligible = max_correlation <= corr <= min_correlation

            pairs.append(
                HedgePair(
                    asset_a=a,
                    asset_b=b,
                    correlation=corr,
                    hedge_ratio=hedge_ratio,
                    is_eligible=is_eligible,
                )
            )

    return sorted(pairs, key=lambda p: p.correlation)


# ── P24-26: EMA-gated hedge entry ──


def ema_gate_check(
    price: pd.Series,
    fast_period: int = 10,
    slow_period: int = 30,
) -> int:
    """EMA crossover gate: fast > slow → bullish (1), fast < slow → bearish (-1).

    Returns 0 when EMA crossover suggests neutral/not enough data.
    """
    if len(price) < slow_period:
        return 0

    ema_fast = price.ewm(span=fast_period, adjust=False).mean()
    ema_slow = price.ewm(span=slow_period, adjust=False).mean()

    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        return 1
    return -1


# ── P24-27: Binary state machine hedge ──


class TriangularHedgeStateMachine:
    """Two alternating states for triangular hedge.

    State transitions:
      NEUTRAL → LONG_LEG1 (when ema_gate=bull AND negative_corr_pair)
      LONG_LEG1 → LONG_LEG2 (when leg1 in profit AND ema still bullish)
      LONG_LEG2 → HEDGED (when leg2 in profit, open hedge)
      HEDGED → NEUTRAL (when both legs close)

    This prevents averaging/martingale — each state requires validity.
    """

    def __init__(self):
        self.state = HedgeState.NEUTRAL
        self._entry_price_a: float = 0.0
        self._entry_price_b: float = 0.0
        self._bar_count: int = 0

    def transition(
        self,
        ema_direction: int,
        leg1_profit_pct: float,
        leg2_profit_pct: float,
        has_eligible_pair: bool,
    ) -> HedgeSignal:
        """Compute next state and signal."""
        self._bar_count += 1

        if self.state == HedgeState.NEUTRAL:
            if ema_direction > 0 and has_eligible_pair:
                self.state = HedgeState.LONG_LEG1
                return HedgeSignal(
                    state=self.state,
                    action="open_long_a",
                    size_multiplier=1.0,
                    confidence=0.7,
                )
            return HedgeSignal(self.state, "hold", 0.0, 0.0)

        elif self.state == HedgeState.LONG_LEG1:
            if leg1_profit_pct > 0.01 and ema_direction > 0:
                self.state = HedgeState.LONG_LEG2
                return HedgeSignal(
                    state=self.state,
                    action="open_long_b",
                    size_multiplier=0.5,
                    confidence=0.6,
                )
            if ema_direction < 0:
                self.state = HedgeState.NEUTRAL
                return HedgeSignal(self.state, "close_all", 0.0, 0.0)
            return HedgeSignal(self.state, "hold", 0.0, 0.0)

        elif self.state == HedgeState.LONG_LEG2:
            if leg2_profit_pct > 0.005:
                self.state = HedgeState.HEDGED
                return HedgeSignal(
                    state=self.state,
                    action="open_long_b",
                    size_multiplier=0.5,
                    confidence=0.5,
                )
            if ema_direction < 0:
                self.state = HedgeState.NEUTRAL
                return HedgeSignal(self.state, "close_all", 0.0, 0.0)
            return HedgeSignal(self.state, "hold", 0.0, 0.0)

        elif self.state == HedgeState.HEDGED:
            self.state = HedgeState.NEUTRAL
            return HedgeSignal(self.state, "close_all", 0.0, 0.0)

        return HedgeSignal(self.state, "hold", 0.0, 0.0)


# ── P24-28: Hedge-only variant (no averaging) ──


@dataclass
class HedgeOnlyConfig:
    """Configuration for hedge-only triangular variant.

    STRICTLY no averaging/martingale — those cause 61% DD per empirical evidence.
    """

    tp_pct: float = 0.03
    sl_pct: float = -0.025
    max_hold_bars: int = 20
    ema_fast: int = 10
    ema_slow: int = 30
    min_correlation: float = -0.3
    max_correlation: float = -0.05


def run_hedge_only_backtest(
    prices_a: pd.Series,
    prices_b: pd.Series,
    config: HedgeOnlyConfig,
) -> List[dict]:
    """Backtest hedge-only variant — deploy both sides, close at EMA reversal or fixed TP/SL.

    Returns:
        List of trade result dicts.
    """
    trades: List[dict] = []
    state_machine = TriangularHedgeStateMachine()
    in_trade = False
    entry_bar: int = 0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0

    min_len = min(len(prices_a), len(prices_b))
    idx = pd.RangeIndex(min_len)

    for i in range(config.ema_slow + 1, min_len - 1):
        ema_dir = ema_gate_check(prices_a.iloc[: i + 1], config.ema_fast, config.ema_slow)

        if not in_trade:
            if ema_dir > 0:
                entry_bar = i
                entry_price_a = float(prices_a.iloc[i])
                entry_price_b = float(prices_b.iloc[i])
                in_trade = True
            continue

        bars_held = i - entry_bar
        ret_a = (float(prices_a.iloc[i]) / entry_price_a - 1.0) if entry_price_a > 0 else 0.0
        ret_b = (float(prices_b.iloc[i]) / entry_price_b - 1.0) if entry_price_b > 0 else 0.0
        portfolio_ret = (ret_a + ret_b) / 2.0

        close_reason = ""

        if bars_held >= config.max_hold_bars:
            close_reason = "max_hold"
        elif ema_dir < 0:
            close_reason = "ema_reversal"
        elif portfolio_ret >= config.tp_pct:
            close_reason = "tp_hit"
        elif portfolio_ret <= config.sl_pct:
            close_reason = "sl_hit"

        if close_reason:
            trades.append(
                {
                    "entry_bar": int(entry_bar),
                    "exit_bar": i,
                    "hold_bars": bars_held,
                    "ret_a": round(ret_a, 4),
                    "ret_b": round(ret_b, 4),
                    "portfolio_ret": round(portfolio_ret, 4),
                    "close_reason": close_reason,
                }
            )
            in_trade = False

    return trades
