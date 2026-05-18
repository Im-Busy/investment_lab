"""
HFT Order Flow Alpha Factor — Golden Ratio Weighted Signal

Origin: FMZ strategies repo, JavaScript, Author: 发明者量化

Converts real-time trade data into a normalized [-1, 1] alpha factor signal
for market-making and directional trading. Positive = buyer's market,
negative = seller's market.

Methodology:
1. Collect recent N trades (default QSize=100)
2. Split at golden ratio (38.2%): rightmost portion = "recent" trades
3. Calculate buy/sell volume ratios
4. Normalize via tanh to [-1, 1]

This module operates on trade-level data (list of (side, quantity, price) tuples),
not OHLCV bars. It provides both a single-calculation function and a streaming
accumulator for real-time use.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TradeTick:
    """Single trade record."""

    side: str  # 'buy' or 'sell'
    qty: float
    price: float


@dataclass
class OrderFlowSignal:
    """Output of order flow analysis."""

    alpha: float  # Normalized [-1, 1] signal
    last_price: float
    cumulative_volume: float
    buy_ratio: float  # Recent buy vol / total sell vol
    sell_ratio: float  # Recent sell vol / total buy vol
    trade_count: int


class OrderFlowAccumulator:
    """
    Streaming accumulator for order flow alpha factor calculation.

    Accumulates trade ticks and computes the alpha factor when sufficient
    data is available. Uses golden ratio (0.382) weighted window for
    recent trade emphasis.

    Usage:
        acc = OrderFlowAccumulator(q_size=100)
        for trade in trade_stream:
            signal = acc.add_trade(trade)
            if signal:
                print(f"Alpha: {signal.alpha:.3f}")

    Attributes:
        q_size: Maximum number of trades to retain
        golden_ratio: Fraction of trades considered "recent" (default 0.382)
    """

    GOLDEN_RATIO = 0.382

    def __init__(self, q_size: int = 100):
        self.q_size = q_size
        self._trades: List[TradeTick] = []
        self._cum_vol: float = 0.0

    def add_trade(self, tick: TradeTick) -> OrderFlowSignal:
        """
        Add a trade tick and compute the order flow alpha factor.

        Args:
            tick: TradeTick with side, qty, price

        Returns:
            OrderFlowSignal with normalized alpha factor
        """
        self._trades.append(tick)
        self._cum_vol += tick.qty

        if len(self._trades) > self.q_size:
            removed = self._trades.pop(0)
            self._cum_vol -= removed.qty

        alpha, buy_ratio, sell_ratio = self._calc_alpha()

        return OrderFlowSignal(
            alpha=alpha,
            last_price=tick.price,
            cumulative_volume=self._cum_vol,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            trade_count=len(self._trades),
        )

    def _calc_alpha(self) -> Tuple[float, float, float]:
        """Calculate alpha factor from accumulated trades."""
        if len(self._trades) < self.q_size:
            return 0.0, 0.0, 0.0

        right_pos = math.ceil(len(self._trades) * self.GOLDEN_RATIO)

        tick_sell_vol = 0.0
        tick_buy_vol = 0.0
        last_buy_vol = 0.0
        last_sell_vol = 0.0

        for idx, trade in enumerate(self._trades):
            if trade.side == "buy":
                if idx >= right_pos:
                    last_buy_vol += trade.qty
                tick_buy_vol += trade.qty
            else:
                if idx >= right_pos:
                    last_sell_vol += trade.qty
                tick_sell_vol += trade.qty

        if tick_sell_vol == 0 or tick_buy_vol == 0:
            return 0.0, 0.0, 0.0

        positive_ratio = last_buy_vol / tick_sell_vol if tick_sell_vol > 0 else 0.0
        negative_ratio = last_sell_vol / tick_buy_vol if tick_buy_vol > 0 else 0.0

        if positive_ratio > negative_ratio:
            alpha = math.tanh(positive_ratio)
        else:
            alpha = -math.tanh(negative_ratio)

        return round(alpha, 3), round(positive_ratio, 4), round(negative_ratio, 4)

    def reset(self) -> None:
        """Clear accumulated trades."""
        self._trades.clear()
        self._cum_vol = 0.0


def calc_order_flow_alpha(trades: List[TradeTick], q_size: int = 100) -> OrderFlowSignal:
    """
    One-shot calculation of order flow alpha factor from a list of trades.

    Args:
        trades: List of TradeTick objects
        q_size: Maximum trades to use (takes last N)

    Returns:
        OrderFlowSignal with normalized alpha factor
    """
    acc = OrderFlowAccumulator(q_size=q_size)
    result = None
    for trade in trades:
        result = acc.add_trade(trade)
    return (
        result
        if result
        else OrderFlowSignal(
            alpha=0.0,
            last_price=0.0,
            cumulative_volume=0.0,
            buy_ratio=0.0,
            sell_ratio=0.0,
            trade_count=0,
        )
    )


def trades_from_ohlcv(df, i: int, lookback: int = 20) -> List[TradeTick]:
    """
    Synthesize approximate trade direction from OHLCV data.

    This is a rough approximation for backtesting when real trade-level
    data is not available. Uses candle direction and wick proportions
    to estimate buy/sell pressure.

    Args:
        df: DataFrame with OHLCV columns
        i: Current bar index
        lookback: Number of bars to synthesize from

    Returns:
        List of synthesized TradeTick objects
    """
    trades = []
    start = max(0, i - lookback + 1)
    for j in range(start, i + 1):
        high = float(df["High"].iloc[j])
        low = float(df["Low"].iloc[j])
        open_p = float(df["Open"].iloc[j])
        close_p = float(df["Close"].iloc[j])
        volume = float(df["Volume"].iloc[j]) if "Volume" in df.columns else 100.0

        if close_p > open_p:
            # Bullish candle: estimate buy volume at upper portion
            buy_qty = volume * 0.6
            sell_qty = volume * 0.4
            trades.append(TradeTick(side="buy", qty=buy_qty, price=close_p))
            trades.append(TradeTick(side="sell", qty=sell_qty, price=close_p))
        else:
            # Bearish candle: estimate sell volume at lower portion
            buy_qty = volume * 0.4
            sell_qty = volume * 0.6
            trades.append(TradeTick(side="buy", qty=buy_qty, price=close_p))
            trades.append(TradeTick(side="sell", qty=sell_qty, price=close_p))

    return trades
