"""P24-22: ETF portfolio rotation strategy.

Train for 45 days → select top-10 by predicted return → trade for 45 days → repeat.
Paper reports 102% vs 33% for single-symbol baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Pre-configured ETF universe
ETF_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLU",
    "XLB",
    "XLY",
    "XLRE",
    "XLC",
    "GLD",
    "SLV",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "TIP",
    "EEM",
    "EFA",
    "EWJ",
    "VNQ",
    "USO",
    "DBC",
]


@dataclass
class RotationWindow:
    """A single rotation window with training + execution periods."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp
    selected_tickers: List[str] = field(default_factory=list)
    pred_returns: Dict[str, float] = field(default_factory=dict)
    actual_returns: Dict[str, float] = field(default_factory=dict)


@dataclass
class RotationResult:
    """ETF rotation strategy results."""

    windows: List[RotationWindow] = field(default_factory=list)
    total_return: float = 0.0
    num_windows: int = 0
    avg_win_rate: float = 0.0


def compute_predicted_returns(
    data: Dict[str, pd.DataFrame],
    lookback: int = 45,
) -> Dict[str, float]:
    """Compute momentum-based predicted return for next period.

    Uses a simple multi-factor adaptive blend: 70% momentum + 30% mean reversion.
    """
    preds: Dict[str, float] = {}
    for ticker, df in data.items():
        if len(df) < lookback:
            continue
        recent = df["Close"].iloc[-lookback:].values
        momentum = (recent[-1] / recent[0] - 1.0) if recent[0] > 0 else 0.0
        mean_rev = -(recent[-1] / recent.mean() - 1.0) if recent.mean() > 0 else 0.0
        preds[ticker] = 0.7 * momentum + 0.3 * mean_rev
    return preds


def select_top_n(
    predicted_returns: Dict[str, float],
    n: int = 10,
) -> List[str]:
    """Select top-N tickers by predicted return."""
    sorted_tickers = sorted(predicted_returns, key=lambda t: predicted_returns[t], reverse=True)
    return sorted_tickers[:n]


def compute_rotation_return(
    ticker: str,
    df: pd.DataFrame,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
) -> float:
    """Compute buy-and-hold return over trade window."""
    mask = (df.index >= trade_start) & (df.index <= trade_end)
    window = df.loc[mask, "Close"]
    if len(window) < 2:
        return 0.0
    return float(window.iloc[-1] / window.iloc[0] - 1.0)


def run_etf_rotation(
    data: Dict[str, pd.DataFrame],
    train_days: int = 45,
    trade_days: int = 45,
    top_n: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> RotationResult:
    """Run ETF rotation backtest.

    Args:
        data: Dict[ticker, DataFrame] with at least Close column.
        train_days: Number of trading days for training window.
        trade_days: Number of trading days for execution window.
        top_n: Number of tickers to select each window.
        start_date: Override start date (YYYY-MM-DD).
        end_date: Override end date (YYYY-MM-DD).

    Returns:
        RotationResult with per-window details and total return.
    """
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in data.values()])))
    if start_date:
        all_dates = all_dates[all_dates >= pd.Timestamp(start_date)]
    if end_date:
        all_dates = all_dates[all_dates <= pd.Timestamp(end_date)]

    if len(all_dates) < train_days + trade_days:
        logger.warning("Not enough data for one rotation window")
        return RotationResult()

    windows: List[RotationWindow] = []
    total_cumulative = 1.0
    window_id = 0
    pos = train_days

    while pos + trade_days < len(all_dates):
        train_start = all_dates[pos - train_days]
        train_end = all_dates[pos]
        trade_start = all_dates[pos]
        trade_end = all_dates[min(pos + trade_days, len(all_dates) - 1)]

        window_data = {
            ticker: df.loc[(df.index >= train_start) & (df.index <= train_end)]
            for ticker, df in data.items()
            if ticker in df.index
        }
        window_data = {k: v for k, v in window_data.items() if len(v) >= 20}

        preds = compute_predicted_returns(window_data, lookback=train_days)
        selected = select_top_n(preds, n=top_n)

        actual_returns: Dict[str, float] = {}
        for ticker in selected:
            if ticker in data:
                actual_returns[ticker] = compute_rotation_return(
                    ticker, data[ticker], trade_start, trade_end
                )

        portfolio_return = np.mean(list(actual_returns.values())) if actual_returns else 0.0

        windows.append(
            RotationWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                trade_start=trade_start,
                trade_end=trade_end,
                selected_tickers=selected,
                pred_returns=preds,
                actual_returns=actual_returns,
            )
        )

        total_cumulative *= 1.0 + portfolio_return
        pos += trade_days
        window_id += 1

    win_count = sum(1 for w in windows if any(r > 0 for r in w.actual_returns.values()))

    return RotationResult(
        windows=windows,
        total_return=total_cumulative - 1.0,
        num_windows=len(windows),
        avg_win_rate=win_count / max(len(windows), 1) if windows else 0.0,
    )
