"""P24-14: Volatility no-trade switch with event calendar.

Halts trading when:
  - 5-day rolling vol exceeds 1.5% (turbulent regimes)
  - FOMC days (Fed announcements inject binary risk)
  - NFP days (employment data causes gap risk)

Simple but high-impact structural risk filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pre-built FOMC and NFP calendar for 2025-2026
FOMC_DATES: Set[date] = {
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 11, 5),
    date(2025, 12, 17),
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 20),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
}

NFP_DATES: Set[date] = {
    date(2025, 1, 10),
    date(2025, 2, 7),
    date(2025, 3, 7),
    date(2025, 4, 4),
    date(2025, 5, 2),
    date(2025, 6, 6),
    date(2025, 7, 3),
    date(2025, 8, 1),
    date(2025, 9, 5),
    date(2025, 10, 3),
    date(2025, 11, 7),
    date(2025, 12, 5),
    date(2026, 1, 9),
    date(2026, 2, 6),
    date(2026, 3, 6),
    date(2026, 4, 3),
    date(2026, 5, 1),
    date(2026, 6, 5),
}

# FOMC dates where market halts or limits trading recommended
FOMC_BLACKOUT_DATES: Set[date] = {
    date(2025, 1, 28),
    date(2025, 3, 18),
    date(2025, 5, 6),
    date(2025, 6, 17),
    date(2025, 7, 29),
    date(2025, 9, 16),
    date(2025, 11, 4),
    date(2025, 12, 16),
    date(2026, 1, 27),
    date(2026, 3, 17),
    date(2026, 5, 19),
    date(2026, 6, 16),
    date(2026, 7, 28),
    date(2026, 9, 15),
    date(2026, 11, 3),
    date(2026, 12, 15),
}


@dataclass
class NoTradeDecision:
    """Result of volatility + event calendar gate check."""

    timestamp: pd.Timestamp
    vol_5d: float
    vol_5d_exceeds: bool
    is_fomc_day: bool
    is_nfp_day: bool
    is_fomc_blackout: bool
    no_trade: bool
    reason: str


class VolNoTradeSwitch:
    """Volatility + event-based no-trade gate.

    Args:
        vol_threshold: 5-day rolling vol threshold (default 0.015 = 1.5%).
        fomc_gate: Halt on FOMC day? (default True)
        nfp_gate: Halt on NFP day? (default True)
        fomc_blackout: Halt on FOMC blackout day? (default True — no new positions)
    """

    def __init__(
        self,
        vol_threshold: float = 0.015,
        fomc_gate: bool = True,
        nfp_gate: bool = True,
        fomc_blackout: bool = True,
    ):
        self.vol_threshold = vol_threshold
        self.fomc_gate = fomc_gate
        self.nfp_gate = nfp_gate
        self.fomc_blackout = fomc_blackout
        self._vol_5d: Optional[np.ndarray] = None

    def precompute(self, data: pd.DataFrame, window: int = 5) -> np.ndarray:
        """Precompute rolling 5-day log-return volatility."""
        log_ret = np.log(data["Close"] / data["Close"].shift(1))
        self._vol_5d = log_ret.rolling(window, min_periods=3).std().fillna(0.0).values
        return self._vol_5d

    def check(self, idx: int, ts: Optional[pd.Timestamp] = None) -> NoTradeDecision:
        """Check no-trade conditions at bar index.

        Returns:
            NoTradeDecision: no_trade=True means skip this bar.
        """
        reasons: List[str] = []
        vol_5d = 0.0
        vol_exceeds = False
        is_fomc = False
        is_nfp = False
        is_blackout = False

        if self._vol_5d is not None and idx < len(self._vol_5d):
            vol_5d = float(self._vol_5d[idx])
            vol_exceeds = vol_5d > self.vol_threshold
            if vol_exceeds:
                reasons.append(f"vol_5d={vol_5d:.4f}>{self.vol_threshold}")

        if ts is not None:
            bar_date = ts.date() if hasattr(ts, "date") else ts
            if self.fomc_gate and isinstance(bar_date, date) and bar_date in FOMC_DATES:
                is_fomc = True
                reasons.append("FOMC_day")
            if self.nfp_gate and isinstance(bar_date, date) and bar_date in NFP_DATES:
                is_nfp = True
                reasons.append("NFP_day")
            if (
                self.fomc_blackout
                and isinstance(bar_date, date)
                and bar_date in FOMC_BLACKOUT_DATES
            ):
                is_blackout = True
                reasons.append("FOMC_blackout")

        no_trade = vol_exceeds or is_fomc or is_nfp or is_blackout
        reason = "; ".join(reasons) if reasons else "clear"

        return NoTradeDecision(
            timestamp=ts or pd.Timestamp.now(),
            vol_5d=vol_5d,
            vol_5d_exceeds=vol_exceeds,
            is_fomc_day=is_fomc,
            is_nfp_day=is_nfp,
            is_fomc_blackout=is_blackout,
            no_trade=no_trade,
            reason=reason,
        )

    def compute_vol_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series of vol regime labels (low/normal/high) per bar."""
        if self._vol_5d is None:
            self.precompute(data)
        labels = []
        for v in self._vol_5d:
            if v > 0.02:
                labels.append("high")
            elif v > self.vol_threshold:
                labels.append("elevated")
            else:
                labels.append("normal")
        return pd.Series(labels, index=data.index[: len(labels)])
