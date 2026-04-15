"""
Pairs Scanner - Cointegration-based pair finder

Scans a universe of symbols for cointegrated pairs using the
Engle-Granger two-step method.

Usage:
    from src.strategies.pairs_scanner import PairsScanner

    scanner = PairsScanner()
    pairs = scanner.scan(symbols, df_dict)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass
class CointegratedPair:
    """Represents a cointegrated pair with statistics."""

    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    p_value: float
    adf_statistic: float
    half_life: float
    spread_mean: float
    spread_std: float
    correlation: float


class PairsScanner:
    """
    Scan for cointegrated pairs in a universe of symbols.

    Uses the Engle-Granger two-step cointegration test:
    1. Regress price_A on price_B to get hedge ratio
    2. Test residuals for stationarity using ADF test

    Parameters:
        significance_level: P-value threshold for cointegration (default: 0.05)
        min_half_life: Minimum acceptable half-life in days (default: 5)
        max_half_life: Maximum acceptable half-life in days (default: 60)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_half_life: float = 5,
        max_half_life: float = 60,
    ):
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def scan(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        lookback: int = 252,
    ) -> List[CointegratedPair]:
        """
        Scan for cointegrated pairs.

        Args:
            symbols: List of symbol tickers
            price_data: Dict mapping symbol -> OHLCV DataFrame (must have 'Close')
            lookback: Number of days to use for cointegration test

        Returns:
            List of CointegratedPair objects, sorted by p-value (most significant first)
        """
        # Extract close prices
        close_prices = {}
        for sym in symbols:
            if sym in price_data and len(price_data[sym]) >= lookback:
                df = price_data[sym].tail(lookback)
                close_prices[sym] = df["Close"].values

        if len(close_prices) < 2:
            return []

        pairs = []
        symbol_list = list(close_prices.keys())

        for i in range(len(symbol_list)):
            for j in range(i + 1, len(symbol_list)):
                sym_a = symbol_list[i]
                sym_b = symbol_list[j]

                price_a = close_prices[sym_a]
                price_b = close_prices[sym_b]

                result = self._test_pair(sym_a, sym_b, price_a, price_b)
                if result and result.p_value < self.significance_level:
                    if self.min_half_life <= result.half_life <= self.max_half_life:
                        pairs.append(result)

        # Sort by p-value (most significant first)
        pairs.sort(key=lambda p: p.p_value)
        return pairs

    def _test_pair(
        self,
        sym_a: str,
        sym_b: str,
        price_a: np.ndarray,
        price_b: np.ndarray,
    ) -> Optional[CointegratedPair]:
        """Test if two price series are cointegrated."""
        try:
            # Step 1: Cointegration test (Engle-Granger)
            score, p_value, _ = coint(price_a, price_b)

            if p_value >= self.significance_level:
                return None

            # Step 2: Calculate hedge ratio via OLS
            hedge_result = np.polyfit(price_b, price_a, 1)
            hedge_ratio = hedge_result[0]

            # Step 3: Calculate spread
            spread = price_a - hedge_ratio * price_b

            # Step 4: Calculate half-life of mean reversion
            half_life = self._calculate_half_life(spread)

            # Step 5: Calculate spread statistics
            spread_mean = float(np.mean(spread))
            spread_std = float(np.std(spread))

            # Step 6: Calculate correlation
            correlation = float(np.corrcoef(price_a, price_b)[0, 1])

            return CointegratedPair(
                symbol_a=sym_a,
                symbol_b=sym_b,
                hedge_ratio=hedge_ratio,
                p_value=float(p_value),
                adf_statistic=float(score),
                half_life=half_life,
                spread_mean=spread_mean,
                spread_std=spread_std,
                correlation=correlation,
            )

        except Exception:
            return None

    @staticmethod
    def _calculate_half_life(spread: np.ndarray) -> float:
        """
        Calculate the half-life of mean reversion.

        Uses the Ornstein-Uhlenbeck process:
        dS = θ(μ - S)dt + σdW

        Half-life = ln(2) / θ
        """
        spread_lag = np.roll(spread, 1)
        spread_lag[0] = 0
        spread_change = spread - spread_lag
        spread_lag[0] = spread[0]

        # Regression: ΔS = θ * S_{t-1} + ε
        result = np.polyfit(spread_lag[1:], spread_change[1:], 1)
        theta = result[0]

        if theta >= 0:
            return float("inf")

        half_life = -np.log(2) / theta
        return float(half_life)
