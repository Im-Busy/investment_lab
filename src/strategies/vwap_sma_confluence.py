"""
VWAP + SMA Confluence Module

Combines VWAP (institutional benchmark) with SMA 50/200 (self-fulfilling prophecy).

Key Principles:
1. VWAP is the institutional benchmark (the "king")
2. SMA 50/200 works because everyone watches it
3. Price position relative to both = confluence score
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators.technical import sma
from ..indicators.vwap import compute_vwap


class VWAPPosition(Enum):
    """Price position relative to VWAP."""

    ABOVE = "above"
    BELOW = "below"
    AT = "at"


class SMAPosition(Enum):
    """Price position relative to SMA array."""

    ABOVE_BOTH = "above_both"
    BETWEEN = "between"
    BELOW_BOTH = "below_both"


@dataclass
class VWAPConfluence:
    """
    VWAP + SMA confluence state.

    Attributes:
        timestamp: Timestamp
        price: Current price
        vwap: Current VWAP value
        sma_50: 50-period SMA
        sma_200: 200-period SMA
        vwap_position: Price vs VWAP position
        sma_position: Price vs SMA position
        vwap_distance_pct: Distance from VWAP as percentage
        confluence_score: Overall confluence score (0.0 to 1.0)
        bias: Directional bias ('bullish', 'bearish', 'neutral')
    """

    timestamp: pd.Timestamp
    price: float
    vwap: float
    sma_50: float
    sma_200: float
    vwap_position: VWAPPosition
    sma_position: SMAPosition
    vwap_distance_pct: float
    confluence_score: float
    bias: str


@dataclass
class VWAPConfig:
    """
    VWAP confluence configuration.

    Attributes:
        vwap_deviation_pct: VWAP deviation band percentage (default 0.5%)
        sma_short: Short SMA period (default 50)
        sma_long: Long SMA period (default 200)
        at_vwap_threshold: Threshold for "at VWAP" (default 0.1%)
    """

    vwap_deviation_pct: float = 0.5
    sma_short: int = 50
    sma_long: int = 200
    at_vwap_threshold: float = 0.1


class VWAPConfluenceAnalyzer:
    """
    VWAP + SMA Confluence Analyzer

    Analyzes price position relative to:
    1. VWAP (institutional benchmark)
    2. SMA 50 (medium-term trend)
    3. SMA 200 (long-term trend)

    Confluence Scoring:
    - Price > VWAP > SMA50 > SMA200 = Strong bullish (1.0)
    - Price < VWAP < SMA50 < SMA200 = Strong bearish (0.0)
    - Mixed positions = Neutral/transition (0.5)

    Example:
        >>> config = VWAPConfluenceConfig()
        >>> analyzer = VWAPConfluenceAnalyzer(config)
        >>> confluence = analyzer.analyze(df)
    """

    def __init__(self, config: Optional[VWAPConfig] = None):
        """
        Initialize VWAP Confluence Analyzer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or VWAPConfig()

    def analyze(self, df: pd.DataFrame) -> List[VWAPConfluence]:
        """
        Analyze VWAP + SMA confluence for all bars.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of VWAPConfluence objects
        """
        self._validate_dataframe(df)

        # Calculate indicators
        vwap_df = compute_vwap(df, deviation_pct=self.config.vwap_deviation_pct)
        sma_50 = sma(df, period=self.config.sma_short)
        sma_200 = sma(df, period=self.config.sma_long)

        confluences = []
        for i in range(len(df)):
            if i < self.config.sma_long:
                continue

            confluence = self._calculate_confluence(
                df=df,
                vwap_df=vwap_df,
                sma_50=sma_50,
                sma_200=sma_200,
                idx=i,
            )
            confluences.append(confluence)

        return confluences

    def get_current_confluence(self, df: pd.DataFrame) -> VWAPConfluence:
        """
        Get current confluence state (last bar).

        Args:
            df: OHLCV DataFrame

        Returns:
            VWAPConfluence for current bar
        """
        self._validate_dataframe(df)

        vwap_df = compute_vwap(df, deviation_pct=self.config.vwap_deviation_pct)
        sma_50 = sma(df, period=self.config.sma_short)
        sma_200 = sma(df, period=self.config.sma_long)

        return self._calculate_confluence(
            df=df,
            vwap_df=vwap_df,
            sma_50=sma_50,
            sma_200=sma_200,
            idx=len(df) - 1,
        )

    def _calculate_confluence(
        self,
        df: pd.DataFrame,
        vwap_df: pd.DataFrame,
        sma_50: pd.Series,
        sma_200: pd.Series,
        idx: int,
    ) -> VWAPConfluence:
        """Calculate confluence at a specific bar."""
        price = df["Close"].iloc[idx]
        vwap = vwap_df["vwap"].iloc[idx]
        sma50_val = sma_50.iloc[idx] if idx < len(sma_50) else 0
        sma200_val = sma_200.iloc[idx] if idx < len(sma_200) else 0

        # Determine VWAP position
        vwap_dist_pct = (price - vwap) / vwap if vwap > 0 else 0

        if abs(vwap_dist_pct) < self.config.at_vwap_threshold / 100:
            vwap_position = VWAPPosition.AT
        elif price > vwap:
            vwap_position = VWAPPosition.ABOVE
        else:
            vwap_position = VWAPPosition.BELOW

        # Determine SMA position
        if price > sma50_val and price > sma200_val and sma50_val > sma200_val:
            sma_position = SMAPosition.ABOVE_BOTH
        elif price < sma50_val and price < sma200_val and sma50_val < sma200_val:
            sma_position = SMAPosition.BELOW_BOTH
        else:
            sma_position = SMAPosition.BETWEEN

        # Calculate confluence score
        confluence_score = self._calculate_score(
            vwap_position=vwap_position,
            sma_position=sma_position,
            vwap_distance_pct=vwap_dist_pct * 100,
        )

        # Determine bias
        bias = self._determine_bias(confluence_score, vwap_position, sma_position)

        timestamp = df.index[idx]

        return VWAPConfluence(
            timestamp=timestamp,
            price=float(price),
            vwap=float(vwap),
            sma_50=float(sma50_val),
            sma_200=float(sma200_val),
            vwap_position=vwap_position,
            sma_position=sma_position,
            vwap_distance_pct=vwap_dist_pct * 100,
            confluence_score=confluence_score,
            bias=bias,
        )

    def _calculate_score(
        self,
        vwap_position: VWAPPosition,
        sma_position: SMAPosition,
        vwap_distance_pct: float,
    ) -> float:
        """
        Calculate confluence score (0.0 to 1.0).

        Scoring matrix (bullish perspective):
        - Price > VWAP: +0.3
        - Price > SMA50: +0.2
        - Price > SMA200: +0.2
        - SMA50 > SMA200: +0.3
        - At VWAP (within threshold): +0.1 (potential bounce)

        Args:
            vwap_position: Price vs VWAP
            sma_position: Price vs SMA
            vwap_distance_pct: Distance from VWAP

        Returns:
            Confluence score (0.0 to 1.0)
        """
        score = 0.5  # Neutral starting point

        # VWAP position scoring
        if vwap_position == VWAPPosition.ABOVE:
            score += 0.3
        elif vwap_position == VWAPPosition.BELOW:
            score -= 0.3
        elif vwap_position == VWAPPosition.AT:
            score += 0.1  # Potential bounce zone

        # SMA position scoring
        if sma_position == SMAPosition.ABOVE_BOTH:
            score += 0.4
        elif sma_position == SMAPosition.BELOW_BOTH:
            score -= 0.4
        else:  # BETWEEN
            score += 0.0  # Neutral

        # Distance bonus (further from VWAP = stronger trend)
        if abs(vwap_distance_pct) > 1.0:
            if vwap_distance_pct > 0:
                score += 0.1
            else:
                score -= 0.1

        # Normalize to 0.0-1.0
        return np.clip(score, 0.0, 1.0)

    def _determine_bias(
        self,
        confluence_score: float,
        vwap_position: VWAPPosition,
        sma_position: SMAPosition,
    ) -> str:
        """
        Determine directional bias.

        Args:
            confluence_score: Overall confluence score
            vwap_position: Price vs VWAP
            sma_position: Price vs SMA

        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if confluence_score >= 0.7:
            return "bullish"
        elif confluence_score <= 0.3:
            return "bearish"
        else:
            return "neutral"

    def get_entry_signals(
        self,
        df: pd.DataFrame,
        min_confluence: float = 0.7,
        require_vwap_reclaim: bool = True,
    ) -> List[Dict]:
        """
        Generate entry signals based on VWAP confluence.

        Entry Rules:
        - Bullish: Confluence >= min_confluence + VWAP reclaim
        - Bearish: Confluence <= (1 - min_confluence) + VWAP loss

        Args:
            df: OHLCV DataFrame
            min_confluence: Minimum confluence for signal
            require_vwap_reclaim: Require VWAP cross for entry

        Returns:
            List of signal dictionaries
        """
        confluences = self.analyze(df)
        signals = []

        for i, confluence in enumerate(confluences):
            if i == 0:
                continue

            prev_confluence = confluences[i - 1]

            # Bullish entry: reclaim VWAP with high confluence
            if confluence.bias == "bullish" and confluence.confluence_score >= min_confluence:
                if not require_vwap_reclaim or (
                    confluence.vwap_position == VWAPPosition.ABOVE
                    and prev_confluence.vwap_position != VWAPPosition.ABOVE
                ):
                    signals.append(
                        {
                            "timestamp": confluence.timestamp,
                            "direction": "long",
                            "price": confluence.price,
                            "confluence_score": confluence.confluence_score,
                            "vwap": confluence.vwap,
                            "sma_50": confluence.sma_50,
                            "sma_200": confluence.sma_200,
                        }
                    )

            # Bearish entry: lose VWAP with low confluence
            elif confluence.bias == "bearish" and confluence.confluence_score <= (
                1 - min_confluence
            ):
                if not require_vwap_reclaim or (
                    confluence.vwap_position == VWAPPosition.BELOW
                    and prev_confluence.vwap_position != VWAPPosition.BELOW
                ):
                    signals.append(
                        {
                            "timestamp": confluence.timestamp,
                            "direction": "short",
                            "price": confluence.price,
                            "confluence_score": confluence.confluence_score,
                            "vwap": confluence.vwap,
                            "sma_50": confluence.sma_50,
                            "sma_200": confluence.sma_200,
                        }
                    )

        return signals

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

    def get_vwap_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get current VWAP levels for support/resistance.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with VWAP levels
        """
        vwap_df = compute_vwap(df, deviation_pct=self.config.vwap_deviation_pct)

        current_vwap = vwap_df["vwap"].iloc[-1]
        vwap_upper = vwap_df["vwap_upper"].iloc[-1]
        vwap_lower = vwap_df["vwap_lower"].iloc[-1]
        current_price = df["Close"].iloc[-1]

        return {
            "vwap": float(current_vwap),
            "vwap_upper": float(vwap_upper),
            "vwap_lower": float(vwap_lower),
            "price": float(current_price),
            "price_vs_vwap_pct": float((current_price - current_vwap) / current_vwap * 100),
            "deviation_pct": float(self.config.vwap_deviation_pct),
        }
