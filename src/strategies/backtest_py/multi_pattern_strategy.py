"""
Multi-Pattern Strategy for backtesting.py

Wrapper for running multi-pattern confluence strategy in backtesting.py framework.
Integrates all 20 pattern detectors with confluence scoring.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from backtesting import Strategy

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import pattern detectors
from src.indicators.regime import MarketRegimeDetector
from src.patterns.basic.floor_pivot import FloorPivotBreakout
from src.patterns.basic.matching_lows import MatchingLows
from src.patterns.basic.msl import MarketStructureLow
from src.patterns.basic.n_bar_decline import NBarDecline
from src.patterns.basic.nr7id import NR7ID
from src.patterns.breakout.donchian import DonchianChannelBreakout
from src.patterns.classic.dead_cat_bounce import DeadCatBounce
from src.patterns.classic.double_bottom import DoubleBottom
from src.patterns.classic.double_top import DoubleTop
from src.patterns.classic.trader_vic_2b import TraderVic2B
from src.patterns.classic.triple_top import TripleTop
from src.patterns.complex.cup_handle import CupAndHandle
from src.patterns.complex.head_shoulders import HeadAndShoulders
from src.patterns.complex.parabolic_arc import ParabolicArc
from src.patterns.complex.spike_ledge import SpikeAndLedge
from src.patterns.complex.three_hills import ThreeHillsMountain
from src.patterns.harmonic.abc import ABCPattern
from src.patterns.harmonic.bollinger import BollingerBands
from src.patterns.harmonic.gartley import GartleyPattern
from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle

# Import confluence and regime detection
from src.strategies.confluence import ConfluenceScorer


@dataclass
class StrategyConfig:
    """Configuration for Multi-Pattern Strategy."""

    min_confidence: float = 0.60
    min_confluence_count: int = 2
    risk_per_trade: float = 0.02
    max_open_positions: int = 5
    use_regime_filter: bool = True
    use_take_profit_1: bool = True
    use_take_profit_2: bool = False
    use_take_profit_3: bool = False
    stop_loss_atr_mult: float = 2.0
    take_profit_1_ratio: float = 1.0
    take_profit_2_ratio: float = 1.5
    take_profit_3_ratio: float = 2.0


class MultiPatternStrategy(Strategy):
    """
    Multi-Pattern Confluence Strategy for backtesting.py

    Uses confluence of multiple pattern detections to generate trading signals.
    Supports 20 patterns across 4 categories: basic, harmonic, complex, classic.

    Parameters:
        min_confidence: Minimum confidence threshold for signals
        min_confluence_count: Minimum patterns that must agree
        risk_per_trade: Risk per trade as fraction of equity
        max_open_positions: Maximum concurrent positions
        use_regime_filter: Whether to filter by market regime
    """

    # Strategy parameters (can be optimized)
    min_confidence = 0.60
    min_confluence_count = 2
    risk_per_trade = 0.02
    max_open_positions = 5
    use_regime_filter = True

    # Take profit settings
    use_take_profit_1 = True
    use_take_profit_2 = False
    use_take_profit_3 = False

    def init(self):
        """Initialize indicators and pattern detectors."""
        # Initialize all pattern detectors
        self.patterns = self._init_patterns()

        # Initialize confluence scorer
        self.confluence_scorer = ConfluenceScorer(
            min_confidence=self.min_confidence, regime_adaptation=self.use_regime_filter
        )

        # Initialize regime detector if enabled
        if self.use_regime_filter:
            self.regime_detector = MarketRegimeDetector()

        # Track signals for analysis
        self.signals = []
        self.signal_count = 0

        # Pre-calculate indicators for patterns using backtesting.py's I() decorator
        # This ensures indicators are computed efficiently
        self._init_indicators()

        # Store the full DataFrame for pattern detection
        self._df = None

    def _init_patterns(self) -> List:
        """Initialize all pattern detectors."""
        patterns = []

        # Basic patterns
        patterns.extend(
            [MarketStructureLow(), MatchingLows(), NR7ID(), NBarDecline(), FloorPivotBreakout()]
        )

        # Harmonic patterns
        patterns.extend([GartleyPattern(), ABCPattern(), SymmetricTriangle(), BollingerBands()])

        # Breakout patterns
        patterns.extend([DonchianChannelBreakout()])

        # Complex patterns
        patterns.extend(
            [
                CupAndHandle(),
                HeadAndShoulders(),
                SpikeAndLedge(),
                ThreeHillsMountain(),
                ParabolicArc(),
            ]
        )

        # Classic patterns
        patterns.extend([DoubleTop(), DoubleBottom(), TraderVic2B(), TripleTop(), DeadCatBounce()])

        return patterns

    def _init_indicators(self):
        """Initialize indicators needed for patterns."""
        # Get the data as DataFrame
        # backtesting.py provides data as arrays, we need to convert
        pass

    def _get_dataframe(self) -> pd.DataFrame:
        """Convert backtesting.py data to DataFrame."""
        # Create DataFrame from backtesting.py data
        df = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
                "Volume": self.data.Volume if hasattr(self.data, "Volume") else 0,
            }
        )
        df.index = self.data.index
        return df

    def _get_atr(self, period: int = 14) -> float:
        """
        Calculate Average True Range for current bar.

        Args:
            period: ATR calculation period

        Returns:
            Current ATR value
        """
        df = self._get_dataframe()
        if len(df) < period + 1:
            return df["Close"].iloc[-1] * 0.02  # Default 2% if not enough data

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        true_range = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
        ).max(axis=1)

        atr = true_range.rolling(window=period).mean().iloc[-1]
        return float(atr) if not pd.isna(atr) else df["Close"].iloc[-1] * 0.02

    def next(self):
        """Execute trading logic for current bar."""
        # Check position limit
        if len(self.trades) >= self.max_open_positions:
            return

        # Get current DataFrame
        df = self._get_dataframe()
        current_idx = len(df) - 1

        # Skip if not enough bars
        min_bars = max(p.min_bars_required for p in self.patterns)
        if current_idx < min_bars:
            return

        # Detect patterns at current bar
        signals = []
        for pattern in self.patterns:
            try:
                result = pattern.detect(df, current_idx)
                if result.detected and result.signal:
                    signals.append(
                        {
                            "pattern_name": result.pattern_name,
                            "direction": result.signal.direction.value,
                            "entry_price": result.signal.entry_price,
                            "stop_loss": result.signal.stop_loss,
                            "take_profit_1": result.signal.take_profit_1,
                            "take_profit_2": result.signal.take_profit_2,
                            "take_profit_3": result.signal.take_profit_3,
                            "confidence": result.signal.confidence,
                            "pattern_type": result.pattern_type.value,
                        }
                    )
            except Exception:
                # Skip pattern if detection fails
                continue

        # Check if we have enough confluence
        if len(signals) < self.min_confluence_count:
            return

        # Group signals by direction
        long_signals = [s for s in signals if s["direction"] == "Long"]
        short_signals = [s for s in signals if s["direction"] == "Short"]

        # Determine direction with most signals
        if (
            len(long_signals) >= len(short_signals)
            and len(long_signals) >= self.min_confluence_count
        ):
            active_signals = long_signals
            direction = "LONG"
        elif len(short_signals) >= self.min_confluence_count:
            active_signals = short_signals
            direction = "SHORT"
        else:
            return

        # Calculate confluence score
        try:
            # Use simple confidence average for now
            avg_confidence = sum(s["confidence"] for s in active_signals) / len(active_signals)

            if avg_confidence < self.min_confidence:
                return

            # Create a simple confluence-like object
            from dataclasses import dataclass

            @dataclass
            class SimpleConfluence:
                score: float
                direction: str
                pattern_count: int

            confluence = SimpleConfluence(
                score=avg_confidence, direction=direction, pattern_count=len(active_signals)
            )

        except Exception:
            # Use simple average if confluence scoring fails
            confluence = None

        # Calculate entry, stop, and targets
        entry_price = self.data.Close[-1]

        # Use weighted average of signal levels
        total_confidence = sum(s["confidence"] for s in active_signals)
        if total_confidence > 0:
            stop_loss = (
                sum(s["stop_loss"] * s["confidence"] for s in active_signals) / total_confidence
            )
        else:
            stop_loss = active_signals[0]["stop_loss"]

        # Calculate position size based on risk
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)

        if stop_distance <= 0:
            return

        # Calculate position size as fraction of equity
        # position_size = (risk_amount / stop_distance) / entry_price
        # This gives us the fraction of equity to risk
        position_size_frac = (risk_amount / stop_distance) / entry_price

        # Limit position size to max 20% of equity
        position_size_frac = min(position_size_frac, 0.20)

        # Ensure position size is valid (between 0 and 1)
        if position_size_frac <= 0 or position_size_frac > 1:
            return

        # Execute trade
        if direction == "LONG":
            # Check if we already have a long position
            if any(t.is_long for t in self.trades):
                return

            self.buy(
                size=position_size_frac,
                sl=stop_loss,
                # Note: backtesting.py only supports single TP
                # For multiple TPs, we'd need custom exit logic
            )

        else:  # SHORT
            # Check if we already have a short position
            if any(t.is_short for t in self.trades):
                return

            self.sell(size=position_size_frac, sl=stop_loss)

        # Record signal for analysis
        self.signal_count += 1
        signal_log = {
            "bar": current_idx,
            "timestamp": df.index[-1] if hasattr(df.index[-1], "isoformat") else str(df.index[-1]),
            "direction": direction,
            "patterns": [s["pattern_name"] for s in active_signals],
            "pattern_count": len(active_signals),
            "confidence": confluence.score if confluence else 0.5,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "atr": self._get_atr(),
            "risk_amount": risk_amount,
            "position_size_frac": position_size_frac,
        }
        self.signals.append(signal_log)
        logger.info(
            f"Signal #{self.signal_count}: {direction} @ {entry_price:.2f} | "
            f"Patterns: {len(active_signals)} | Confidence: {signal_log['confidence']:.2f}"
        )


class MultiPatternStrategySimple(Strategy):
    """
    Simplified Multi-Pattern Strategy for backtesting.py

    A simpler version that uses fewer patterns and more straightforward logic.
    Better for optimization and faster backtesting.
    """

    # Strategy parameters
    min_confidence = 0.55
    risk_per_trade = 0.02
    max_open_positions = 3

    def init(self):
        """Initialize with just the most reliable patterns."""
        # Use a subset of patterns for faster backtesting
        self.patterns = [
            DoubleTop(),
            DoubleBottom(),
            HeadAndShoulders(),
            MarketStructureLow(),
            NR7ID(),
        ]

        # Simple confluence counter
        self.signal_history = []

    def next(self):
        """Execute trading logic."""
        if len(self.trades) >= self.max_open_positions:
            return

        # Get DataFrame
        df = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
                "Volume": self.data.Volume if hasattr(self.data, "Volume") else 0,
            }
        )
        df.index = self.data.index

        current_idx = len(df) - 1
        min_bars = max(p.min_bars_required for p in self.patterns)

        if current_idx < min_bars:
            return

        # Detect patterns
        long_count = 0
        short_count = 0
        long_signals = []
        short_signals = []

        for pattern in self.patterns:
            try:
                result = pattern.detect(df, current_idx)
                if result.detected and result.signal:
                    if result.signal.direction.value == "Long":
                        long_count += 1
                        long_signals.append(result.signal)
                    else:
                        short_count += 1
                        short_signals.append(result.signal)
            except:
                continue

        # Require at least 2 patterns to agree
        if long_count >= 2 and long_count > short_count:
            signal = long_signals[0]
            self._execute_long(signal)
        elif short_count >= 2 and short_count > long_count:
            signal = short_signals[0]
            self._execute_short(signal)

    def _execute_long(self, signal):
        """Execute long trade."""
        equity = self.equity
        risk = equity * self.risk_per_trade
        stop_distance = signal.entry_price - signal.stop_loss

        if stop_distance <= 0:
            return

        # Calculate position size as fraction of equity
        size_frac = (risk / stop_distance) / signal.entry_price
        size_frac = min(size_frac, 0.20)  # Max 20% of equity

        # Ensure valid fraction
        if size_frac <= 0 or size_frac > 1:
            return

        self.buy(size=size_frac, sl=signal.stop_loss)

    def _execute_short(self, signal):
        """Execute short trade."""
        equity = self.equity
        risk = equity * self.risk_per_trade
        stop_distance = signal.stop_loss - signal.entry_price

        if stop_distance <= 0:
            return

        # Calculate position size as fraction of equity
        size_frac = (risk / stop_distance) / signal.entry_price
        size_frac = min(size_frac, 0.20)  # Max 20% of equity

        # Ensure valid fraction
        if size_frac <= 0 or size_frac > 1:
            return

        self.sell(size=size_frac, sl=signal.stop_loss)
