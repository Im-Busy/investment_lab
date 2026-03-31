"""
Optimized Multi-Pattern Strategy for backtesting.py

Performance optimizations (Phase 1):
1. DataFrame cached once in init(), not created every bar
2. Common indicators pre-computed once via IndicatorCache
3. No DataFrame slicing - pass window bounds instead
4. NumPy arrays for fast access in pattern detectors
5. Pre-allocated result arrays
6. Parallel pattern detection with ThreadPoolExecutor
7. Batch processing support

This version maintains the same functionality but runs significantly faster.
"""

import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from backtesting import Strategy

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import indicator cache for performance optimization
from src.indicators.indicator_cache import IndicatorCache

# Import pattern detectors
from src.indicators.regime import MarketRegimeDetector
from src.patterns.basic.floor_pivot import FloorPivotBreakout
from src.patterns.basic.matching_lows import MatchingLows
from src.patterns.basic.msl import MarketStructureLow
from src.patterns.basic.n_bar_decline import NBarDecline
from src.patterns.basic.nr7id import NR7ID
from src.patterns.basic.two_bar_reversal import TwoBarReversal
from src.patterns.classic.dead_cat_bounce import DeadCatBounce
from src.patterns.classic.double_bottom import DoubleBottom
from src.patterns.classic.double_top import DoubleTop
from src.patterns.classic.trader_vic_2b import TraderVic2B
from src.patterns.classic.triple_top import TripleTop
from src.patterns.classic.triple_bottom import TripleBottom
from src.patterns.classic.ascending_triangle import AscendingTriangle
from src.patterns.classic.descending_triangle import DescendingTriangle
from src.patterns.classic.rectangle import Rectangle
from src.patterns.classic.wedge import Wedge
from src.patterns.complex.cup_handle import CupAndHandle
from src.patterns.complex.head_shoulders import HeadAndShoulders
from src.patterns.complex.parabolic_arc import ParabolicArc
from src.patterns.complex.spike_ledge import SpikeAndLedge
from src.patterns.complex.three_hills import ThreeHillsMountain
from src.patterns.harmonic.abc import ABCPattern
from src.patterns.harmonic.bollinger import BollingerBands
from src.patterns.harmonic.donchian import DonchianChannel
from src.patterns.harmonic.gartley import GartleyPattern
from src.patterns.harmonic.symmetric_triangle import SymmetricTriangle
from src.patterns.continuation.flag import Flag
from src.patterns.continuation.pennant import Pennant
from src.patterns.breakout.gap import GapPattern

# Import candlestick patterns
from src.patterns.candlestick.doji import Doji
from src.patterns.candlestick.harami import Harami
from src.patterns.candlestick.hammer import Hammer
from src.patterns.candlestick.engulfing import Engulfing
from src.patterns.candlestick.dark_cloud import DarkCloudCover, PiercingLine

# Import confluence and regime detection
from src.strategies.confluence import ConfluenceScorer


@dataclass
class SimpleConfluence:
    """Simple confluence result for internal use."""

    score: float
    direction: str
    pattern_count: int


class MultiPatternStrategyOptimized(Strategy):
    """
    Optimized Multi-Pattern Confluence Strategy for backtesting.py

    PERFORMANCE OPTIMIZATIONS:
    1. DataFrame created ONCE in init(), not every bar
    2. Common indicators pre-computed once
    3. Sliding window for pattern detection
    4. Min bars check done once in init()

    Uses confluence of multiple pattern detections to generate trading signals.
    Supports 20 patterns across 4 categories: basic, harmonic, complex, classic.

    Parameters:
        min_confidence: Minimum confidence threshold for signals
        min_confluence_count: Minimum patterns that must agree
        risk_per_trade: Risk per trade as fraction of equity
        max_open_positions: Maximum concurrent positions
        use_regime_filter: Whether to filter by market regime
        enable_signal_log: Enable signal event logging for contribution analysis (opt-in)
        exclude_patterns: Comma-separated pattern names to exclude (for ablation)
        include_patterns_only: Comma-separated pattern names to include exclusively (for synergy)
    """

    # Strategy parameters (can be optimized)
    min_confidence = 0.60
    min_confluence_count = 2
    risk_per_trade = 0.02
    max_open_positions = 5
    use_regime_filter = True
    enable_signal_log = False  # Opt-in to avoid perf impact when not needed
    exclude_patterns = ""  # Comma-separated pattern names to exclude
    include_patterns_only = ""  # If set, ONLY these patterns are active

    def init(self):
        """Initialize indicators and pattern detectors - OPTIMIZED with Phase 1 & 2 enhancements."""
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

        # OPTIMIZATION 1: Create DataFrame ONCE and cache it
        self._df = self._create_dataframe()

        # OPTIMIZATION 2: Create IndicatorCache and pre-compute common indicators
        self._indicator_cache = IndicatorCache(self._df)
        self._indicator_cache.pre_compute_common()

        # OPTIMIZATION 3: Set indicator cache for all patterns that support it
        for pattern in self.patterns:
            if hasattr(pattern, "set_indicator_cache"):
                pattern.set_indicator_cache(self._indicator_cache)

        # OPTIMIZATION 4: Pre-compute min_bars_required
        self._min_bars = max(p.min_bars_required for p in self.patterns)

        # OPTIMIZATION 5: Pre-extract NumPy arrays for fast access
        self._arrays = self._indicator_cache.arrays

        # OPTIMIZATION 6: Setup parallel detection with ThreadPoolExecutor
        self._max_workers = min(len(self.patterns), multiprocessing.cpu_count())
        self._executor: Optional[ThreadPoolExecutor] = None
        # Note: ThreadPoolExecutor will be created lazily to avoid issues with backtesting.py

        # Track current bar index
        self._current_bar = 0

        # PHASE 2 OPTIMIZATION: Pre-compute all pattern signals using vectorized detection
        self._pattern_signals_cache: Dict[str, Optional[np.ndarray]] = {}
        self._precompute_all_signals()

        # Initialize signal event log for contribution analysis (if enabled)
        self._signal_event_log = None
        if self.enable_signal_log:
            from src.analysis.signal_event_log import SignalEventLog
            self._signal_event_log = SignalEventLog()

    def _create_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from backtesting.py data - called ONCE."""
        df = pd.DataFrame(
            {
                "Open": np.array(self.data.Open),
                "High": np.array(self.data.High),
                "Low": np.array(self.data.Low),
                "Close": np.array(self.data.Close),
                "Volume": np.array(self.data.Volume)
                if hasattr(self.data, "Volume")
                else np.zeros(len(self.data.Close)),
            }
        )
        df.index = self.data.index
        return df

    def _init_patterns(self) -> List:
        """Initialize all pattern detectors."""
        patterns = []

        # Basic patterns
        patterns.extend(
            [MarketStructureLow(), MatchingLows(), NR7ID(), NBarDecline(), FloorPivotBreakout(), TwoBarReversal()]
        )

        # Harmonic patterns
        patterns.extend(
            [
                GartleyPattern(),
                ABCPattern(),
                SymmetricTriangle(),
                DonchianChannel(),
                BollingerBands(),
            ]
        )

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
        patterns.extend(
            [
                DoubleTop(),
                DoubleBottom(),
                TraderVic2B(),
                TripleTop(),
                TripleBottom(),
                AscendingTriangle(),
                DescendingTriangle(),
                Rectangle(),
                Wedge(),
                DeadCatBounce(),
            ]
        )

        # Continuation patterns
        patterns.extend(
            [
                Flag(),
                Pennant(),
            ]
        )

        # Breakout patterns
        patterns.extend(
            [
                GapPattern(),
            ]
        )

        # Candlestick patterns (lower confidence, require confirmation)
        patterns.extend(
            [
                Doji(),
                Harami(),
                Hammer(),
                Engulfing(),
                DarkCloudCover(),
                PiercingLine(),
            ]
        )

        # Apply include/exclude filters for contribution analysis
        if self.include_patterns_only:
            include_set = {p.strip() for p in self.include_patterns_only.split(",")}
            patterns = [p for p in patterns if p.name in include_set]
        elif self.exclude_patterns:
            exclude_set = {p.strip() for p in self.exclude_patterns.split(",")}
            patterns = [p for p in patterns if p.name not in exclude_set]

        return patterns

    def _precompute_all_signals(self) -> None:
        """
        PHASE 2 OPTIMIZATION: Pre-compute signals for all patterns using vectorized detection.

        This method is called once during init() to pre-compute all pattern signals
        across the entire dataset. This provides 50-100x speedup during backtesting
        as signals are looked up from cache instead of re-detected each bar.
        """
        for pattern in self.patterns:
            try:
                # Check if pattern has vectorized detection
                if hasattr(pattern, "detect_vectorized"):
                    signals = pattern.detect_vectorized(self._df)
                    self._pattern_signals_cache[pattern.name] = signals
                    # Also call precompute_signals to set internal cache
                    if hasattr(pattern, "precompute_signals"):
                        pattern.precompute_signals(self._df)
            except Exception:
                # Fall back to no caching for this pattern
                self._pattern_signals_cache[pattern.name] = None

    def _get_cached_signal(self, pattern_name: str, idx: int) -> int:
        """
        Get pre-computed signal for a pattern at a specific bar index.

        Args:
            pattern_name: Name of the pattern
            idx: Bar index

        Returns:
            Signal value: 0=none, 1=long, -1=short
        """
        signals = self._pattern_signals_cache.get(pattern_name)
        if signals is not None and idx < len(signals):
            return int(signals[idx])
        return 0

    def _detect_pattern(
        self, pattern: Any, df: pd.DataFrame, idx: int, window_start: int
    ) -> Optional[Dict[str, Any]]:
        """
        Helper method for pattern detection (used for parallel execution).

        PHASE 2 OPTIMIZATION: Uses cached vectorized signals when available.

        Args:
            pattern: Pattern detector instance
            df: DataFrame with OHLCV data
            idx: Current bar index
            window_start: Window start for bounds checking

        Returns:
            Signal dict if pattern detected, None otherwise
        """
        try:
            # PHASE 2: Check cached signals first
            cached_signal = self._get_cached_signal(pattern.name, idx)
            if cached_signal != 0:
                # Pattern detected in cache, get full details via detect()
                result = pattern.detect(df, idx, window_start=window_start)
                if result.detected and result.signal:
                    return {
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
            elif pattern.name in self._pattern_signals_cache:
                # Pattern has vectorized detection but no signal at this bar
                return None
            else:
                # No vectorized detection available, use standard detect()
                result = pattern.detect(df, idx, window_start=window_start)
                if result.detected and result.signal:
                    return {
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
        except Exception:
            pass
        return None

    def _detect_patterns_parallel(
        self, df: pd.DataFrame, idx: int, window_start: int
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns using parallel execution.

        Args:
            df: DataFrame with OHLCV data
            idx: Current bar index
            window_start: Window start for bounds checking

        Returns:
            List of signal dicts from detected patterns
        """
        # Create executor lazily
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        signals = []
        futures = [
            self._executor.submit(self._detect_pattern, pattern, df, idx, window_start)
            for pattern in self.patterns
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                signals.append(result)

        return signals

    def _detect_patterns_sequential(
        self, df: pd.DataFrame, idx: int, window_start: int
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns sequentially (for comparison/debugging).

        Args:
            df: DataFrame with OHLCV data
            idx: Current bar index
            window_start: Window start for bounds checking

        Returns:
            List of signal dicts from detected patterns
        """
        signals = []
        for pattern in self.patterns:
            result = self._detect_pattern(pattern, df, idx, window_start)
            if result is not None:
                signals.append(result)
        return signals

    def next(self):
        """Execute trading logic for current bar - OPTIMIZED with Phase 1 enhancements."""
        # Check position limit
        if len(self.trades) >= self.max_open_positions:
            return

        # OPTIMIZATION: Use pre-computed min_bars
        current_idx = len(self.data.Close) - 1
        if current_idx < self._min_bars:
            return

        # OPTIMIZATION 2: No DataFrame slicing - pass window bounds instead
        # Calculate window start for pattern detectors
        window_start = max(0, current_idx - 200)  # Max lookback for any pattern

        # OPTIMIZATION 6: Parallel pattern detection (can be toggled)
        # Use sequential for stability; enable parallel for CPU-heavy workloads
        use_parallel = False  # Set to True to enable parallel detection

        if use_parallel:
            signals = self._detect_patterns_parallel(self._df, current_idx, window_start)
        else:
            signals = self._detect_patterns_sequential(self._df, current_idx, window_start)

        # LOG ALL detections (even sub-threshold) for contribution analysis
        if hasattr(self, '_signal_event_log') and self._signal_event_log is not None:
            self._signal_event_log.record_bar_detections(
                bar_index=current_idx,
                timestamp=self.data.index[current_idx],
                all_detections=signals,
                active_patterns=None,  # filled below if threshold passes
                confluence_count=len(signals),
                passed_threshold=False,  # updated below
            )

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
            avg_confidence = sum(s["confidence"] for s in active_signals) / len(active_signals)

            if avg_confidence < self.min_confidence:
                return

            confluence = SimpleConfluence(
                score=avg_confidence, direction=direction, pattern_count=len(active_signals)
            )

        except Exception:
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
        position_size_frac = (risk_amount / stop_distance) / entry_price

        # Limit position size to max 20% of equity
        position_size_frac = min(position_size_frac, 0.20)

        # Ensure position size is valid (between 0 and 1)
        if position_size_frac <= 0 or position_size_frac > 1:
            return

        # Execute trade
        if direction == "LONG":
            if any(t.is_long for t in self.trades):
                return
            self.buy(size=position_size_frac, sl=stop_loss)
        else:  # SHORT
            if any(t.is_short for t in self.trades):
                return
            self.sell(size=position_size_frac, sl=stop_loss)

        # Update signal event log with passed threshold info
        if hasattr(self, '_signal_event_log') and self._signal_event_log is not None:
            self._signal_event_log.update_bar_passed_threshold(
                bar_index=current_idx,
                active_patterns=[s["pattern_name"] for s in active_signals],
                confluence_count=len(active_signals),
            )

        # Record signal for analysis
        self.signal_count += 1
        self.signals.append(
            {
                "bar": current_idx,
                "direction": direction,
                "patterns": [s["pattern_name"] for s in active_signals],
                "confidence": confluence.score if confluence else 0.5,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
            }
        )


class MultiPatternStrategyFast(Strategy):
    """
    Fast Multi-Pattern Strategy - Uses only the most reliable patterns.

    Optimized for speed while maintaining strategy essence:
    - Only 5 most reliable patterns (vs 20)
    - DataFrame created once in init()
    - Minimal indicator computation
    - Streamlined signal logic

    Use this for rapid backtesting and parameter optimization.
    """

    # Strategy parameters
    min_confidence = 0.55
    min_confluence_count = 2
    risk_per_trade = 0.02
    max_open_positions = 3

    def init(self):
        """Initialize with just the most reliable patterns - OPTIMIZED."""
        # Use a subset of patterns for faster backtesting
        self.patterns = [
            DoubleTop(),
            DoubleBottom(),
            HeadAndShoulders(),
            MarketStructureLow(),
            NR7ID(),
        ]

        # OPTIMIZATION: Create DataFrame once
        self._df = pd.DataFrame(
            {
                "Open": np.array(self.data.Open),
                "High": np.array(self.data.High),
                "Low": np.array(self.data.Low),
                "Close": np.array(self.data.Close),
                "Volume": np.array(self.data.Volume)
                if hasattr(self.data, "Volume")
                else np.zeros(len(self.data.Close)),
            }
        )
        self._df.index = self.data.index

        # Pre-compute min bars
        self._min_bars = max(p.min_bars_required for p in self.patterns)

        # Signal history
        self.signal_history = []

    def next(self):
        """Execute trading logic - OPTIMIZED."""
        if len(self.trades) >= self.max_open_positions:
            return

        current_idx = len(self.data.Close) - 1

        if current_idx < self._min_bars:
            return

        # OPTIMIZATION: Use sliding window
        window_start = max(0, current_idx - 100)
        df_window = self._df.iloc[window_start : current_idx + 1]
        window_idx = len(df_window) - 1

        # Detect patterns
        long_count = 0
        short_count = 0
        long_signals = []
        short_signals = []

        for pattern in self.patterns:
            try:
                result = pattern.detect(df_window, window_idx)
                if result.detected and result.signal:
                    if result.signal.direction.value == "Long":
                        long_count += 1
                        long_signals.append(result.signal)
                    else:
                        short_count += 1
                        short_signals.append(result.signal)
            except:
                continue

        # Require at least min_confluence_count patterns to agree
        if long_count >= self.min_confluence_count and long_count > short_count:
            signal = long_signals[0]
            self._execute_long(signal)
        elif short_count >= self.min_confluence_count and short_count > long_count:
            signal = short_signals[0]
            self._execute_short(signal)

    def _execute_long(self, signal):
        """Execute long trade."""
        equity = self.equity
        risk = equity * self.risk_per_trade
        stop_distance = signal.entry_price - signal.stop_loss

        if stop_distance <= 0:
            return

        size_frac = (risk / stop_distance) / signal.entry_price
        size_frac = min(size_frac, 0.20)

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

        size_frac = (risk / stop_distance) / signal.entry_price
        size_frac = min(size_frac, 0.20)

        if size_frac <= 0 or size_frac > 1:
            return

        self.sell(size=size_frac, sl=signal.stop_loss)
