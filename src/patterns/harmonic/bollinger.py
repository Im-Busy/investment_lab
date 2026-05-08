"""
Bollinger Bands Pattern

Detection Logic:
- Calculate Basis = SMA(Close, 20)
- Calculate Upper Band = Basis + 2 * StdDev(Close, 20)
- Calculate Lower Band = Basis - 2 * StdDev(Close, 20)
- Squeeze Condition: (Upper Band - Lower Band) < SMA(Upper Band - Lower Band, 20) * 0.5
- Breakout Signal: Close[i] > Upper Band (Long) or Close[i] < Lower Band (Short)
- Reversal Signal: Close outside band, then subsequent Close inside band

Entry Rules:
- Breakout Entry: Buy Stop = high[i] + 0.01 (if Close > Upper Band after Squeeze)
- Reversal Entry: Sell Stop = low[i] - 0.01 (if Close > Upper Band then Close < Upper Band)
- Focus on Breakout from Contraction per text 'impending sharp price change'

Stop Loss Rules:
- Stop: Basis (Middle Band) level
- Alternative Stop: Opposite Band (for wide stops)

Take Profit Rules:
- Target 1: Basis + (Upper Band - Basis) * 2 (Projection of band width)
- Target 2: Exit when price touches opposite band
- Target 3: Fixed ATR multiple (e.g., 2 * ATR)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...indicators.technical import atr, bollinger_bands, volume_sma
from ..base import BasePattern, PatternResult, PatternType, SignalDirection, TradeSignal


class BollingerBands(BasePattern):
    """
    Bollinger Bands Pattern Detector

    Detects breakouts from volatility contraction (squeeze) and
    potential reversal signals when price exits and re-enters bands.
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        squeeze_threshold: float = 0.5,
        entry_offset: float = 0.01,
        use_middle_band_stop: bool = True,
        use_atr_targets: bool = True,
        volume_filter: bool = False,
    ):
        """
        Initialize Bollinger Bands pattern detector.

        Args:
            period: Bollinger Band period (default 20)
            num_std: Number of standard deviations (default 2.0)
            squeeze_threshold: Bandwidth threshold for squeeze (default 0.5)
            entry_offset: Price offset for entry orders
            use_middle_band_stop: Use middle band as stop loss
            use_atr_targets: Use ATR-based targets
            volume_filter: Require volume confirmation
        """
        super().__init__(
            name="Bollinger Bands",
            pattern_type=PatternType.VOLATILITY,
            min_bars_required=period + 1,
        )
        self.period = period
        self.num_std = num_std
        self.squeeze_threshold = squeeze_threshold
        self.entry_offset = entry_offset
        self.use_middle_band_stop = use_middle_band_stop
        self.use_atr_targets = use_atr_targets
        self.volume_filter = volume_filter

    def detect_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized detection of Bollinger Bands breakout and reversal signals.

        Returns:
            np.ndarray of np.int8: 0=no signal, 1=LONG, -1=SHORT
        """
        n = len(df)
        result = np.zeros(n, dtype=np.int8)
        if n < self.period + 1:
            return result

        close_a = df["Close"].to_numpy()
        period = self.period
        num_std = self.num_std
        sq_thresh = self.squeeze_threshold

        sma = df["Close"].rolling(period).mean().to_numpy()
        std = df["Close"].rolling(period).std(ddof=0).to_numpy()

        upper = sma + num_std * std
        lower = sma - num_std * std
        bandwidth = (upper - lower) / sma
        avg_bandwidth = pd.Series(bandwidth).rolling(period).mean().to_numpy()

        for i in range(period, n):
            if np.isnan(sma[i]):
                continue

            sq_ratio = bandwidth[i] / avg_bandwidth[i] if avg_bandwidth[i] > 0 else 0.0
            is_squeeze = sq_ratio < sq_thresh

            up_band = upper[i]
            lo_band = lower[i]
            prev_up = upper[i - 1] if i > 0 else up_band
            prev_lo = lower[i - 1] if i > 0 else lo_band

            # Breakout from squeeze
            if close_a[i] > up_band and is_squeeze:
                result[i] = 1
            elif close_a[i] < lo_band and is_squeeze:
                result[i] = -1
            elif i > 0:
                # Reversal: was below lower band, now back inside → bullish
                prev_close = close_a[i - 1]
                if prev_close < prev_lo and close_a[i] > lo_band:
                    result[i] = 1
                elif prev_close > prev_up and close_a[i] < up_band:
                    result[i] = -1

        return result

    def _calculate_bands(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Calculate Bollinger Band values at bar i.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index

        Returns:
            Dictionary with band values or None
        """
        if i < self.period:
            return None

        bands = bollinger_bands(df["Close"], self.period, self.num_std)

        if i >= len(bands["upper"]):
            return None

        upper = self._safe_float(bands["upper"].iloc[i])
        lower = self._safe_float(bands["lower"].iloc[i])
        middle = self._safe_float(bands["middle"].iloc[i])
        bandwidth = self._safe_float(bands["bandwidth"].iloc[i])

        if np.isnan(upper) or np.isnan(lower) or np.isnan(middle):
            return None

        # Calculate average bandwidth for squeeze detection
        avg_bandwidth = self._safe_float(bands["bandwidth"].rolling(window=20).mean().iloc[i])

        # Previous band values for reversal detection
        prev_upper = self._safe_float(bands["upper"].iloc[i - 1]) if i > 0 else upper
        prev_lower = self._safe_float(bands["lower"].iloc[i - 1]) if i > 0 else lower

        return {
            "upper": upper,
            "lower": lower,
            "middle": middle,
            "bandwidth": bandwidth,
            "avg_bandwidth": avg_bandwidth,
            "prev_upper": prev_upper,
            "prev_lower": prev_lower,
        }

    def _is_squeeze(self, bands: Dict) -> bool:
        """
        Check if Bollinger Bands are in squeeze condition.

        Args:
            bands: Dictionary with band values

        Returns:
            True if squeeze detected
        """
        if bands["avg_bandwidth"] == 0:
            return False

        squeeze_ratio = bands["bandwidth"] / bands["avg_bandwidth"]
        return bool(squeeze_ratio < self.squeeze_threshold)

    def _detect_breakout(self, df: pd.DataFrame, i: int, bands: Dict) -> Optional[str]:
        """
        Detect Bollinger Band breakout.

        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            bands: Dictionary with band values

        Returns:
            'up', 'down', or None
        """
        current_close = self._safe_float(df.iloc[i]["Close"])

        if current_close > bands["upper"]:
            return "up"
        elif current_close < bands["lower"]:
            return "down"

        return None

    def _detect_reversal(self, df: pd.DataFrame, i: int, bands: Dict) -> Optional[str]:
        """
        Detect Bollinger Band reversal (outside then inside).

        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            bands: Dictionary with band values

        Returns:
            'bullish_reversal', 'bearish_reversal', or None
        """
        if i < 1:
            return None

        prev_close = self._safe_float(df.iloc[i - 1]["Close"])
        current_close = self._safe_float(df.iloc[i]["Close"])

        # Bullish reversal: was below lower band, now back inside
        if prev_close < bands["prev_lower"] and current_close > bands["lower"]:
            return "bullish_reversal"

        # Bearish reversal: was above upper band, now back inside
        if prev_close > bands["prev_upper"] and current_close < bands["upper"]:
            return "bearish_reversal"

        return None

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """
        Detect Bollinger Band pattern at bar index i.

        Args:
            df: DataFrame with OHLCV data
            i: Current bar index

        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Calculate bands
        bands = self._calculate_bands(df, i)

        if bands is None:
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        # Check for squeeze
        is_squeeze = self._is_squeeze(bands)
        bands["is_squeeze"] = is_squeeze

        # Check for breakout
        breakout = self._detect_breakout(df, i, bands)

        # Check for reversal
        reversal = self._detect_reversal(df, i, bands)

        # Generate signal based on detected condition
        signal = None
        pattern_type = None

        if breakout and is_squeeze:
            # Breakout from squeeze - primary pattern
            signal = self._generate_breakout_signal(df, i, bands, breakout)
            pattern_type = "squeeze_breakout"
        elif reversal:
            # Reversal pattern
            signal = self._generate_reversal_signal(df, i, bands, reversal)
            pattern_type = reversal
        elif breakout:
            # Regular breakout (no squeeze)
            signal = self._generate_breakout_signal(df, i, bands, breakout)
            pattern_type = "breakout"

        if signal is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points={
                    **bands,
                    "is_squeeze": is_squeeze,
                    "breakout": breakout,
                    "reversal": reversal,
                },
            )

        return PatternResult(
            detected=True,
            pattern_name=f"{self.name} ({(pattern_type or 'unknown').replace('_', ' ').title()})"
            if pattern_type
            else f"{self.name} (Unknown)",
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                **bands,
                "is_squeeze": is_squeeze,
                "breakout": breakout,
                "reversal": reversal,
                "pattern_type": pattern_type,
            },
            bars_since_detection=0,
            start_index=i - self.period,
            end_index=i,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None

    def _generate_breakout_signal(
        self, df: pd.DataFrame, i: int, bands: Dict, breakout: str
    ) -> Optional[TradeSignal]:
        """Generate signal for Bollinger Band breakout."""

        current_high = self._safe_float(df.iloc[i]["High"])
        current_low = self._safe_float(df.iloc[i]["Low"])

        if breakout == "up":
            entry_price = current_high + self.entry_offset

            if self.use_middle_band_stop:
                stop_loss = bands["middle"] - self.entry_offset
            else:
                stop_loss = bands["lower"] - self.entry_offset

            band_width = bands["upper"] - bands["middle"]

            if self.use_atr_targets:
                atr_values = atr(df, 14)
                if i < len(atr_values):
                    current_atr = self._safe_float(atr_values.iloc[i])
                    if not np.isnan(current_atr):
                        take_profit_1 = entry_price + (current_atr * 2)
                        take_profit_2 = entry_price + (current_atr * 3)
                    else:
                        take_profit_1 = entry_price + (band_width * 2)
                        take_profit_2 = bands["lower"]  # Opposite band
                else:
                    take_profit_1 = entry_price + (band_width * 2)
                    take_profit_2 = bands["lower"]
            else:
                take_profit_1 = entry_price + (band_width * 2)
                take_profit_2 = bands["lower"]

            direction = SignalDirection.LONG

        else:  # breakout == 'down'
            entry_price = current_low - self.entry_offset

            if self.use_middle_band_stop:
                stop_loss = bands["middle"] + self.entry_offset
            else:
                stop_loss = bands["upper"] + self.entry_offset

            band_width = bands["middle"] - bands["lower"]

            if self.use_atr_targets:
                atr_values = atr(df, 14)
                if i < len(atr_values):
                    current_atr = self._safe_float(atr_values.iloc[i])
                    if not np.isnan(current_atr):
                        take_profit_1 = entry_price - (current_atr * 2)
                        take_profit_2 = entry_price - (current_atr * 3)
                    else:
                        take_profit_1 = entry_price - (band_width * 2)
                        take_profit_2 = bands["upper"]  # Opposite band
                else:
                    take_profit_1 = entry_price - (band_width * 2)
                    take_profit_2 = bands["upper"]
            else:
                take_profit_1 = entry_price - (band_width * 2)
                take_profit_2 = bands["upper"]

            direction = SignalDirection.SHORT

        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]["Volume"])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol

        # Confidence
        confidence = 0.5
        if bands["is_squeeze"]:
            confidence += 0.15  # Squeeze breakouts are stronger
        if volume_confirmed:
            confidence += 0.1

        return TradeSignal(
            pattern_name=f"{self.name} ({breakout.title()} Breakout)",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i],
            metadata={
                "breakout_direction": breakout,
                "is_squeeze": bands["is_squeeze"],
                "upper_band": bands["upper"],
                "middle_band": bands["middle"],
                "lower_band": bands["lower"],
                "bandwidth": bands["bandwidth"],
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop" if breakout == "up" else "sell_stop",
            },
        )

    def _generate_reversal_signal(
        self, df: pd.DataFrame, i: int, bands: Dict, reversal: str
    ) -> Optional[TradeSignal]:
        """Generate signal for Bollinger Band reversal."""

        current_high = self._safe_float(df.iloc[i]["High"])
        current_low = self._safe_float(df.iloc[i]["Low"])

        if reversal == "bullish_reversal":
            # Price was below lower band, now back inside - bullish
            entry_price = current_high + self.entry_offset
            stop_loss = current_low - self.entry_offset

            take_profit_1 = bands["middle"]
            take_profit_2 = bands["upper"]

            direction = SignalDirection.LONG

        else:  # reversal == 'bearish_reversal'
            # Price was above upper band, now back inside - bearish
            entry_price = current_low - self.entry_offset
            stop_loss = current_high + self.entry_offset

            take_profit_1 = bands["middle"]
            take_profit_2 = bands["lower"]

            direction = SignalDirection.SHORT

        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df["Volume"], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]["Volume"])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol

        confidence = 0.55
        if volume_confirmed:
            confidence += 0.1

        return TradeSignal(
            pattern_name=f"{self.name} ({reversal.replace('_', ' ').title()})",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.index[i],
            metadata={
                "reversal_type": reversal,
                "upper_band": bands["upper"],
                "middle_band": bands["middle"],
                "lower_band": bands["lower"],
                "volume_confirmed": volume_confirmed,
                "entry_type": "buy_stop" if reversal == "bullish_reversal" else "sell_stop",
            },
        )


class BollingerSqueeze(BasePattern):
    """
    Bollinger Band Squeeze Detector

    Identifies periods of low volatility that often precede
    significant price moves.
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        squeeze_threshold: float = 0.5,
        lookback: int = 20,
    ):
        super().__init__(
            name="Bollinger Squeeze",
            pattern_type=PatternType.VOLATILITY,
            min_bars_required=period + lookback,
        )
        self.period = period
        self.num_std = num_std
        self.squeeze_threshold = squeeze_threshold
        self.lookback = lookback

    def detect(self, df: pd.DataFrame, i: int, window_start: Optional[int] = None) -> PatternResult:
        """Detect Bollinger Band squeeze."""
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        bands = bollinger_bands(df["Close"], self.period, self.num_std)

        if i >= len(bands["bandwidth"]):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        bandwidth = self._safe_float(bands["bandwidth"].iloc[i])
        avg_bandwidth = self._safe_float(
            bands["bandwidth"].rolling(window=self.lookback).mean().iloc[i]
        )

        if np.isnan(bandwidth) or np.isnan(avg_bandwidth):
            return PatternResult(
                detected=False, pattern_name=self.name, pattern_type=self.pattern_type
            )

        squeeze_ratio = bandwidth / avg_bandwidth if avg_bandwidth > 0 else 1
        is_squeeze = squeeze_ratio < self.squeeze_threshold

        return PatternResult(
            detected=is_squeeze,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            pivot_points={
                "bandwidth": bandwidth,
                "avg_bandwidth": avg_bandwidth,
                "squeeze_ratio": squeeze_ratio,
                "is_squeeze": is_squeeze,
                "upper": self._safe_float(bands["upper"].iloc[i]),
                "middle": self._safe_float(bands["middle"].iloc[i]),
                "lower": self._safe_float(bands["lower"].iloc[i]),
            },
            bars_since_detection=0,
        )

    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Squeeze detection doesn't generate signals directly."""
        return None
