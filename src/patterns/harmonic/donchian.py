"""
Donchian Channel Breakout Pattern

Detection Logic:
- Calculate Upper Channel = Max(High, lookback=20) (4-week range)
- Calculate Lower Channel = Min(Low, lookback=20)
- Calculate Mid Channel = (Upper Channel + Lower Channel) / 2
- Long Signal: Close[i] > Upper Channel[i-1]
- Short Signal: Close[i] < Lower Channel[i-1]

Entry Rules:
- Long Entry: Buy Stop at high[i] + 0.01 (Exceeds 20-period high)
- Short Entry: Sell Stop at low[i] - 0.01 (Falls below 20-period low)
- Entry triggered on close outside channel

Stop Loss Rules:
- Initial Stop: Mid Channel level
- Trailing Stop: Update to Mid Channel as price moves in favor
- Alternative Stop: 10-day Low (for Long) or 10-day High (for Short)

Take Profit Rules:
- Target 1: Entry + 1.5 * ATR(10)
- Target 2: Entry + 2.0 * ATR(10)
- Exit Signal: Price touches Mid Channel (if used as trailing stop)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from ..base import BasePattern, PatternType, SignalDirection, TradeSignal, PatternResult
from ...indicators.technical import atr, donchian_channel, average_range, volume_sma, adx


class DonchianChannel(BasePattern):
    """
    Donchian Channel Breakout Pattern Detector
    
    A trend-following breakout system based on channel breakouts.
    """
    
    def __init__(
        self,
        channel_period: int = 20,
        alt_stop_period: int = 10,
        entry_offset: float = 0.01,
        use_mid_channel_stop: bool = True,
        use_atr_targets: bool = True,
        volume_filter: bool = False,
        adx_filter: bool = False,
        adx_threshold: float = 25.0
    ):
        """
        Initialize Donchian Channel pattern detector.
        
        Args:
            channel_period: Period for channel calculation (default 20)
            alt_stop_period: Period for alternative stop (default 10)
            entry_offset: Price offset for entry orders
            use_mid_channel_stop: Use mid-channel as stop loss
            use_atr_targets: Use ATR-based targets
            volume_filter: Require volume confirmation
            adx_filter: Filter out low ADX environments
            adx_threshold: Minimum ADX for valid breakout
        """
        super().__init__(
            name="Donchian Channel Breakout",
            pattern_type=PatternType.BREAKOUT,
            min_bars_required=channel_period + 1
        )
        self.channel_period = channel_period
        self.alt_stop_period = alt_stop_period
        self.entry_offset = entry_offset
        self.use_mid_channel_stop = use_mid_channel_stop
        self.use_atr_targets = use_atr_targets
        self.volume_filter = volume_filter
        self.adx_filter = adx_filter
        self.adx_threshold = adx_threshold
    
    def _calculate_channels(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Calculate Donchian Channel values at bar i.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            
        Returns:
            Dictionary with channel values or None
        """
        if i < self.channel_period:
            return None
        
        # Calculate channel values
        channels = donchian_channel(df, self.channel_period)
        
        if i >= len(channels['upper']):
            return None
        
        upper = self._safe_float(channels['upper'].iloc[i])
        lower = self._safe_float(channels['lower'].iloc[i])
        middle = self._safe_float(channels['middle'].iloc[i])
        
        if np.isnan(upper) or np.isnan(lower) or np.isnan(middle):
            return None
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'prev_upper': self._safe_float(channels['upper'].iloc[i-1]) if i > 0 else upper,
            'prev_lower': self._safe_float(channels['lower'].iloc[i-1]) if i > 0 else lower
        }
    
    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        """
        Detect Donchian Channel breakout at bar index i.
        
        Args:
            df: DataFrame with OHLCV data
            i: Current bar index
            
        Returns:
            PatternResult with detection status and signal
        """
        if not self._validate_data(df, i):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        # Calculate channels
        channels = self._calculate_channels(df, i)
        
        if channels is None:
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type
            )
        
        current_close = self._safe_float(df.iloc[i]['Close'])
        current_high = self._safe_float(df.iloc[i]['High'])
        current_low = self._safe_float(df.iloc[i]['Low'])
        
        # Check for breakout
        breakout_up = current_close > channels['prev_upper']
        breakout_down = current_close < channels['prev_lower']
        
        if not (breakout_up or breakout_down):
            return PatternResult(
                detected=False,
                pattern_name=self.name,
                pattern_type=self.pattern_type,
                pivot_points=channels
            )
        
        # ADX filter
        if self.adx_filter:
            adx_values = adx(df, 14)
            if i < len(adx_values):
                current_adx = self._safe_float(adx_values.iloc[i])
                if not np.isnan(current_adx) and current_adx < self.adx_threshold:
                    return PatternResult(
                        detected=False,
                        pattern_name=self.name,
                        pattern_type=self.pattern_type,
                        pivot_points={**channels, 'adx': current_adx}
                    )
        
        # Generate signal
        signal = self._generate_signal(df, i, channels, breakout_up, breakout_down)
        
        return PatternResult(
            detected=True,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            signal=signal,
            pivot_points={
                **channels,
                'breakout_direction': 'up' if breakout_up else 'down'
            },
            bars_since_detection=0,
            start_index=i - self.channel_period,
            end_index=i
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trade signal."""
        result = self.detect(df, i)
        return result.signal if result.detected else None
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        i: int,
        channels: Dict,
        breakout_up: bool,
        breakout_down: bool
    ) -> Optional[TradeSignal]:
        """Generate trade signal for Donchian breakout."""
        
        current_high = self._safe_float(df.iloc[i]['High'])
        current_low = self._safe_float(df.iloc[i]['Low'])
        
        if breakout_up:
            # Long signal
            entry_price = current_high + self.entry_offset
            
            # Stop loss options
            if self.use_mid_channel_stop:
                stop_loss = channels['middle'] - self.entry_offset
            else:
                # Use 10-day low as alternative stop
                if i >= self.alt_stop_period:
                    alt_stop = df.iloc[i-self.alt_stop_period:i+1]['Low'].min()
                    stop_loss = self._safe_float(alt_stop) - self.entry_offset
                else:
                    stop_loss = channels['lower'] - self.entry_offset
            
            # ATR-based targets
            if self.use_atr_targets:
                atr_values = atr(df, 10)
                if i < len(atr_values):
                    current_atr = self._safe_float(atr_values.iloc[i])
                    if not np.isnan(current_atr):
                        take_profit_1 = entry_price + (current_atr * 1.5)
                        take_profit_2 = entry_price + (current_atr * 2.0)
                    else:
                        risk = entry_price - stop_loss
                        take_profit_1 = entry_price + (risk * 2)
                        take_profit_2 = entry_price + (risk * 3)
                else:
                    risk = entry_price - stop_loss
                    take_profit_1 = entry_price + (risk * 2)
                    take_profit_2 = entry_price + (risk * 3)
            else:
                risk = entry_price - stop_loss
                take_profit_1 = entry_price + (risk * 2)
                take_profit_2 = entry_price + (risk * 3)
            
            direction = SignalDirection.LONG
            breakout_dir = 'up'
        
        else:  # breakout_down
            # Short signal
            entry_price = current_low - self.entry_offset
            
            # Stop loss options
            if self.use_mid_channel_stop:
                stop_loss = channels['middle'] + self.entry_offset
            else:
                # Use 10-day high as alternative stop
                if i >= self.alt_stop_period:
                    alt_stop = df.iloc[i-self.alt_stop_period:i+1]['High'].max()
                    stop_loss = self._safe_float(alt_stop) + self.entry_offset
                else:
                    stop_loss = channels['upper'] + self.entry_offset
            
            # ATR-based targets
            if self.use_atr_targets:
                atr_values = atr(df, 10)
                if i < len(atr_values):
                    current_atr = self._safe_float(atr_values.iloc[i])
                    if not np.isnan(current_atr):
                        take_profit_1 = entry_price - (current_atr * 1.5)
                        take_profit_2 = entry_price - (current_atr * 2.0)
                    else:
                        risk = stop_loss - entry_price
                        take_profit_1 = entry_price - (risk * 2)
                        take_profit_2 = entry_price - (risk * 3)
                else:
                    risk = stop_loss - entry_price
                    take_profit_1 = entry_price - (risk * 2)
                    take_profit_2 = entry_price - (risk * 3)
            else:
                risk = stop_loss - entry_price
                take_profit_1 = entry_price - (risk * 2)
                take_profit_2 = entry_price - (risk * 3)
            
            direction = SignalDirection.SHORT
            breakout_dir = 'down'
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_filter:
            vol_sma = volume_sma(df['Volume'], 20)
            if i < len(vol_sma):
                current_vol = self._safe_float(df.iloc[i]['Volume'])
                avg_vol = self._safe_float(vol_sma.iloc[i])
                volume_confirmed = current_vol > avg_vol
        
        # Confidence
        confidence = 0.5
        if volume_confirmed:
            confidence += 0.1
        
        # Check for volatility expansion (breakout bar range > average range)
        avg_range = average_range(df, 20)
        if i < len(avg_range):
            current_range = current_high - current_low
            avg_r = self._safe_float(avg_range.iloc[i])
            if current_range > avg_r:
                confidence += 0.05
        
        return TradeSignal(
            pattern_name=f"{self.name} ({breakout_dir.capitalize()})",
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=None,
            confidence=min(confidence, 1.0),
            timestamp=df.iloc[i].name if hasattr(df.iloc[i], 'name') else None,
            metadata={
                'breakout_direction': breakout_dir,
                'upper_channel': channels['upper'],
                'lower_channel': channels['lower'],
                'mid_channel': channels['middle'],
                'volume_confirmed': volume_confirmed,
                'entry_type': 'buy_stop' if breakout_up else 'sell_stop'
            }
        )


class DonchianChannelTrend(BasePattern):
    """
    Donchian Channel Trend Filter
    
    Used to determine trend direction based on channel position.
    """
    
    def __init__(self, channel_period: int = 20):
        super().__init__(
            name="Donchian Channel Trend",
            pattern_type=PatternType.CONTINUATION,
            min_bars_required=channel_period + 1
        )
        self.channel_period = channel_period
    
    def get_trend(self, df: pd.DataFrame, i: int) -> str:
        """
        Get trend direction based on Donchian Channel.
        
        Args:
            df: DataFrame with OHLC data
            i: Current bar index
            
        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        if i < self.channel_period:
            return 'sideways'
        
        channels = donchian_channel(df, self.channel_period)
        
        if i >= len(channels['upper']):
            return 'sideways'
        
        upper = self._safe_float(channels['upper'].iloc[i])
        lower = self._safe_float(channels['lower'].iloc[i])
        middle = self._safe_float(channels['middle'].iloc[i])
        close = self._safe_float(df.iloc[i]['Close'])
        
        if np.isnan(upper) or np.isnan(lower) or np.isnan(middle):
            return 'sideways'
        
        # Price above mid-channel = uptrend
        if close > middle:
            return 'uptrend'
        # Price below mid-channel = downtrend
        elif close < middle:
            return 'downtrend'
        else:
            return 'sideways'
    
    def detect(self, df: pd.DataFrame, i: int) -> PatternResult:
        """Detect trend - not a pattern per se."""
        trend = self.get_trend(df, i)
        
        return PatternResult(
            detected=False,
            pattern_name=self.name,
            pattern_type=self.pattern_type,
            pivot_points={'trend': trend}
        )
    
    def generate_signal(self, df: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        return None
