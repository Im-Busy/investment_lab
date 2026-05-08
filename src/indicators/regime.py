"""
Market Regime Detector

Detects market conditions including trend strength, volatility regime,
and market phase to adapt pattern selection and risk parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    """Trend direction classification."""

    UPTREND = "Uptrend"
    DOWNTREND = "Downtrend"
    SIDEWAYS = "Sideways"


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    HIGH = "High"
    NORMAL = "Normal"
    LOW = "Low"


class MarketPhase(Enum):
    """Market phase classification."""

    STRONG_BULL = "Strong Bull"
    WEAK_BULL = "Weak Bull"
    STRONG_BEAR = "Strong Bear"
    WEAK_BEAR = "Weak Bear"
    RANGING = "Ranging"
    CHOPPY = "Choppy"


@dataclass
class RegimeState:
    """
    Current market regime state.

    Attributes:
        trend_direction: Current trend direction
        trend_strength: ADX-based trend strength (0-100)
        volatility_regime: Current volatility level
        volatility_value: Current ATR relative to average
        market_phase: Overall market phase
        sma_position: Price position relative to SMA (above/below)
        adx: Current ADX value
        atr: Current ATR value
        atr_ratio: ATR ratio to moving average
    """

    trend_direction: TrendDirection
    trend_strength: float
    volatility_regime: VolatilityRegime
    volatility_value: float
    market_phase: MarketPhase
    sma_position: str
    adx: float
    atr: float
    atr_ratio: float

    def __lt__(self, other: "RegimeState") -> bool:
        return self.to_tuple() < other.to_tuple()

    def __le__(self, other: "RegimeState") -> bool:
        return self.to_tuple() <= other.to_tuple()

    def __gt__(self, other: "RegimeState") -> bool:
        return self.to_tuple() > other.to_tuple()

    def __ge__(self, other: "RegimeState") -> bool:
        return self.to_tuple() >= other.to_tuple()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegimeState):
            return NotImplemented
        return self.to_tuple() == other.to_tuple()

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def to_tuple(self) -> tuple:
        return (
            self.trend_direction.value,
            self.volatility_regime.value,
            self.market_phase.value,
            self.sma_position,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert regime state to dictionary."""
        return {
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "volatility_regime": self.volatility_regime.value,
            "volatility_value": self.volatility_value,
            "market_phase": self.market_phase.value,
            "sma_position": self.sma_position,
            "adx": self.adx,
            "atr": self.atr,
            "atr_ratio": self.atr_ratio,
        }


class MarketRegimeDetector:
    """
    Market Regime Detector

    Analyzes market conditions to determine trend, volatility, and phase.
    Used to adapt pattern selection and risk parameters.
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_strong_threshold: float = 30.0,
        atr_period: int = 14,
        atr_lookback: int = 50,
        atr_high_mult: float = 1.5,
        atr_low_mult: float = 0.75,
        sma_period: int = 50,
        sma_fast_period: int = 20,
    ):
        """
        Initialize Market Regime Detector.

        Args:
            adx_period: Period for ADX calculation
            adx_trend_threshold: ADX level indicating trend presence
            adx_strong_threshold: ADX level indicating strong trend
            atr_period: Period for ATR calculation
            atr_lookback: Lookback for ATR moving average
            atr_high_mult: Multiplier for high volatility threshold
            atr_low_mult: Multiplier for low volatility threshold
            sma_period: Period for trend SMA
            sma_fast_period: Period for fast SMA
        """
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_threshold = adx_strong_threshold
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.atr_high_mult = atr_high_mult
        self.atr_low_mult = atr_low_mult
        self.sma_period = sma_period
        self.sma_fast_period = sma_fast_period

    def _calculate_adx(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX, +DI, and -DI.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (ADX, +DI, -DI) Series
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth the values
        atr = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=self.adx_period).mean() / atr

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()

        return adx, plus_di, minus_di

    def _calculate_atr_ratio(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ATR and its ratio to moving average.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (ATR, ATR_MA, ATR_Ratio) Series
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = tr.rolling(window=self.atr_period).mean()

        # Calculate ATR moving average
        atr_ma = atr.rolling(window=self.atr_lookback).mean()

        # Calculate ATR ratio
        atr_ratio = atr / atr_ma

        return atr, atr_ma, atr_ratio

    def _determine_trend_direction(
        self, df: pd.DataFrame, plus_di: pd.Series, minus_di: pd.Series
    ) -> TrendDirection:
        """
        Determine trend direction based on DI and SMA.

        Args:
            df: DataFrame with OHLC data
            plus_di: +DI Series
            minus_di: -DI Series

        Returns:
            TrendDirection enum value
        """
        close = df["Close"]
        sma = close.rolling(window=self.sma_period).mean()

        # Get latest values
        current_close = close.iloc[-1]
        current_sma = sma.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        # Determine direction
        if current_plus_di > current_minus_di and current_close > current_sma:
            return TrendDirection.UPTREND
        elif current_minus_di > current_plus_di and current_close < current_sma:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS

    def _determine_volatility_regime(self, atr_ratio: pd.Series) -> VolatilityRegime:
        """
        Determine volatility regime based on ATR ratio.

        Args:
            atr_ratio: ATR ratio Series

        Returns:
            VolatilityRegime enum value
        """
        current_ratio = atr_ratio.iloc[-1]

        if pd.isna(current_ratio):
            return VolatilityRegime.NORMAL

        if current_ratio > self.atr_high_mult:
            return VolatilityRegime.HIGH
        elif current_ratio < self.atr_low_mult:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL

    def _determine_market_phase(
        self,
        adx: pd.Series,
        trend_direction: TrendDirection,
        volatility_regime: VolatilityRegime,
        df: pd.DataFrame,
    ) -> MarketPhase:
        """
        Determine overall market phase.

        Args:
            adx: ADX Series
            trend_direction: Current trend direction
            volatility_regime: Current volatility regime
            df: DataFrame with OHLC data

        Returns:
            MarketPhase enum value
        """
        current_adx = adx.iloc[-1]

        if pd.isna(current_adx):
            return MarketPhase.RANGING

        # Strong trend conditions
        if current_adx >= self.adx_strong_threshold:
            if trend_direction == TrendDirection.UPTREND:
                return MarketPhase.STRONG_BULL
            elif trend_direction == TrendDirection.DOWNTREND:
                return MarketPhase.STRONG_BEAR

        # Weak trend conditions
        elif current_adx >= self.adx_trend_threshold:
            if trend_direction == TrendDirection.UPTREND:
                return MarketPhase.WEAK_BULL
            elif trend_direction == TrendDirection.DOWNTREND:
                return MarketPhase.WEAK_BEAR

        # No trend - check volatility
        if volatility_regime == VolatilityRegime.HIGH:
            return MarketPhase.CHOPPY
        else:
            return MarketPhase.RANGING

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeState object with regime information
        """
        # Calculate indicators
        adx, plus_di, minus_di = self._calculate_adx(df)
        atr, atr_ma, atr_ratio = self._calculate_atr_ratio(df)

        # Determine regime components
        trend_direction = self._determine_trend_direction(df, plus_di, minus_di)
        volatility_regime = self._determine_volatility_regime(atr_ratio)
        market_phase = self._determine_market_phase(adx, trend_direction, volatility_regime, df)

        # Get current values
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        current_atr_ratio = atr_ratio.iloc[-1] if not pd.isna(atr_ratio.iloc[-1]) else 1.0

        # Determine trend strength
        if current_adx >= self.adx_strong_threshold:
            trend_strength = current_adx
        elif current_adx >= self.adx_trend_threshold:
            trend_strength = current_adx
        else:
            trend_strength = current_adx

        # Price position relative to SMA
        close = df["Close"]
        sma = close.rolling(window=self.sma_period).mean()
        current_close = close.iloc[-1]
        current_sma = sma.iloc[-1]
        sma_position = "above" if current_close > current_sma else "below"

        return RegimeState(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            volatility_value=current_atr,
            market_phase=market_phase,
            sma_position=sma_position,
            adx=current_adx,
            atr=current_atr,
            atr_ratio=current_atr_ratio,
        )

    def get_pattern_weights(self, regime: RegimeState) -> Dict[str, float]:
        """
        Get pattern category weights based on market regime.

        Args:
            regime: Current RegimeState

        Returns:
            Dictionary of pattern category weights
        """
        weights = {"basic": 1.0, "harmonic": 1.0, "complex": 1.0, "classic": 1.0}

        # Adjust based on trend
        if regime.trend_direction == TrendDirection.SIDEWAYS:
            weights["basic"] = 1.2
            weights["harmonic"] = 0.8
            weights["complex"] = 0.6
        else:
            weights["complex"] = 1.2
            weights["classic"] = 1.0

        # Adjust based on volatility
        if regime.volatility_regime == VolatilityRegime.HIGH:
            weights["basic"] *= 0.7
            weights["harmonic"] *= 0.6
            weights["complex"] *= 0.8
        elif regime.volatility_regime == VolatilityRegime.LOW:
            weights["harmonic"] *= 1.2
            weights["classic"] *= 1.1

        # Adjust based on market phase
        if regime.market_phase == MarketPhase.CHOPPY:
            weights["basic"] *= 0.8
            weights["classic"] *= 0.9
        elif regime.market_phase == MarketPhase.RANGING:
            weights["harmonic"] *= 1.1
            weights["classic"] *= 1.1

        return weights

    def get_regime_multipliers(self, regime: RegimeState) -> Dict[str, float]:
        """
        Get pattern type multipliers based on market regime.

        Args:
            regime: Current RegimeState

        Returns:
            Dictionary of pattern type multipliers
        """
        multipliers = {"reversal": 1.0, "continuation": 1.0, "breakout": 1.0, "counter_trend": 1.0}

        # Adjust based on trend direction and strength
        if regime.trend_direction == TrendDirection.UPTREND:
            if regime.trend_strength >= self.adx_strong_threshold:
                multipliers["continuation"] = 1.3
                multipliers["reversal"] = 0.7
                multipliers["breakout"] = 1.1
            else:
                multipliers["continuation"] = 1.1
                multipliers["reversal"] = 0.9

        elif regime.trend_direction == TrendDirection.DOWNTREND:
            if regime.trend_strength >= self.adx_strong_threshold:
                multipliers["continuation"] = 1.3
                multipliers["reversal"] = 0.7
                multipliers["breakout"] = 1.1
            else:
                multipliers["continuation"] = 1.1
                multipliers["reversal"] = 0.9

        else:  # SIDEWAYS
            multipliers["reversal"] = 1.2
            multipliers["continuation"] = 0.6
            multipliers["breakout"] = 0.8

        # Adjust for volatility
        if regime.volatility_regime == VolatilityRegime.HIGH:
            multipliers["breakout"] *= 1.2
            multipliers["reversal"] *= 0.8
            multipliers["counter_trend"] *= 0.7
        elif regime.volatility_regime == VolatilityRegime.LOW:
            multipliers["breakout"] *= 0.7
            multipliers["reversal"] *= 1.1

        return multipliers

    def get_risk_parameters(self, regime: RegimeState) -> Dict[str, float]:
        """
        Get risk management parameters based on market regime.

        Args:
            regime: Current RegimeState

        Returns:
            Dictionary of risk parameters
        """
        # Base parameters
        params = {
            "risk_per_trade": 0.02,
            "stop_multiplier": 1.0,
            "target_multiplier": 1.0,
            "position_size_mult": 1.0,
        }

        # Adjust based on market phase
        if regime.market_phase == MarketPhase.STRONG_BULL:
            params["risk_per_trade"] = 0.025
            params["stop_multiplier"] = 1.0
            params["target_multiplier"] = 1.5
            params["position_size_mult"] = 1.0

        elif regime.market_phase == MarketPhase.STRONG_BEAR:
            params["risk_per_trade"] = 0.025
            params["stop_multiplier"] = 1.0
            params["target_multiplier"] = 1.5
            params["position_size_mult"] = 1.0

        elif regime.market_phase == MarketPhase.WEAK_BULL:
            params["risk_per_trade"] = 0.015
            params["stop_multiplier"] = 1.25
            params["target_multiplier"] = 1.0
            params["position_size_mult"] = 0.9

        elif regime.market_phase == MarketPhase.WEAK_BEAR:
            params["risk_per_trade"] = 0.015
            params["stop_multiplier"] = 1.25
            params["target_multiplier"] = 1.0
            params["position_size_mult"] = 0.9

        elif regime.market_phase == MarketPhase.RANGING:
            params["risk_per_trade"] = 0.01
            params["stop_multiplier"] = 0.75
            params["target_multiplier"] = 0.75
            params["position_size_mult"] = 0.8

        elif regime.market_phase == MarketPhase.CHOPPY:
            params["risk_per_trade"] = 0.01
            params["stop_multiplier"] = 1.5
            params["target_multiplier"] = 1.25
            params["position_size_mult"] = 0.7

        # Adjust for volatility
        if regime.volatility_regime == VolatilityRegime.HIGH:
            params["risk_per_trade"] *= 0.5
            params["stop_multiplier"] *= 1.5
            params["position_size_mult"] *= 0.75

        elif regime.volatility_regime == VolatilityRegime.LOW:
            params["risk_per_trade"] *= 1.0
            params["stop_multiplier"] *= 0.75
            params["position_size_mult"] *= 1.1

        return params
