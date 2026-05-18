"""
FMZ Strategy Conversions — PineScript/JS → Python Pattern Detectors

7 strategies converted from the FMZ strategies repository (5,807 files)
to native Python pattern detectors integrating with the BasePattern framework.

Detectors:
- AlphaBeast: Supertrend + RSI + Volume triple confirmation
- MultiFactorTrend: SAR + EMA + RSI + ADX quad confirmation
- MomentumZigZag: Momentum-driven ZigZag with force detection (QQE/MACD/MA)
- EMAMACDHF: EMA crossover + MACD confirmation (high-frequency params)
- AdaptiveBollinger: Bollinger Band breakout reversion with 4-layer exit
- AIVolatilityBreakout: Gap fill + VWAP + compression breakout (multi-logic)
"""

from .adaptive_bollinger import AdaptiveBollinger
from .ai_volatility_breakout import AIVolatilityBreakout
from .alpha_beast import AlphaBeast
from .ema_macd_hf import EMAMACDHF
from .momentum_zigzag import MomentumZigZag
from .multi_factor_trend import MultiFactorTrend

__all__ = [
    "AlphaBeast",
    "MultiFactorTrend",
    "MomentumZigZag",
    "EMAMACDHF",
    "AdaptiveBollinger",
    "AIVolatilityBreakout",
]
